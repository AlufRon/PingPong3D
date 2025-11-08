"""
Speech Editing Model with Masked Diffusion

Based on LLaDA's bidirectional transformer, adapted for speech editing:
- Bidirectional attention (no causal masking)
- Grouped Query Attention (GQA) for efficiency
- RoPE positional embeddings
- RMSNorm + SwiGLU activation
- Boundary-aware processing for prosody preservation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SpeechEditConfig:
    """Configuration for Speech Editing Model"""

    # Vocabulary
    vocab_size: int = 2052  # 2048 + 4 special tokens
    mask_token_id: int = 2048
    pad_token_id: int = 2049
    eos_token_id: int = 2050
    boundary_token_id: int = 2051

    # Model architecture (smaller than LLaDA 8B)
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4  # GQA: fewer KV heads
    d_ff: int = 5504  # FFN hidden size (2.7x d_model)
    max_seq_len: int = 2048

    # Dropout
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # RoPE
    rope_theta: float = 10000.0

    # Training
    vocab_parallel: bool = False

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads {self.n_heads} must be divisible by n_kv_heads {self.n_kv_heads}"


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE)

    Applies rotational position encoding to queries and keys
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for max sequence length
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, dim]

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, n_heads, head_dim]
            seq_len: sequence length

        Returns:
            cos, sin: [1, seq_len, 1, head_dim] for broadcasting
        """
        return (
            self.cos_cached[:seq_len].view(1, seq_len, 1, self.dim),
            self.sin_cached[:seq_len].view(1, seq_len, 1, self.dim),
        )


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to queries and keys

    Args:
        q: [batch, seq_len, n_heads, head_dim]
        k: [batch, seq_len, n_kv_heads, head_dim]
        cos, sin: [1, seq_len, 1, head_dim]

    Returns:
        q_rot, k_rot: rotated queries and keys
    """
    # Rotate first half and second half
    q_half1, q_half2 = q.chunk(2, dim=-1)
    k_half1, k_half2 = k.chunk(2, dim=-1)

    cos = cos[..., :q.size(-1)]  # Match head_dim
    sin = sin[..., :q.size(-1)]

    cos_half1, cos_half2 = cos.chunk(2, dim=-1)
    sin_half1, sin_half2 = sin.chunk(2, dim=-1)

    q_rot = torch.cat([
        q_half1 * cos_half1 - q_half2 * sin_half1,
        q_half2 * cos_half2 + q_half1 * sin_half2,
    ], dim=-1)

    k_rot = torch.cat([
        k_half1 * cos_half1 - k_half2 * sin_half1,
        k_half2 * cos_half2 + k_half1 * sin_half2,
    ], dim=-1)

    return q_rot, k_rot


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)

    More efficient than MHA: fewer KV heads, shared across query heads
    """

    def __init__(self, config: SpeechEditConfig):
        super().__init__()
        self.config = config

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # Repetition factor

        # Linear projections
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        self.attention_dropout = config.attention_dropout

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len] (1 = attend, 0 = mask out)

        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        cos, sin = self.rotary_emb(q, seq_len)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Repeat KV heads to match Q heads (GQA)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # Transpose for attention: [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] → [batch, 1, 1, seq_len]
            mask = attention_mask.view(batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [batch, n_heads, seq_len, head_dim]

        # Transpose back and combine heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(output)

        return output


class SwiGLU(nn.Module):
    """
    SwiGLU activation function

    FFN(x) = (Swish(W1 * x) ⊙ W2 * x) * W3
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    """
    Transformer block with:
    - Grouped Query Attention
    - SwiGLU FFN
    - RMSNorm
    """

    def __init__(self, config: SpeechEditConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLU(config.d_model, config.d_ff)

        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len]

        Returns:
            output: [batch, seq_len, d_model]
        """
        # Attention with residual
        h = x + F.dropout(
            self.attention(self.attention_norm(x), attention_mask),
            p=self.dropout,
            training=self.training,
        )

        # FFN with residual
        out = h + F.dropout(
            self.feed_forward(self.ffn_norm(h)),
            p=self.dropout,
            training=self.training,
        )

        return out


class SpeechEditModel(nn.Module):
    """
    Speech Editing Model with Masked Diffusion

    Architecture:
    - Token embeddings
    - Stack of bidirectional transformer blocks
    - Output projection to vocabulary

    Key features:
    - Bidirectional attention (no causal masking)
    - Boundary-aware processing
    - Selective masking for edit regions
    """

    def __init__(self, config: SpeechEditConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        # Output normalization and projection
        self.norm = RMSNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie input and output embeddings (optional, saves parameters)
        # self.output.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] attention mask (1 = attend, 0 = ignore)

        Returns:
            logits: [batch, seq_len, vocab_size] output logits
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]

        # Create attention mask if not provided (attend to all)
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len),
                dtype=torch.long,
                device=input_ids.device,
            )

        # Transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Output normalization and projection
        x = self.norm(x)
        logits = self.output(x)  # [batch, seq_len, vocab_size]

        return logits

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get number of parameters

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params


# Example usage
if __name__ == "__main__":
    # Create config
    config = SpeechEditConfig(
        vocab_size=2052,
        d_model=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=2048,
    )

    # Create model
    model = SpeechEditModel(config)

    # Count parameters
    n_params = model.get_num_params(non_embedding=False)
    n_params_no_emb = model.get_num_params(non_embedding=True)

    print(f"Speech Edit Model:")
    print(f"  Total parameters: {n_params:,} ({n_params / 1e9:.2f}B)")
    print(f"  Non-embedding parameters: {n_params_no_emb:,} ({n_params_no_emb / 1e9:.2f}B)")
    print(f"  Layers: {config.n_layers}")
    print(f"  Model dim: {config.d_model}")
    print(f"  Heads: {config.n_heads} (KV heads: {config.n_kv_heads})")
    print(f"  Max sequence length: {config.max_seq_len}")

    # Test forward pass
    batch_size = 2
    seq_len = 400  # 50 frames * 8 codebooks

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    print(f"\nTest forward pass:")
    print(f"  Input shape: {input_ids.shape}")

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    print(f"  Output shape: {logits.shape}")
    print(f"  ✓ Forward pass successful")

    # Test with masking
    mask_token_id = config.mask_token_id
    input_ids_masked = input_ids.clone()
    input_ids_masked[:, 100:200] = mask_token_id  # Mask middle section

    with torch.no_grad():
        logits_masked = model(input_ids_masked, attention_mask)

    print(f"\nTest with masked input:")
    print(f"  Masked tokens: 100 out of {seq_len}")
    print(f"  Output shape: {logits_masked.shape}")
    print(f"  ✓ Masked forward pass successful")
