"""
Prosody Preservation Losses for Speech Editing

Key losses for ensuring high-quality edits:
1. Boundary Continuity Loss: Smooth prosody at edit boundaries
2. Speaker Consistency Loss: Maintain speaker identity in edited region
3. Duration Consistency Loss: Match expected duration from text

These losses are the key innovation for speech editing quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BoundaryContinuityLoss(nn.Module):
    """
    Boundary Continuity Loss

    Ensures prosody features (pitch, energy, spectral features) are smooth
    at edit boundaries by comparing generated vs original features.

    This is critical for natural-sounding edits without audible artifacts.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        boundary_weight: float = 2.0,
    ):
        """
        Args:
            feature_dim: Dimension of prosody features
            boundary_weight: Extra weight for boundary tokens
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.boundary_weight = boundary_weight

        # Feature extractor (could be pre-trained or learned)
        # For now, simple linear projection
        # In practice: use pre-trained prosody encoder
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Compressed prosody features
        )

    def forward(
        self,
        generated_features: torch.Tensor,
        original_features: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute boundary continuity loss

        Args:
            generated_features: [batch, seq_len, feature_dim] from generated tokens
            original_features: [batch, seq_len, feature_dim] from original tokens
            boundary_mask: [batch, seq_len] boolean mask (True = boundary region)

        Returns:
            loss: scalar loss value
        """
        batch_size, seq_len, _ = generated_features.shape

        # Extract prosody features
        gen_prosody = self.feature_extractor(generated_features)
        orig_prosody = self.feature_extractor(original_features)

        # Compute MSE loss
        mse_loss = F.mse_loss(gen_prosody, orig_prosody, reduction='none')
        mse_loss = mse_loss.mean(dim=-1)  # [batch, seq_len]

        # Apply boundary weighting
        weights = torch.ones_like(boundary_mask, dtype=torch.float)
        weights[boundary_mask] = self.boundary_weight

        # Weighted loss
        weighted_loss = mse_loss * weights

        # Average over batch and sequence
        loss = weighted_loss.sum() / weights.sum()

        return loss


class TokenBoundaryContinuityLoss(nn.Module):
    """
    Simplified boundary continuity loss operating directly on tokens

    Instead of extracting acoustic features, this compares token embeddings
    at boundaries. More efficient but less precise than full acoustic loss.
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        boundary_weight: float = 2.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.boundary_weight = boundary_weight

    def forward(
        self,
        generated_embeddings: torch.Tensor,
        original_embeddings: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute boundary continuity loss on token embeddings

        Args:
            generated_embeddings: [batch, seq_len, d_model] from model
            original_embeddings: [batch, seq_len, d_model] from original tokens
            boundary_mask: [batch, seq_len] boolean mask

        Returns:
            loss: scalar loss value
        """
        # Cosine similarity loss (1 - cosine similarity)
        cos_sim = F.cosine_similarity(
            generated_embeddings,
            original_embeddings,
            dim=-1,
        )  # [batch, seq_len]

        loss = 1.0 - cos_sim  # [batch, seq_len]

        # Apply boundary weighting
        weights = torch.ones_like(boundary_mask, dtype=torch.float)
        weights[boundary_mask] = self.boundary_weight

        # Weighted loss
        weighted_loss = loss * weights

        # Average
        return weighted_loss.sum() / weights.sum()


class SpeakerConsistencyLoss(nn.Module):
    """
    Speaker Consistency Loss

    Ensures edited region maintains same speaker characteristics as context.
    Uses a pre-trained speaker encoder to extract speaker embeddings.
    """

    def __init__(
        self,
        speaker_encoder: Optional[nn.Module] = None,
        embedding_dim: int = 256,
    ):
        """
        Args:
            speaker_encoder: Pre-trained speaker encoder (e.g., from Moshi)
            embedding_dim: Dimension of speaker embeddings
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        if speaker_encoder is None:
            # Placeholder - in practice use pre-trained encoder
            print("⚠️  Using placeholder speaker encoder.")
            print("    In practice, use pre-trained encoder (e.g., from Moshi)")
            self.speaker_encoder = self._create_placeholder_encoder()
        else:
            self.speaker_encoder = speaker_encoder

        # Freeze speaker encoder
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False

    def _create_placeholder_encoder(self) -> nn.Module:
        """Create simple placeholder speaker encoder"""
        return nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.embedding_dim),
        )

    def forward(
        self,
        generated_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute speaker consistency loss

        Args:
            generated_tokens: [batch, edit_len, d_model] embeddings of edited region
            context_tokens: [batch, context_len, d_model] embeddings of context

        Returns:
            loss: scalar loss value
        """
        # Extract speaker embeddings (average pooling)
        gen_spk_emb = self.speaker_encoder(generated_tokens.mean(dim=1))
        ctx_spk_emb = self.speaker_encoder(context_tokens.mean(dim=1))

        # Cosine similarity loss
        cos_sim = F.cosine_similarity(gen_spk_emb, ctx_spk_emb, dim=-1)
        loss = 1.0 - cos_sim  # [batch]

        return loss.mean()


class DurationConsistencyLoss(nn.Module):
    """
    Duration Consistency Loss

    Ensures edited region has appropriate duration based on content.
    Uses simple L1 loss on frame count mismatch.
    """

    def __init__(self, weight: float = 0.01):
        """
        Args:
            weight: Loss weight (usually small)
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        predicted_duration_frames: torch.Tensor,
        target_duration_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute duration consistency loss

        Args:
            predicted_duration_frames: [batch] predicted durations
            target_duration_frames: [batch] target durations

        Returns:
            loss: scalar loss value
        """
        # L1 loss on duration
        duration_diff = torch.abs(
            predicted_duration_frames.float() - target_duration_frames.float()
        )

        return self.weight * duration_diff.mean()


class CombinedEditingLoss(nn.Module):
    """
    Combined loss for speech editing

    Integrates:
    1. Token prediction loss (cross-entropy)
    2. Boundary continuity loss
    3. Speaker consistency loss
    4. Duration consistency loss (optional)

    This multi-component loss is key to high-quality speech editing.
    """

    def __init__(
        self,
        vocab_size: int = 2052,
        d_model: int = 2048,
        boundary_loss_weight: float = 0.1,
        speaker_loss_weight: float = 0.05,
        duration_loss_weight: float = 0.01,
        use_token_boundary_loss: bool = True,
    ):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            boundary_loss_weight: Weight for boundary continuity
            speaker_loss_weight: Weight for speaker consistency
            duration_loss_weight: Weight for duration consistency
            use_token_boundary_loss: If True, use token-based boundary loss (faster)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.boundary_loss_weight = boundary_loss_weight
        self.speaker_loss_weight = speaker_loss_weight
        self.duration_loss_weight = duration_loss_weight

        # Boundary loss
        if use_token_boundary_loss:
            self.boundary_loss = TokenBoundaryContinuityLoss(
                embedding_dim=d_model,
                boundary_weight=2.0,
            )
        else:
            self.boundary_loss = BoundaryContinuityLoss(
                feature_dim=d_model,
                boundary_weight=2.0,
            )

        # Speaker loss
        self.speaker_loss = SpeakerConsistencyLoss(
            speaker_encoder=None,  # Placeholder
            embedding_dim=256,
        )

        # Duration loss
        self.duration_loss = DurationConsistencyLoss(
            weight=duration_loss_weight,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target_tokens: torch.Tensor,
        generated_embeddings: torch.Tensor,
        original_embeddings: torch.Tensor,
        edit_mask: torch.Tensor,
        boundary_mask: torch.Tensor,
        context_mask: torch.Tensor,
        target_duration: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss

        Args:
            logits: [batch, seq_len, vocab_size] model output
            target_tokens: [batch, seq_len] ground truth tokens
            generated_embeddings: [batch, seq_len, d_model] from model
            original_embeddings: [batch, seq_len, d_model] from original
            edit_mask: [batch, seq_len] boolean mask for edit region
            boundary_mask: [batch, seq_len] boolean mask for boundaries
            context_mask: [batch, seq_len] boolean mask for context
            target_duration: [batch] target duration in frames (optional)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses for logging
        """
        batch_size, seq_len, _ = logits.shape

        # 1. Token prediction loss (only on edit region)
        token_loss = F.cross_entropy(
            logits[edit_mask].reshape(-1, self.vocab_size),
            target_tokens[edit_mask].reshape(-1),
            reduction='mean',
        )

        # 2. Boundary continuity loss
        boundary_cont_loss = self.boundary_loss(
            generated_embeddings,
            original_embeddings,
            boundary_mask,
        )

        # 3. Speaker consistency loss
        # Extract edit and context embeddings
        edit_embeddings = []
        context_embeddings = []

        for b in range(batch_size):
            edit_emb = generated_embeddings[b, edit_mask[b], :]
            ctx_emb = original_embeddings[b, context_mask[b], :]

            if edit_emb.size(0) > 0 and ctx_emb.size(0) > 0:
                edit_embeddings.append(edit_emb.unsqueeze(0))
                context_embeddings.append(ctx_emb.unsqueeze(0))

        if len(edit_embeddings) > 0:
            edit_embeddings = torch.cat(edit_embeddings, dim=0)
            context_embeddings = torch.cat(context_embeddings, dim=0)

            speaker_cons_loss = self.speaker_loss(
                edit_embeddings,
                context_embeddings,
            )
        else:
            speaker_cons_loss = torch.tensor(0.0, device=logits.device)

        # 4. Duration consistency loss (optional)
        if target_duration is not None:
            predicted_duration = edit_mask.sum(dim=1) // 8  # Convert to frames
            duration_cons_loss = self.duration_loss(
                predicted_duration,
                target_duration,
            )
        else:
            duration_cons_loss = torch.tensor(0.0, device=logits.device)

        # Combined loss
        total_loss = (
            token_loss +
            self.boundary_loss_weight * boundary_cont_loss +
            self.speaker_loss_weight * speaker_cons_loss +
            duration_cons_loss  # Already weighted in duration_loss
        )

        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'token': token_loss.item(),
            'boundary': boundary_cont_loss.item(),
            'speaker': speaker_cons_loss.item(),
            'duration': duration_cons_loss.item(),
        }

        return total_loss, loss_dict


# Example usage
if __name__ == "__main__":
    print("Testing prosody preservation losses...")

    batch_size = 2
    seq_len = 400
    d_model = 2048
    vocab_size = 2052

    # Create dummy data
    logits = torch.randn(batch_size, seq_len, vocab_size)
    target_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    generated_embeddings = torch.randn(batch_size, seq_len, d_model)
    original_embeddings = torch.randn(batch_size, seq_len, d_model)

    # Create masks
    edit_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    edit_mask[:, 100:200] = True

    boundary_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    boundary_mask[:, 96:100] = True  # Left boundary
    boundary_mask[:, 200:204] = True  # Right boundary

    context_mask = ~edit_mask

    target_duration = torch.tensor([12, 13], dtype=torch.long)  # In frames

    # Create combined loss
    combined_loss = CombinedEditingLoss(
        vocab_size=vocab_size,
        d_model=d_model,
        boundary_loss_weight=0.1,
        speaker_loss_weight=0.05,
        duration_loss_weight=0.01,
    )

    # Compute loss
    total_loss, loss_dict = combined_loss(
        logits=logits,
        target_tokens=target_tokens,
        generated_embeddings=generated_embeddings,
        original_embeddings=original_embeddings,
        edit_mask=edit_mask,
        boundary_mask=boundary_mask,
        context_mask=context_mask,
        target_duration=target_duration,
    )

    print(f"\nLoss breakdown:")
    for k, v in loss_dict.items():
        print(f"  {k:10s}: {v:.4f}")

    print(f"\n✓ Loss computation successful")
