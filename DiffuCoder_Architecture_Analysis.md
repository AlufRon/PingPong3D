# DiffuCoder Model Architecture Analysis

## Overview

DiffuCoder is a 7B parameter **masked diffusion large language model (dLLM)** for code generation developed by Apple. It is based on the LLaDA (Large Language Diffusion with mAsking) architecture and trained on 130B tokens of code using the Qwen-2.5-Coder as the base model.

---

## 1. Base Model: Qwen 2.5 Coder

### Architecture
- **Type**: Transformer-based decoder with modifications for diffusion
- **Key Components**:
  - **Grouped Query Attention (GQA)** for efficient KV cache utilization
  - **SwiGLU activation function** for non-linear activation
  - **Rotary Positional Embeddings (RoPE)** for positional information
  - **RMSNorm** for layer normalization
  - **Attention QKV bias** configurable

### Model Specifications (7B variant)
- **Parameters**: 7.61B total (6.53B non-embedding)
- **Layers**: 28 transformer layers
- **Attention Heads**: 28 heads for Q, 4 heads for KV (grouped query attention)
- **Hidden Dimension (d_model)**: Varies by model size
- **Intermediate Size**: 18,944 (for 7B model)
- **Context Length**: 131,072 tokens (full)
- **Vocabulary Size**: 151,646 tokens
- **Training Tokens**: 5.5 trillion tokens

### Special Tokens
- `<|endoftext|>`: End of text/sequence marker
- `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`: Fill-in-the-Middle (FIM) technique
- `<|repo_name|>`: Repository name identifier
- `<|file_sep|>`: File separator for repository-level information
- `<|dlm_pad|>`: Diffusion LM padding token
- `<|im_start|>`, `<|im_end|>`: Instruction markers (for Instruct variant)
- **MASK token ID**: 126336 (critical for diffusion process)
- **EOS token ID**: 126081
- **EoT token ID**: 126348

### Tokenizer
- Inherits from PreTrainedTokenizer
- Treats spaces as parts of tokens
- Word encoding differs based on position (beginning vs. middle of sentence)
- Supports both left and right padding (left padding preferred for generation)

---

## 2. LLaDA/DiffuCoder Architecture

### Key Difference from Autoregressive Models

**Autoregressive (e.g., GPT)**:
- Uses causal mask (attends only to previous tokens)
- Generates text left-to-right sequentially
- Each token depends only on previous tokens

**LLaDA/DiffuCoder (Diffusion)**:
- **No causal mask** - uses bidirectional attention
- Generates entire sequence via iterative denoising
- Can attend to all positions simultaneously
- Allows for parallel generation and non-sequential planning

### Model Structure

```python
LLaDAModel(
    transformer = {
        'wte': Embedding(vocab_size=151646, d_model),  # Token embeddings
        'wpe': Embedding(max_seq_len, d_model),         # Position embeddings (if not using RoPE)
        'emb_drop': Dropout(p=embedding_dropout),
        'blocks': [LLaDABlock × n_layers],              # Transformer blocks
        'ln_f': LayerNorm(d_model),                     # Final layer norm
        'ff_out': Linear(d_model, vocab_size)           # Output projection (if not weight tying)
    }
)
```

### LLaDABlock Structure

Each block contains:
1. **Attention Layer Norm**
2. **Multi-Head Attention** (with Grouped Query Attention)
   - Q: (batch, seq_len, n_heads, head_dim)
   - K, V: (batch, seq_len, n_kv_heads, head_dim)
   - No causal masking (bidirectional)
   - Optional RoPE embeddings
3. **Residual Connection**
4. **Feed-Forward Layer Norm**
5. **Feed-Forward Network**
   - `ff_proj`: Linear(d_model, mlp_hidden_size)
   - `up_proj`: Linear(d_model, mlp_hidden_size) [for SwiGLU]
   - `act`: SwiGLU activation
   - `ff_out`: Linear(mlp_hidden_size, d_model)
6. **Residual Connection**

### Configuration Parameters

```python
ModelConfig(
    d_model=4096,                           # Hidden dimension
    n_heads=28,                             # Query heads
    n_kv_heads=4,                           # Key/Value heads (GQA)
    n_layers=28,                            # Number of blocks
    mlp_ratio=4,                            # MLP expansion ratio
    activation_type='swiglu',               # Activation function
    block_type='sequential',                # Block implementation
    rope=True,                              # Use RoPE embeddings
    rope_theta=10000.0,                     # RoPE base frequency
    flash_attention=True,                   # Use FlashAttention
    attention_dropout=0.0,                  # Attention dropout
    residual_dropout=0.0,                   # Residual dropout
    embedding_dropout=0.0,                  # Embedding dropout
    layer_norm_type='rms',                  # RMSNorm
    max_sequence_length=131072,             # Max context length
    vocab_size=151646,                      # Vocabulary size
    mask_token_id=126336,                   # MASK token for diffusion
    eos_token_id=126081,                    # End of sequence
    weight_tying=True                       # Tie input/output embeddings
)
```

---

## 3. Diffusion Process

### Forward Process (Training - Masking)

**During Pretraining**:
```python
def forward_process(batch, prompt_index, mask_id):
    """
    Randomly mask tokens at ratio t ~ U[0,1]
    """
    target_len = total_length - prompt_length
    k = random.randint(1, target_len + 1)
    
    # Create mask for k random tokens
    is_mask = create_random_mask(k, target_len)
    
    # Mask the tokens
    noisy_batch = where(is_mask, mask_id, batch)
    
    return noisy_batch, (k / target_len)  # Return batch and mask ratio
```

**Key Points**:
- Mask ratio `t ~ U[0,1]` (uniform distribution between 0 and 1)
- During pretraining: mask all tokens randomly
- During SFT: only response tokens may be masked
- Prompt tokens can optionally be masked with probability `p_mask_prompt` (default 0.15)

### Reverse Process (Generation - Denoising)

The generation process iteratively unmasks tokens:

```python
def generate(model, prompt, steps=128, gen_length=128, block_length=128, 
             temperature=0., cfg_scale=0., remasking='low_confidence'):
    """
    Generate by iteratively denoising masked tokens
    
    Args:
        steps: Number of denoising steps
        gen_length: Total tokens to generate
        block_length: Tokens per block (for semi-autoregressive)
        temperature: Gumbel noise temperature
        cfg_scale: Classifier-free guidance scale
        remasking: 'low_confidence' or 'random'
    """
    # Initialize with all masks
    x = full(prompt_len + gen_length, mask_id)
    x[:prompt_len] = prompt
    
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    
    for block_idx in range(num_blocks):
        # Get number of tokens to unmask per step
        num_transfer = get_num_transfer_tokens(mask_index, steps_per_block)
        
        for step in range(steps_per_block):
            # 1. Forward pass through model
            logits = model(x, attention_mask)
            
            # 2. Add Gumbel noise for sampling
            if temperature > 0:
                logits = add_gumbel_noise(logits, temperature)
            
            # 3. Predict tokens
            x0 = argmax(logits, dim=-1)
            
            # 4. Compute confidence scores
            if remasking == 'low_confidence':
                p = softmax(logits, dim=-1)
                confidence = gather(p, index=x0)
            elif remasking == 'random':
                confidence = random_uniform()
            
            # 5. Select top-k most confident predictions to keep
            transfer_index = topk(confidence, k=num_transfer[step])
            x[transfer_index] = x0[transfer_index]
    
    return x
```

**Key Features**:

1. **Linear Noise Schedule**: Equal number of tokens unmasked per step
   ```python
   num_transfer_per_step = total_masked / steps
   ```

2. **Gumbel Max Sampling** (optional):
   ```python
   gumbel_logits = logits.exp() / (-log(uniform_noise)) ** temperature
   ```
   - Uses float64 for numerical stability
   - Temperature=0 → greedy decoding
   - Temperature>0 → stochastic sampling

3. **Remasking Strategies**:
   - **Low Confidence**: Keep most confident predictions
   - **Random**: Random selection of tokens to keep

4. **Semi-Autoregressive Generation**:
   - Can generate in blocks (e.g., block_length=32)
   - Allows partial autoregressive behavior
   - Trade-off between speed and quality

5. **Classifier-Free Guidance** (optional):
   ```python
   # Conditional generation
   logits_cond = model(x)
   
   # Unconditional generation (mask prompts)
   x_uncond = x.clone()
   x_uncond[prompt_index] = mask_id
   logits_uncond = model(x_uncond)
   
   # Guided logits
   logits = logits_uncond + (cfg_scale + 1) * (logits_cond - logits_uncond)
   ```

### Diffusion Parameters

```python
# Training
diffusion_steps: 128              # Number of denoising steps
random_masking: True              # Use random mask seeds per iteration
p_mask_prompt: 0.15               # Probability of masking prompt tokens
generation_temperature: 1.2       # Temperature for generation
generation_batch_size: 10         # Batch size for generation

# Inference (from inference_demo.py)
TOKEN_PER_STEP = 1                # Tokens per diffusion step
steps = 256                       # Total diffusion steps
temperature = 0.3                 # Sampling temperature
top_p = 0.95                      # Nucleus sampling
alg = "entropy"                   # Unmasking algorithm
alg_temp = 0.0                    # Algorithm temperature
```

---

## 4. Training: Coupled-GRPO

### Overview
DiffuCoder uses **Coupled Group Relative Policy Optimization (Coupled-GRPO)** for post-training to improve code generation performance.

### Coupled Sampling Scheme

**Problem**: In diffusion LLMs, per-timestep loss only computes log-probabilities at masked positions, leading to inefficiency and high variance.

**Solution**: Coupled-GRPO introduces complementary masking:

```python
def forward_process(batch, prompt_index, mask_id, seed):
    """
    Create 3 versions of the batch with complementary masks
    """
    b, l = batch.shape
    mask_ratio = random.uniform(0.2, 0.8)  # Random mask ratio
    
    # Create random mask matrix
    random_matrix = rand(b, l)
    
    # Version 1: Mask all completion tokens
    is_mask_v1 = ~prompt_index
    noisy_batch_v1 = where(is_mask_v1, mask_id, batch)
    
    # Version 2: Mask with probability mask_ratio
    is_mask_v2 = ~prompt_index & (random_matrix < mask_ratio)
    noisy_batch_v2 = where(is_mask_v2, mask_id, batch)
    
    # Version 3: Mask complementary tokens (reverse of v2)
    is_mask_v3 = ~prompt_index & (random_matrix > mask_ratio)
    noisy_batch_v3 = where(is_mask_v3, mask_id, batch)
    
    # Weights for each version
    weights = [1, 1/mask_ratio, 1/(1-mask_ratio)]
    
    return [noisy_batch_v1, noisy_batch_v2, noisy_batch_v3], weights, is_mask_v2
```

**Benefits**:
1. Every token's log-probability is computed at least once
2. More accurate probability estimates (evaluated in realistic partially-masked context)
3. Uses 2λ sampling passes (λ=1 typically) with modest computational overhead
4. Significantly improves performance (+4.4% on EvalPlus benchmark)

### Loss Computation

```python
def selective_log_softmax(logits, index, weights, mask):
    """
    Memory-efficient log_softmax -> gather with weighted probabilities
    
    For each sequence, computes:
        p0: Original sequence probability
        p1: Masked sequence probability (mask=True)
        p2: Reverse masked sequence probability (mask=False)
        
    Final: (p0 + weighted_sum(p1, p2)) / 2
    where weighted_sum = weights[1]*p1 + weights[2]*p2
    """
    # Process in chunks to reduce memory
    full_batch_size = logits.size(0) // 3
    per_token_logps = []
    
    for i in range(full_batch_size):
        # Get 3 versions for this sequence
        seq_logits = logits[[i, i+batch_size, i+2*batch_size]]  # [3, seq_len, vocab]
        
        # Compute log probs
        seq_logps = log_softmax(seq_logits, dim=-1)
        seq_per_token_logps = gather(seq_logps, index=labels)  # [3, seq_len]
        
        # Weight and combine
        weighted_logps = where(mask, 
                               seq_per_token_logps[1] * weights[1],
                               seq_per_token_logps[2] * weights[2])
        
        final_logps = (seq_per_token_logps[0] + weighted_logps) / 2
        per_token_logps.append(final_logps)
    
    return stack(per_token_logps)
```

### GRPO Training Loop

```python
class DiffuGRPOTrainer:
    def compute_loss(self, model, inputs):
        # Get masked inputs with random seeds
        prompt_ids, completion_ids = inputs['prompt_ids'], inputs['completion_ids']
        mask_seeds = inputs['mask_seeds']
        
        # Get current iteration mask seed
        this_itr_mask_seed = mask_seeds[this_iteration]
        
        # Compute per-token log probabilities
        per_token_logps = self._get_per_token_logps(
            model, input_ids, logits_to_keep, [this_itr_mask_seed]
        )
        
        # Compute KL divergence with reference model
        if beta != 0:
            ref_per_token_logps = inputs['ref_per_token_logps']
            per_token_kl = exp(ref - current) - (ref - current) - 1
        
        # GRPO loss with clipping
        advantages = inputs['advantages']
        old_logps = inputs['old_per_token_logps']
        
        ratio = exp(per_token_logps - old_logps)
        clipped_ratio = clamp(ratio, 1-epsilon_low, 1+epsilon_high)
        
        loss1 = ratio * advantages
        loss2 = clipped_ratio * advantages
        loss = -min(loss1, loss2)
        
        if beta != 0:
            loss = loss + beta * per_token_kl
        
        return loss.mean()
```

### Reward Functions

DiffuCoder uses multiple reward functions for code generation:

```python
REWARD_FUNCS = {
    'code': code_reward,              # Execute code and check pass rate
    'code_format': code_format_reward, # Check Python syntax validity
    'binary_code': binary_code_reward, # Binary version of code reward
    'accuracy': accuracy_reward,       # Check correctness
    'format': format_reward,           # Check tag formatting
    'tag_count': tag_count_reward,     # Count special tags
}
```

**Code Reward**:
1. Extract code from markdown code blocks
2. Execute code with test cases in sandbox (E2B)
3. Compute pass rate across test cases
4. Return pass_rate as reward

**Code Format Reward**:
1. Check for proper markdown code block formatting
2. Verify Python syntax using `ast.parse()`
3. Return 1.0 if valid, 0.5 if syntax error, 0.0 if format error

---

## 5. Model Usage

### Loading the Model

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "apple/DiffuCoder-7B-Base"

# Load model with trust_remote_code for custom implementation
model = AutoModel.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True  # Required for diffusion architecture
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True
)

# Set padding side to left (important for generation)
tokenizer.padding_side = 'left'

model = model.to("cuda").eval()
```

### Generation

```python
prompt = """
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\"Check if any two numbers are closer than threshold.\"\"\"
"""

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids.to("cuda")
attention_mask = inputs.attention_mask.to("cuda")

# Generate with diffusion
TOKEN_PER_STEP = 1  # 1 token per step (slower but better quality)
max_new_tokens = 256

output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    output_history=True,
    return_dict_in_generate=True,
    steps=max_new_tokens // TOKEN_PER_STEP,  # 256 steps
    temperature=0.2,     # Low for more deterministic
    top_p=0.95,          # Nucleus sampling
    alg="entropy",       # Unmasking algorithm
    alg_temp=0.,         # Algorithm temperature
)

# Decode
generations = tokenizer.decode(
    output.sequences[0, len(input_ids[0]):].tolist(),
    skip_special_tokens=True
)

print(generations.split(tokenizer.eos_token)[0])
```

### Key Generation Parameters

- **steps**: Number of denoising iterations (trade-off: speed vs quality)
  - Fewer steps (e.g., 64) → faster but lower quality
  - More steps (e.g., 256) → slower but better quality
  
- **TOKEN_PER_STEP**: Tokens generated per step
  - TOKEN_PER_STEP=1: Generate 1 token per step (total steps = max_new_tokens)
  - TOKEN_PER_STEP=2: Generate 2 tokens per step (total steps = max_new_tokens/2)
  
- **temperature**: Controls randomness
  - 0: Greedy decoding (deterministic)
  - 0.2-0.4: Low randomness (recommended for code)
  - 1.0+: High randomness (creative but less reliable)

- **alg**: Unmasking algorithm
  - "entropy": Use entropy-based confidence
  - Can affect generation order

---

## 6. Architecture Adaptations for Other Modalities

### Key Insights for Adaptation

1. **Tokenization**:
   - Text: Subword tokenization (BPE)
   - **Images**: Patch-based tokenization (e.g., 16×16 patches)
   - **3D Data (Ping Pong)**: 
     - Spatial tokenization (voxel grids, point clouds)
     - Temporal tokenization (frame sequences)
     - Trajectory tokenization (position + velocity)

2. **Embedding Layer**:
   ```python
   # Text (current)
   embeddings = nn.Embedding(vocab_size, d_model)
   
   # Images (e.g., Vision Transformer style)
   patch_embeddings = nn.Conv2d(3, d_model, kernel_size=16, stride=16)
   
   # 3D Trajectories
   trajectory_embeddings = nn.Linear(trajectory_features, d_model)
   position_embeddings = PositionalEncoding3D(d_model)
   ```

3. **Positional Embeddings**:
   - Text: 1D (RoPE or learned)
   - Images: 2D positional embeddings
   - 3D Data: 3D positional embeddings (x, y, t)

4. **Masking Strategy**:
   - Text: Token-level masking
   - Images: Patch-level masking (MAE-style)
   - 3D Trajectories: Temporal masking, spatial masking, or both

5. **Output Head**:
   ```python
   # Text (current)
   output = nn.Linear(d_model, vocab_size)
   
   # Continuous data (trajectories)
   output = nn.Linear(d_model, action_dim)  # e.g., (x, y, z, vx, vy, vz)
   
   # Images
   output = nn.Conv2d(d_model, num_channels, kernel_size=1)
   ```

### Ping Pong 3D Adaptation Considerations

**Input Representation**:
- Ball position: (x, y, z)
- Ball velocity: (vx, vy, vz)
- Paddle position: (paddle_x, paddle_y)
- Game state: Time, score, etc.

**Possible Tokenization Schemes**:

1. **Discrete Binning**:
   ```python
   # Quantize continuous values to discrete bins
   x_bin = quantize(x, num_bins=100)
   vocab = num_bins ** num_features
   ```

2. **Vector Quantization**:
   ```python
   # Learn codebook of trajectory patterns
   from vector_quantize_pytorch import VectorQuantize
   vq = VectorQuantize(dim=d_model, codebook_size=8192)
   ```

3. **Direct Continuous**:
   ```python
   # Skip tokenization, work directly with continuous values
   trajectory_embedding = MLP(input_dim, d_model)
   ```

**Masking for Trajectories**:
- Temporal masking: Mask future timesteps
- Spatial masking: Mask position or velocity
- Random masking: Mask random features
- Structured masking: Mask contiguous segments

**Training Objective**:
- Predict masked trajectory segments
- Conditional generation given partial trajectories
- Reward: Distance to optimal trajectory, success rate

---

## 7. Key Files Downloaded

### Model Implementation Files
- `llada_model_implementation.py` - Full LLaDA/DiffuCoder model architecture
- `llada_config_implementation.py` - Model configuration classes
- `llada_generation_implementation.py` - Diffusion generation algorithm
- `llada_eval_implementation.py` - Evaluation harness

### Training Files
- `diffucoder_coupled_grpo.py` - Coupled-GRPO trainer implementation
- `diffucoder_configs.py` - Training configuration
- `diffucoder_inference_demo.py` - Interactive inference demo

### Repository Locations
- DiffuCoder: `/home/user/PingPong3D/diffucoder_repo/`
- LLaDA: `/home/user/PingPong3D/llada_repo/`

---

## 8. References

### Papers
- **DiffuCoder**: [Understanding and Improving Masked Diffusion Models for Code Generation](https://arxiv.org/abs/2506.20639)
- **LLaDA**: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
- **Qwen 2.5**: [Qwen Technical Report](https://arxiv.org/abs/2412.15115)
- **Qwen 2.5-Coder**: [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186)

### Repositories
- DiffuCoder: https://github.com/apple/ml-diffucoder
- LLaDA: https://github.com/ML-GSAI/LLaDA
- Qwen: https://github.com/QwenLM/Qwen2.5-Coder

### Models
- apple/DiffuCoder-7B-Base (HuggingFace - access restricted)
- apple/DiffuCoder-7B-Instruct (HuggingFace - access restricted)
- apple/DiffuCoder-7B-cpGRPO (HuggingFace - access restricted)
- GSAI-ML/LLaDA-8B-Base (HuggingFace - public)
- GSAI-ML/LLaDA-8B-Instruct (HuggingFace - public)

---

## Summary

DiffuCoder represents a novel approach to code generation using masked diffusion models instead of traditional autoregressive generation. Key innovations include:

1. **Bidirectional Context**: No causal masking allows global planning
2. **Iterative Refinement**: Generates entire sequences through denoising
3. **Coupled-GRPO**: Efficient post-training with complementary masking
4. **Flexible Generation**: Trade-off between speed (fewer steps) and quality (more steps)

The architecture is highly adaptable to other modalities by modifying:
- Tokenization strategy (discrete, vector quantization, or continuous)
- Embedding layers (appropriate for input modality)
- Positional encodings (1D, 2D, 3D, or custom)
- Masking strategies (temporal, spatial, random, or structured)
- Output heads (discrete tokens vs. continuous values)

For Ping Pong 3D trajectory generation, the diffusion approach offers advantages in:
- Planning long-term trajectories
- Handling uncertainty
- Generating diverse but valid trajectories
- Incorporating physical constraints through masking patterns
