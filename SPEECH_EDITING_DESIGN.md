# Speech Editing with Masked Diffusion: Design Document

## Executive Summary

**Goal**: Enable selective editing of specific words/phrases in speech while preserving surrounding prosody, speaker characteristics, and acoustic continuity.

**Core Innovation**: Selective remasking strategy that only targets edit regions while leveraging full bidirectional context from unedited regions.

**Why This is Novel**: Existing work (MaskGCT, DiSTAR) focuses on generation. NaturalSpeech 3 focuses on zero-shot synthesis. Speech editing with prosody preservation is underexplored.

---

## 1. Problem Statement

**Task**: Given speech sequence S, target region [i, j], and new text T_new:
- Replace tokens S[i:j] with new content matching T_new
- Preserve prosody continuity at boundaries (i-1 → i, j → j+1)
- Maintain speaker characteristics throughout
- Ensure acoustic smoothness (no artifacts at edit boundaries)

**Challenges**:
1. Prosody mismatch at boundaries
2. Speaker characteristic drift in edited region
3. Acoustic discontinuities (clicks, pops)
4. Duration/timing inconsistencies

---

## 2. Architecture Overview

### 2.1 Base Model: LLaDA-Style Bidirectional Transformer

```
Input: [c0_0, c1_0, c2_0, ..., c7_0, c0_1, c1_1, ..., c7_T]
       └─────────────────────────┘  └──────────────────┘
       Frame 0 (8 codebooks)        Frame T (8 codebooks)

Model: Bidirectional Transformer
├─ No causal masking (full attention)
├─ Grouped Query Attention (GQA)
├─ RoPE positional embeddings
├─ RMSNorm + SwiGLU
└─ Output: Logits for each token position

Special tokens:
├─ [MASK]: 2048 (after codebook vocab)
├─ [PAD]: 2049
├─ [EOS]: 2050
└─ [BOUNDARY]: 2051 (NEW - marks edit boundaries)
```

### 2.2 Mimi Codec Integration

**Mimi Specs** (from Moshi):
- 8 codebooks (RVQ structure)
- 2048 codes per codebook
- 12.5 Hz frame rate (80ms per frame)
- Hierarchical: c0 = coarse (semantic), c1-c7 = fine (acoustic)

**Tokenization Strategy**: Delay Pattern (flatten all codebooks)

```python
# Delay pattern for frame-aligned representation
# Frame 0: [c0_0, PAD, PAD, PAD, PAD, PAD, PAD, PAD]
# Frame 1: [c0_1, c1_0, PAD, PAD, PAD, PAD, PAD, PAD]
# Frame 2: [c0_2, c1_1, c2_0, PAD, PAD, PAD, PAD, PAD]
# Frame 3: [c0_3, c1_2, c2_1, c3_0, PAD, PAD, PAD, PAD]
# ...
# Frame 7: [c0_7, c1_6, c2_5, c3_4, c4_3, c5_2, c6_1, c7_0]
# Frame 8+: [c0_t, c1_t-1, c2_t-2, c3_t-3, c4_t-4, c5_t-5, c6_t-6, c7_t-7]

# Total vocab: 2048 * 8 = 16384 tokens + special tokens
# Or simpler: 2048 + special (treat codebook index separately)
```

**Decision**: Use **codebook-agnostic** vocabulary (2048 + special tokens), track codebook index separately in positional encoding.

---

## 3. Selective Remasking Strategy

### 3.1 Edit Region Definition

Given target edit span [i, j] in frame indices:
1. Convert to token indices with delay pattern
2. Add boundary context: [i-B, j+B] where B = boundary frames (e.g., B=4)
3. Mark boundary tokens with [BOUNDARY] for special attention

```python
def define_edit_region(edit_start_frame, edit_end_frame, boundary_frames=4):
    """
    Define edit region with boundary context

    Args:
        edit_start_frame: Starting frame index
        edit_end_frame: Ending frame index
        boundary_frames: Number of frames to include at boundaries

    Returns:
        mask_region: Boolean mask for tokens to regenerate
        boundary_region: Boolean mask for boundary tokens (special attention)
    """
    # Main edit region (all codebooks)
    edit_tokens = get_token_indices(edit_start_frame, edit_end_frame)

    # Boundary regions (for prosody smoothing)
    left_boundary = get_token_indices(
        edit_start_frame - boundary_frames,
        edit_start_frame
    )
    right_boundary = get_token_indices(
        edit_end_frame,
        edit_end_frame + boundary_frames
    )

    mask_region = edit_tokens
    boundary_region = left_boundary + right_boundary

    return mask_region, boundary_region
```

### 3.2 Selective Masking Process

**Key Innovation**: Only mask edit region, keep context fully visible

```python
def selective_mask(tokens, mask_region, mask_ratio):
    """
    Mask only the edit region at given ratio

    Args:
        tokens: [batch, seq_len] token sequence
        mask_region: [seq_len] boolean mask
        mask_ratio: float in [0, 1]

    Returns:
        masked_tokens: Tokens with edit region partially masked
    """
    # Context region: never masked (always visible)
    context_mask = ~mask_region

    # Edit region: mask with probability mask_ratio
    edit_mask_prob = torch.rand(mask_region.sum()) < mask_ratio

    masked_tokens = tokens.clone()
    masked_tokens[mask_region] = torch.where(
        edit_mask_prob,
        MASK_TOKEN_ID,
        tokens[mask_region]
    )

    return masked_tokens
```

### 3.3 Iterative Unmasking with Boundary Awareness

```python
def iterative_unmask_with_boundaries(
    model,
    masked_tokens,
    mask_region,
    boundary_region,
    num_steps=12
):
    """
    Iteratively unmask edit region with boundary smoothing

    Key: Unmask from boundaries inward (progressive filling)
    """
    current_tokens = masked_tokens.clone()

    for step in range(num_steps):
        # Forward pass
        logits = model(current_tokens)

        # Get confidence scores
        confidence = logits.softmax(dim=-1).max(dim=-1).values

        # Unmask schedule: prioritize boundaries first
        mask_ratio = 1.0 - (step + 1) / num_steps

        # Boundary boost: unmask boundaries earlier
        boundary_boost = 0.2 if step < num_steps // 3 else 0.0
        confidence[boundary_region] += boundary_boost

        # Select tokens to unmask
        num_to_unmask = int((1 - mask_ratio) * mask_region.sum())
        masked_positions = (current_tokens == MASK_TOKEN_ID) & mask_region

        # Sort by confidence, unmask top-k
        confidence_masked = confidence.clone()
        confidence_masked[~masked_positions] = -float('inf')
        topk_indices = confidence_masked.argsort(descending=True)[:num_to_unmask]

        # Unmask selected positions
        predicted_tokens = logits.argmax(dim=-1)
        current_tokens[topk_indices] = predicted_tokens[topk_indices]

    return current_tokens
```

---

## 4. Prosody Preservation Mechanisms

### 4.1 Boundary Continuity Loss

**Goal**: Ensure prosody features (pitch, energy, duration) are smooth at edit boundaries

```python
def boundary_continuity_loss(
    generated_tokens,
    original_tokens,
    boundary_region,
    mimi_encoder
):
    """
    Compute continuity loss at edit boundaries

    Measures acoustic similarity between:
    - Left boundary: original vs generated
    - Right boundary: original vs generated
    """
    # Decode to acoustic features (mel or waveform)
    gen_audio = mimi_encoder.decode(generated_tokens)
    orig_audio = mimi_encoder.decode(original_tokens)

    # Extract prosody features at boundaries
    gen_prosody = extract_prosody_features(gen_audio, boundary_region)
    orig_prosody = extract_prosody_features(orig_audio, boundary_region)

    # Compute similarity loss (cosine or L2)
    loss = F.mse_loss(gen_prosody, orig_prosody)

    return loss
```

### 4.2 Speaker Consistency Loss

**Goal**: Maintain speaker characteristics in edited region

```python
def speaker_consistency_loss(
    generated_tokens,
    context_tokens,
    speaker_encoder
):
    """
    Ensure edited region matches context speaker embedding

    Uses pre-trained speaker encoder (e.g., from Moshi)
    """
    # Extract speaker embeddings
    gen_speaker_emb = speaker_encoder(generated_tokens)
    ctx_speaker_emb = speaker_encoder(context_tokens)

    # Cosine similarity loss
    loss = 1 - F.cosine_similarity(gen_speaker_emb, ctx_speaker_emb, dim=-1)

    return loss.mean()
```

### 4.3 Duration Consistency

**Goal**: Match duration of edited region to expected duration from text

```python
def duration_loss(generated_tokens, target_duration_frames):
    """
    Penalize duration mismatch

    Uses learned duration predictor from text
    """
    actual_duration = generated_tokens.size(1) // 8  # Divide by codebooks
    duration_diff = abs(actual_duration - target_duration_frames)

    return duration_diff * 0.01  # Small weight
```

---

## 5. Training Strategy

### 5.1 Data Preparation

**Dataset**: LibriSpeech, VCTK, or any speech with transcripts

**Edit Simulation**:
1. Take original speech sequence S
2. Randomly select edit span [i, j]
3. Replace with different text T_new from same speaker
4. Create training pair: (S with masked [i:j], target tokens)

```python
def create_edit_training_data(speech_tokens, transcript, speaker_id):
    """
    Simulate edit operation for training

    Returns:
        input_tokens: Original with masked edit region
        target_tokens: Ground truth for edit region
        edit_region_mask: Boolean mask
        boundary_mask: Boolean mask for boundaries
    """
    # Select random span
    seq_len = speech_tokens.size(0)
    edit_len = random.randint(seq_len // 10, seq_len // 4)
    edit_start = random.randint(0, seq_len - edit_len)
    edit_end = edit_start + edit_len

    # Mask edit region
    input_tokens = speech_tokens.clone()
    mask_ratio = random.uniform(0.2, 0.8)
    input_tokens = selective_mask(input_tokens, edit_start, edit_end, mask_ratio)

    # Define regions
    edit_region_mask = torch.zeros(seq_len, dtype=torch.bool)
    edit_region_mask[edit_start:edit_end] = True

    boundary_mask = torch.zeros(seq_len, dtype=torch.bool)
    boundary_mask[max(0, edit_start-4):edit_start] = True
    boundary_mask[edit_end:min(seq_len, edit_end+4)] = True

    return input_tokens, speech_tokens, edit_region_mask, boundary_mask
```

### 5.2 Training Objective

**Multi-Component Loss**:

```python
def training_loss(model, batch):
    """
    Combined loss for speech editing

    Components:
    1. Token prediction loss (standard cross-entropy)
    2. Boundary continuity loss (prosody smoothing)
    3. Speaker consistency loss (maintain speaker ID)
    4. Duration consistency loss (match expected duration)
    """
    input_tokens, target_tokens, edit_mask, boundary_mask = batch

    # Forward pass
    logits = model(input_tokens)

    # 1. Token prediction loss (only on edit region)
    token_loss = F.cross_entropy(
        logits[edit_mask],
        target_tokens[edit_mask]
    )

    # 2. Boundary continuity
    predicted_tokens = logits.argmax(dim=-1)
    boundary_loss = boundary_continuity_loss(
        predicted_tokens,
        target_tokens,
        boundary_mask,
        mimi_encoder
    )

    # 3. Speaker consistency
    speaker_loss = speaker_consistency_loss(
        predicted_tokens[edit_mask],
        target_tokens[~edit_mask],  # Context
        speaker_encoder
    )

    # 4. Combined loss
    total_loss = (
        token_loss +
        0.1 * boundary_loss +
        0.05 * speaker_loss
    )

    return total_loss
```

### 5.3 Training Schedule

**Phase 1: Token Prediction (3-5 epochs)**
- Focus on accurate token prediction
- Weight: token_loss = 1.0, others = 0.0

**Phase 2: Boundary Smoothing (3-5 epochs)**
- Add boundary continuity loss
- Weight: token_loss = 1.0, boundary_loss = 0.1

**Phase 3: Full Objective (5-10 epochs)**
- All loss components active
- Fine-tune for prosody and speaker consistency

---

## 6. Model Configuration

### 6.1 Architecture Specs

```python
class SpeechEditConfig:
    # Vocabulary
    vocab_size = 2048           # Mimi codebook size
    num_codebooks = 8           # RVQ codebooks
    mask_token_id = 2048
    pad_token_id = 2049
    eos_token_id = 2050
    boundary_token_id = 2051

    # Model architecture
    d_model = 2048              # Smaller than LLaDA (8B → ~1B)
    n_layers = 24               # Moderate depth
    n_heads = 16                # Multi-head attention
    n_kv_heads = 4              # GQA for efficiency
    d_ff = 5504                 # SwiGLU FFN

    # Sequence length
    max_seq_len = 2048          # ~200 frames * 8 codebooks

    # Training
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 0.01
    warmup_steps = 2000
    max_steps = 100000

    # Editing
    boundary_frames = 4         # Frames for boundary smoothing
    num_unmask_steps = 12       # Iterative unmasking steps
```

### 6.2 Minimal Changes from LLaDA

**Changes needed**:
1. ✅ vocab_size: 126336 → 2048 (+ special tokens)
2. ✅ d_model: 4096 → 2048 (reduce size)
3. ✅ n_layers: 32 → 24 (reduce depth)
4. ✅ Add boundary_token_id and boundary handling
5. ✅ Modify forward pass for selective masking
6. ✅ Add prosody/speaker loss modules

**Code changes**: ~200 lines total
- Config changes: ~10 lines
- Selective masking: ~50 lines
- Boundary handling: ~40 lines
- Loss modules: ~100 lines

---

## 7. Evaluation Metrics

### 7.1 Objective Metrics

**Edit Quality**:
- Word Error Rate (WER) on edited region
- Phoneme Error Rate (PER) on edited region

**Prosody Preservation**:
- Pitch continuity at boundaries (MAE in Hz)
- Energy continuity at boundaries (MAE in dB)
- Speaking rate consistency (frames/phoneme)

**Speaker Consistency**:
- Speaker verification accuracy (cosine similarity)
- Equal Error Rate (EER) for speaker ID

**Acoustic Quality**:
- MOS (Mean Opinion Score) via human evaluation
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)

### 7.2 Subjective Metrics

**A/B Testing**:
- Naturalness: edited vs original
- Prosody smoothness: edit boundaries
- Speaker consistency: edited region vs context

**MOS Evaluation** (1-5 scale):
- Overall quality
- Naturalness of edit
- Boundary smoothness
- Speaker consistency

---

## 8. Novelty Analysis

### 8.1 What Makes This Novel?

| Aspect | Prior Work | This Work |
|--------|-----------|-----------|
| **Task** | Generation (MaskGCT, DiSTAR, NaturalSpeech 3) | **Selective editing** |
| **Masking** | Full sequence or 2-stage | **Targeted region only** |
| **Context** | Limited or none | **Full bidirectional context** |
| **Boundaries** | Not addressed | **Explicit prosody smoothing** |
| **Use Case** | Zero-shot TTS, voice cloning | **Post-production editing** |

### 8.2 Key Contributions

1. **Selective Remasking**: Only mask edit region, preserve context
2. **Boundary-Aware Unmasking**: Prioritize boundaries for smooth transitions
3. **Multi-Component Loss**: Token + boundary + speaker consistency
4. **Edit-Specific Architecture**: Boundary tokens, region-specific attention

### 8.3 Comparison to Related Work

**FluentEditor (2023)**:
- Uses diffusion for speech editing
- Continuous space (mel-spectrograms)
- Our work: Discrete tokens (higher quality via neural codecs)

**A3T (2023)**:
- Token-based editing with autoregressive models
- Left-to-right generation (slow, error accumulation)
- Our work: Non-autoregressive (parallel, faster)

**CampNet (2024)**:
- Context-aware masked prediction for TTS
- Generation-focused, not editing
- Our work: Explicit edit operations with boundary handling

**Conclusion**: Speech editing with masked diffusion on discrete codes + boundary prosody preservation is **novel and publishable**.

---

## 9. Publication Strategy

### 9.1 Target Venues

**Top Tier** (⭐⭐⭐⭐):
- ICASSP 2026 (deadline: Oct 2025)
- Interspeech 2026 (deadline: Mar 2026)
- NeurIPS 2025 (deadline: May 2025) - if we get strong results

**Backup** (⭐⭐⭐):
- ICML 2026 Workshop on Generative Models
- NeurIPS 2025 Workshop on Audio Imagination
- SLT 2026 (Spoken Language Technology)

### 9.2 Positioning

**Title Options**:
1. "MaskEdit: Selective Speech Editing with Masked Diffusion Models"
2. "Boundary-Aware Speech Editing via Discrete Masked Diffusion"
3. "Context-Aware Speech Editing with Selective Remasking"

**Key Messages**:
- First to apply masked diffusion specifically for speech editing
- Novel selective remasking strategy preserves context
- Explicit boundary prosody smoothing for natural edits
- Strong results on objective and subjective metrics

### 9.3 Baseline Comparisons

**Must compare against**:
1. FluentEditor (continuous diffusion editing)
2. A3T (autoregressive token editing)
3. CampNet (masked prediction TTS)
4. Simple baselines: copy-paste, VALL-E editing

**Expected advantages**:
- Better boundary smoothness vs copy-paste
- Faster inference vs A3T (non-autoregressive)
- Higher quality vs FluentEditor (discrete codes)
- More controllable vs end-to-end approaches

---

## 10. Implementation Roadmap

### Phase 1: Core Model (Week 1-2)
- ✅ Implement SpeechEditTransformer (LLaDA-based)
- ✅ Integrate Mimi tokenizer with delay pattern
- ✅ Add selective masking logic
- ✅ Implement boundary token handling

### Phase 2: Training Pipeline (Week 3-4)
- ✅ Create edit simulation data loader
- ✅ Implement multi-component loss
- ✅ Set up training loop with phase scheduling
- ✅ Add logging and checkpointing

### Phase 3: Evaluation (Week 5-6)
- ✅ Implement objective metrics (WER, pitch/energy continuity)
- ✅ Integrate speaker encoder for consistency metrics
- ✅ Set up A/B testing framework
- ✅ Run baseline comparisons

### Phase 4: Experiments & Paper (Week 7-10)
- ✅ Train on LibriSpeech + VCTK
- ✅ Ablation studies (boundary loss, speaker loss, num steps)
- ✅ Qualitative analysis with audio examples
- ✅ Write paper draft

### Phase 5: Submission (Week 11-12)
- ✅ Prepare demo page with audio samples
- ✅ Finalize paper writing
- ✅ Submit to target venue
- ✅ Release code on GitHub

---

## 11. Risk Mitigation

### Risk 1: Boundary Artifacts
**Mitigation**:
- Boundary continuity loss with high weight
- Longer boundary regions (6-8 frames instead of 4)
- Post-processing smoothing (optional)

### Risk 2: Speaker Drift
**Mitigation**:
- Strong speaker encoder (pre-trained from Moshi)
- Speaker consistency loss with higher weight
- Data augmentation with same-speaker edits

### Risk 3: Duration Mismatch
**Mitigation**:
- Explicit duration predictor
- Allow flexible edit lengths
- Duration-aware masking schedule

### Risk 4: Insufficient Novelty
**Mitigation**:
- Emphasize selective remasking as key innovation
- Strong empirical results on boundary smoothness
- Clear practical applications (post-production, accessibility)

---

## 12. Next Steps

**Immediate** (Today):
1. Implement SpeechEditTransformer model
2. Set up Mimi tokenizer integration
3. Create selective masking functions

**Short-term** (This Week):
1. Implement training data loader
2. Set up training pipeline
3. Run initial experiments on small dataset

**Medium-term** (Next 2-4 Weeks):
1. Full training on LibriSpeech
2. Evaluate with objective metrics
3. Prepare demo samples

**Long-term** (Next 2-3 Months):
1. Write paper draft
2. Submit to ICASSP 2026
3. Open-source release

---

## Conclusion

Speech editing with masked diffusion is:
- ✅ **Novel**: Not done before with discrete codes + boundary awareness
- ✅ **Practical**: Clear use case in post-production, accessibility
- ✅ **Feasible**: Minimal changes from LLaDA (~200 lines)
- ✅ **Publishable**: Strong positioning for ICASSP/Interspeech

**Publication probability**: ⭐⭐⭐⭐ 70-80% for top-tier venues

Let's build it.
