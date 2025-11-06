# Coupled-GRPO: Deep Algorithm Analysis

## Executive Summary

**Coupled-GRPO** (Coupled Group Relative Policy Optimization) is a novel reinforcement learning algorithm designed specifically for masked diffusion language models (dLLMs). It addresses the high variance problem in token log-likelihood estimation during GRPO training by using **complementary mask sampling** based on **antithetic variates** theory.

**Key Innovation**: Instead of sampling masks independently, Coupled-GRPO samples pairs of complementary masks where each token is evaluated exactly once per pair, dramatically reducing variance while maintaining full token coverage.

---

## Problem Statement

### Challenge in dLLM Reinforcement Learning

Unlike autoregressive (AR) models where token probabilities are directly computed:
- **AR Models**: `π(token_k | tokens_1...k-1)` is computed in one forward pass
- **Diffusion Models**: Token probabilities require Monte Carlo estimation via ELBO

**Mathematical Formulation** (DiffuCoder Eq. 1):
```
L_t^(1:N) = (1/t) * E_q(x_t|x_0) [ -Σ_{n=1}^N δ_{x_t^n, m} * (x_0^n)^T * log f_θ(x_t^1:N)_n ]
```

Where:
- `δ_{x_t^n, m}`: Indicator function (1 if token n is masked, 0 otherwise)
- `f_θ(x_t)`: Model's logits for masked tokens
- `1/t`: Weighting factor from continuous-time formulation

**The Variance Problem**:
Monte Carlo sampling introduces significant variance because:
1. Each sample only evaluates a subset of masked tokens
2. High-entropy tokens (often at the beginning) dominate the gradient
3. Many samples needed to cover all tokens reliably

---

## Baseline Approaches and Limitations

### d1 Baseline (Zhao et al., 2025)

**Method**: Mask all completion tokens at once (t = T)
```python
# Pseudo-code for d1
mask = [1, 1, 1, ..., 1]  # All completion tokens masked
log_prob = compute_loss(tokens, mask, t=T)
```

**Advantages**:
- ✓ Simple: Single forward pass per completion
- ✓ Fast: Minimal computational overhead

**Disadvantages**:
- ✗ High bias: All tokens evaluated at maximum noise
- ✗ Unrealistic context: Tokens never see partial completions
- ✗ Entropy sink effect: Left tokens have artificially high confidence

**DiffuCoder Finding**: "As shown in our entropy sink analysis (§4.2), high-entropy tokens tend to lie on the left side, so RL training still ends up updating early tokens more aggressively."

### d1 with Condition Masking (p=0.15)

**Method**: Also mask 15% of condition tokens randomly
```python
# d1 with condition masking
condition_mask = random_mask(condition, p=0.15)
completion_mask = [1, 1, 1, ..., 1]
combined_mask = condition_mask + completion_mask
log_prob = compute_loss(tokens, combined_mask, t=T)
```

**DiffuCoder Finding** (Page 8): "Masking condition tokens does not yield a stable reward improvement (Figure 7), probably because code tasks demand higher token-level generation accuracy than math tasks."

---

## Coupled-GRPO Solution

### Core Concept: Complementary Masks

**Definition** (DiffuCoder Eq. 12):
```
M_t ∨ M_t̂ = 1    (Union covers all tokens)
M_t ∧ M_t̂ = 0    (Intersection is empty)
```

Where:
- `M_t`: Binary mask vector at timestep t
- `M_t̂`: Binary mask vector at timestep t̂ = T - t
- `∨`: Element-wise OR
- `∧`: Element-wise AND

**Visual Example**:
```
Sequence:     [I love dogs and cats]

M_t (t=0.6):  [1 1 0 0 0]  (mask first 60%)
M_t̂ (t̂=0.4): [0 0 1 1 1]  (mask remaining 40%)
Union:        [1 1 1 1 1]  ✓ All tokens covered
Intersection: [0 0 0 0 0]  ✓ No overlap
```

### Mathematical Formulation

**Probability Estimation** (DiffuCoder Eq. 13):
```
π_θ(o^k | c, o^k_{t<T}) = 1/(λ+1) * [ Σ_{t+t̂=T} (L_t(x_t) + L_t̂(x_t̂)) + L_T(x_T) ]
```

Breaking down each component:

1. **L_t(x_t)**: Loss at timestep t with mask M_t
   ```
   L_t(x_t) = M_t · (1/t) · CE(x_t, x_0)
   ```

2. **L_t̂(x_t̂)**: Loss at complementary timestep t̂ with mask M_t̂
   ```
   L_t̂(x_t̂) = M_t̂ · (1/t̂) · CE(x_t̂, x_0)
   ```

3. **L_T(x_T)**: Loss at full masking (all tokens masked)
   ```
   L_T(x_T) = (1/T) · CE(x_T, x_0)
   ```

4. **λ**: Number of complementary pairs (DiffuCoder uses λ=1)

**For λ=1** (typical setting):
```
π_θ(o^k | c, o^k_{t<T}) = 1/2 * [L_t(x_t) + L_t̂(x_t̂) + L_T(x_T)]
```

### GRPO Objective with Coupled Sampling

**Full Objective** (DiffuCoder Eq. 4):
```
J_GRPO(θ) = E[ Σ_{i=1}^G Σ_{k=1}^|o_i| min(ρ_i^k * A_i, clip(ρ_i^k, 1-ε, 1+ε) * A_i) - β * D_KL ]
```

Where:
- **ρ_i^k**: Importance ratio for token k in completion i
  ```
  ρ_i^k = π_θ(o_i^k | c, o_i^k_{t<T}) / π_θ_old(o_i^k | c, o_i^k_{t<T})
  ```

- **A_i**: Advantage for completion i
  - Standard: `A_i = r(o_i) - (1/G) * Σ_j r(o_j)`
  - LOO (Leave-One-Out): `A_i = r(o_i) - 1/(G-1) * Σ_{j≠i} r(o_j)`

- **clip(ρ, 1-ε, 1+ε)**: PPO-style clipping with ε=0.5 (DiffuCoder default)

- **β * D_KL**: KL penalty to reference model (β=0.01 in DiffuCoder)

---

## Complete Algorithm

### Algorithm 1: Coupled GRPO (from DiffuCoder Page 20)

```python
# Input Parameters
πref = reference_model
C = condition_set  # Training prompts
G = 10  # Completions per condition
T = test_cases  # Code verification tests
μ = 2   # GRPO iterations per batch
β = 0.01  # KL penalty weight
ε = 0.5   # PPO clipping parameter
λ = 1     # Number of complementary pairs

# Initialize
πθ = πref

# Training Loop
while not converged:
    # Update reference
    πref = πθ

    for step in range(I):  # I = number of training steps
        πold = πθ

        # Sample batch of conditions
        Cb ~ C

        # Generate completions with high temperature for diversity
        for c in Cb:
            {oi}_{i=1}^G ~ πold(·|c)  # Temperature = 1.2

        # Compute rewards via test execution
        for oi in completions:
            r(oi) = execute_tests(oi, Tc)

        # Compute advantages
        for i in range(G):
            Ai = r(oi) - (1/G) * Σ_j r(oj)
            # Or LOO: Ai = r(oi) - 1/(G-1) * Σ_{j≠i} r(oj)

        # GRPO iterations
        for j in range(μ):
            # Sample complementary timestep pair
            tj ~ Uniform(0.2, 0.8)  # "Sweet spot" range
            t̂j = T - tj

            # Create complementary masks for batch
            Mtj, Mt̂j = create_complementary_masks(tj, t̂j)

            # Compute losses
            Ltj = compute_loss(batch, Mtj, tj)
            Lt̂j = compute_loss(batch, Mt̂j, t̂j)
            LT = compute_loss(batch, all_masked, T)

            # Compute coupled probability estimates (Eq 13)
            log πθ(o^k|c, o^k_{t<T}) = 1/(λ+1) * [Ltj + Lt̂j + LT]

            # Compute importance ratios
            ρi^k = exp(log πθ - log πold)

            # Update policy via gradient descent on JGRPO (Eq 4)
            θ = θ + ∇θ JGRPO(θ)

return πθ
```

### Implementation Details

**1. Complementary Mask Generation**

```python
def create_complementary_masks(t, t_hat, sequence_length):
    """
    Create complementary masks where each token is masked
    in exactly one of the two masks.

    Args:
        t: timestep in [0, 1]
        t_hat: complementary timestep (T - t)
        sequence_length: length of completion sequence

    Returns:
        Mt, Mt_hat: Complementary binary masks
    """
    # Each token independently masked with probability t
    Mt = torch.bernoulli(torch.ones(sequence_length) * t)

    # Complementary mask: mask what wasn't masked
    Mt_hat = 1 - Mt

    # Verify complementarity
    assert (Mt | Mt_hat).all() == 1  # Union is all 1s
    assert (Mt & Mt_hat).all() == 0  # Intersection is all 0s

    return Mt, Mt_hat
```

**2. Loss Computation with Masks**

```python
def compute_loss(tokens, mask, timestep):
    """
    Compute weighted cross-entropy loss for masked tokens.

    Args:
        tokens: completion tokens [batch, seq_len]
        mask: binary mask [batch, seq_len]
        timestep: current timestep t in [0, 1]

    Returns:
        loss: weighted cross-entropy per token
    """
    # Forward pass to get logits
    logits = model(tokens, mask=mask)

    # Cross-entropy only on masked positions
    ce_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        tokens.view(-1),
        reduction='none'
    ).view(batch, seq_len)

    # Apply mask and weighting
    weighted_loss = mask * (1/timestep) * ce_loss

    return weighted_loss
```

**3. Reward Function** (DiffuCoder Page 24)

```python
def compute_reward(completion, test_cases):
    """
    Weighted reward combining format and correctness.

    Args:
        completion: generated code string
        test_cases: list of test cases for verification

    Returns:
        reward: float in [0, 2.5]
    """
    # Format reward
    rformat = 0.0
    if has_valid_markdown_block(completion):
        if passes_syntax_check(completion):
            rformat = 0.5
        else:
            rformat = 0.25

    # Code correctness reward
    rcode = 0.0
    if rformat == 0.5:  # Only test if format is valid
        rcode = pass_rate(completion, test_cases)

    # Combined reward
    reward = 2.0 * rcode + 0.5 * rformat

    return reward
```

---

## Theoretical Guarantees

### Theorem 1: Unbiasedness (DiffuCoder Appendix A.4)

**Statement**: The coupled estimator is unbiased:
```
E[v̂k,AV] = vk
```

Where:
- `v̂k,AV`: Antithetic variates estimator for token k
- `vk = E_{t,M}[g(t, M, k)]`: True expected score

**Proof Sketch**:
1. The joint distribution p(t, M) = p(t̂, M̂) due to symmetry
2. E[g(t, M, k)] = E[g(t̂, M̂, k)] = vk
3. By linearity: E[(g(t,M,k) + g(t̂,M̂,k))/2] = vk

**Implication**: Coupled-GRPO provides correct gradient estimates on average.

### Theorem 2: Variance Reduction (DiffuCoder Appendix A.4)

**Statement**: The coupled estimator has lower variance than independent sampling:
```
Var(v̂k,AV) < Var(v̂k,MC)
```

**Proof Sketch**:
1. For standard Monte Carlo with 2N samples:
   ```
   Var(v̂k,MC) = σ²g / (2N)
   ```

2. For antithetic variates with N pairs:
   ```
   Var(v̂k,AV) = [Var(g) + Cov(g(t,M,k), g(t̂,M̂,k))] / (2N)
   ```

3. **Key Insight**: The covariance is negative!
   ```
   Cov(g(t,M,k), g(t̂,M̂,k)) = -vk²
   ```

   Reason: For any token k, exactly one of Mk or M̂k is 1, so:
   ```
   g(t,M,k) · g(t̂,M̂,k) = 0  (always!)
   ```
   Therefore:
   ```
   Cov = E[g·g] - E[g]·E[g] = 0 - vk² = -vk²
   ```

4. Variance reduction:
   ```
   Var(v̂k,MC) - Var(v̂k,AV) = vk² / (2N) > 0
   ```

**Implication**: Coupled sampling reduces variance by exactly `vk²/(2N)` for each token.

---

## Comparison with Baselines

### Token Coverage Analysis

| Method | Tokens Evaluated per Forward Pass | Total Coverage | Variance |
|--------|-----------------------------------|----------------|----------|
| **d1 (Full Mask)** | All tokens | High | **High** (all at t=T) |
| **d1 (p=0.15)** | All tokens | High | **Very High** (condition noise) |
| **Independent MC (2λ samples)** | Random subset | Medium | High |
| **Coupled-GRPO (λ pairs)** | All tokens (via pairs) | **Full** | **Low** |

**From DiffuCoder Figure 2**:
```
Traditional Sampling:
- Forward pass 1: Evaluate tokens [1, 3, 5, 7, ...]  (random subset)
- Forward pass 2: Evaluate tokens [2, 4, 6, 8, ...]  (random subset)
- Coverage: Probabilistic (some tokens may be missed)

Coupled Sampling:
- Forward pass 1: Evaluate tokens [1, 2, 3, ...]  (first 60% with Mt)
- Forward pass 2: Evaluate tokens [..., 8, 9, 10] (last 40% with Mt̂)
- Coverage: Guaranteed (each token exactly once per pair)
```

### Empirical Results (DiffuCoder Table 2)

**Performance Gains** (HumanEval+ / MBPP+):

| Model | HumanEval+ | MBPP+ | EvalPlus Avg | Change |
|-------|-----------|-------|--------------|--------|
| DiffuCoder-Instruct | 65.2% | 61.9% | 63.6% | baseline |
| + Full Mask (d1, p=0) | 59.1% | 65.1% | 62.1% | **-1.5%** |
| + Decoupled Sampling | 62.8% | 66.4% | 64.6% | +1.0% |
| + **Coupled-GRPO** | **68.3%** | **67.5%** | **67.9%** | **+4.3%** |
| + Coupled-GRPO (LOO) | 62.2% | 68.5% | 65.4% | +1.8% |

**Key Findings**:
1. ✓ Coupled-GRPO: **+4.3% absolute improvement**
2. ✗ Full mask (d1) baseline: **worse than instruct model alone**
3. ~ Decoupled sampling: small gains but unstable training
4. ~ LOO advantage: better on MBPP+ but worse on HumanEval+

---

## Hyperparameter Sensitivity

### Critical Hyperparameters (from DiffuCoder experiments)

**1. Rollout Temperature**

| Temperature | pass@1 | pass@10 | AR-ness | Training Stability |
|-------------|--------|---------|---------|-------------------|
| 0.2 | 72.0% | 75.2% | High | Stable |
| 1.0 | 56.1% | 82.7% | Medium | Stable |
| **1.2** | 50.4% | **87.8%** | **Low** | **Stable** ✓ |
| 1.5 | 45.3% | 85.1% | Very Low | Unstable |

**DiffuCoder Finding** (Page 9): "DiffuCoder-Instruct attains a higher pass@10 at temperature 1.2 than at 1.0, mirroring the trend observed during coupled-GRPO training."

**Optimal**: Temperature = 1.2 for rollouts

**2. Timestep Sampling Range**

From DiffuCoder Figure 8 (Page 24):
- **[0.0, 0.2]**: Extremely high loss (near-deterministic state)
- **[0.2, 0.8]**: "Sweet spot" ✓ - stable, bounded loss
- **[0.8, 1.0]**: Extremely high loss (full noise)

**DiffuCoder Implementation**: Sample t ~ Uniform(0.2, 0.8)

**3. Number of Pairs (λ)**

| λ | Forward Passes | Training Time | Performance |
|---|----------------|---------------|-------------|
| 1 | 3 (Mt, Mt̂, MT) | 1x | **Optimal** ✓ |
| 2 | 5 | ~1.7x | Similar |
| 4 | 9 | ~3x | Diminishing returns |

**DiffuCoder Choice**: λ = 1 (best trade-off)

**4. GRPO-Specific Parameters**

From DiffuCoder Appendix B.1 (Page 23):
- **G = 10**: Number of rollouts per condition
- **μ = 2**: GRPO iterations per batch
- **β = 0.01**: KL penalty weight
- **ε = 0.5**: Clipping parameter
- **Learning rate = 1e-6**: Conservative for stability
- **Reference sync = 64 steps**: Update πref every 64 gradient steps

---

## Training Dynamics

### Reward Curve Analysis (DiffuCoder Figure 7)

**Coupled-GRPO vs Baselines**:

```
Reward
2.0 │                        ╱─────────
    │                   ╱────          Coupled-GRPO ✓
1.5 │              ╱────
    │         ╱────
1.0 │    ╱────
    │╱───────────────────────          Full Mask (unstable)
0.5 │    ╱╲                            d1 p=0.15 (collapses)
    │   ╱  ╲╱╲
0.0 └───────────────────────────────
    0   0.2  0.4  0.6  0.8  1.0
              Training Epoch
```

**Observations**:
1. **Coupled-GRPO**: Smooth, monotonic improvement
2. **Full Mask (p=0)**: High variance, unstable
3. **d1 (p=0.15)**: Catastrophic collapse after 0.4 epochs
4. **Decoupled**: Better than baselines but still noisy

### AR-ness Evolution (DiffuCoder Figure 4)

**Global AR-ness@1 (left-to-right bias)**:

```
Before GRPO:  0.82 (high causal bias)
After GRPO:   0.65 (reduced bias) ✓
```

**Interpretation**: Coupled-GRPO reduces autoregressive bias, enabling more parallel token generation.

### Decoding Speed Analysis (DiffuCoder Figure 1c)

**Performance at 2× Speed** (half timesteps):

| Model | HumanEval+ (full) | HumanEval+ (0.5×) | Drop |
|-------|------------------|-------------------|------|
| Instruct | 65.2% | 46.3% | **-29.0%** |
| + Coupled-GRPO | 68.3% | 55.1% | **-19.3%** ✓ |

**Key Finding**: Coupled-GRPO models degrade less at higher speeds, indicating better parallelization capability.

---

## Implementation Best Practices

### 1. Batch Processing

**Challenge**: Creating complementary masks for variable-length sequences in a batch.

**Solution**:
```python
def create_batch_complementary_masks(batch_lengths, t, t_hat):
    """
    Create complementary masks for variable-length batch.
    """
    max_len = max(batch_lengths)
    batch_size = len(batch_lengths)

    # Initialize masks
    Mt = torch.zeros(batch_size, max_len)
    Mt_hat = torch.zeros(batch_size, max_len)

    for i, length in enumerate(batch_lengths):
        # Sample complementary masks for this sequence
        mask_t = torch.bernoulli(torch.ones(length) * t)

        # Assign to batch tensors
        Mt[i, :length] = mask_t
        Mt_hat[i, :length] = 1 - mask_t

    return Mt, Mt_hat
```

### 2. Gradient Accumulation

**DiffuCoder Setting** (Page 23):
- Single GPU batch size: 2
- 8 GPUs per node
- Effective batch size: 16 via accumulation

**Why**: GRPO requires stable advantage estimates, which benefit from larger batch sizes.

### 3. Reference Model Updates

**DiffuCoder Strategy**:
```python
# Update reference model every 64 gradient steps
if step % 64 == 0:
    πref = copy.deepcopy(πθ)
```

**Rationale**: Balance between:
- Too frequent: Minimal policy improvement per update
- Too rare: πθ diverges too far from πref, high variance in ρ

### 4. Reward Clipping and Normalization

**Not explicitly described in DiffuCoder**, but standard practice:
```python
# Clip rewards to prevent outliers
rewards = torch.clamp(rewards, min=0, max=2.5)

# Normalize advantages
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

---

## Comparison with LLaDA 1.5 VRPO

### Conceptual Alignment

Both papers propose variance reduction for diffusion LLM training:

| Aspect | LLaDA 1.5 VRPO | DiffuCoder Coupled-GRPO |
|--------|----------------|-------------------------|
| **Base RL Algorithm** | DPO | GRPO |
| **Variance Target** | Preference score estimator | Token probability estimator |
| **Sampling Strategy** | Share timesteps between π and πref | Complementary mask pairs |
| **Theoretical Basis** | Antithetic variates | Antithetic variates |
| **Budget Allocation** | nt = n, nyt = 1 | λ = 1 |

### Technical Differences

**LLaDA 1.5**:
```
Samples n timesteps {t^(j)}_{j=1}^n
Shares same timesteps between policy and reference model
Reduces variance in score difference: sθ(yw, yl)
```

**DiffuCoder**:
```
Samples λ pairs (t, t̂) where t + t̂ = T
Creates complementary masks within same policy
Reduces variance in per-token probability: πθ(o^k|c)
```

**Common Goal**: Both ensure each token is evaluated in correlated contexts to reduce variance through negative covariance.

---

## Limitations and Future Work

### Current Limitations

1. **Computational Cost** (DiffuCoder Appendix C.4):
   - End-to-end GRPO training time: **2× longer than AR GRPO**
   - Forward pass overhead: 3 passes vs 1 for AR models

2. **Timestep Selection**:
   - Restricted to "sweet spot" [0.2, 0.8]
   - Misses extreme noise/deterministic regimes
   - May limit exploration

3. **Temperature Sensitivity**:
   - Requires high temperature (1.2) for diverse rollouts
   - May generate lower-quality initial samples
   - Trade-off between diversity and quality

4. **Sequence Length Limitation**:
   - Current max: 256 tokens for completion
   - Insufficient for long reasoning chains
   - Limited by diffusion timesteps = sequence length

### Promising Directions

**1. Adaptive Timestep Sampling**:
```python
# Instead of uniform sampling in [0.2, 0.8]
# Use learned distribution
t ~ Beta(α, β)  # Learn α, β during training
```

**2. Multi-Scale Coupling**:
```python
# Multiple complementary pairs at different scales
pairs = [
    (0.3, 0.7),  # Coarse-grained
    (0.4, 0.6),  # Fine-grained
    (0.45, 0.55) # Very fine
]
```

**3. Token-Specific Masks**:
```python
# Learn importance weights for tokens
# Mask high-importance tokens less frequently
p_mask[i] = base_prob * (1 - importance[i])
```

**4. Integration with KV-Caching**:
- Exploit recent work on diffusion LLM acceleration (Wu et al., 2025)
- Cache intermediate representations across timesteps
- Potential for 5-10× speedup

---

## Experimental Checklist for Reproduction

### Phase 1: Baseline Training
- [ ] Train DiffuCoder base model (Stage 1 + 2)
- [ ] Instruction tune with classifier-free guidance
- [ ] Verify base performance matches paper

### Phase 2: GRPO Setup
- [ ] Implement coupled mask generation
- [ ] Implement d1 baseline (p=0 and p=0.15)
- [ ] Implement decoupled sampling baseline
- [ ] Set up code execution sandbox (E2B)

### Phase 3: Hyperparameter Tuning
- [ ] Test rollout temperatures: {0.2, 1.0, 1.2, 1.5}
- [ ] Test timestep ranges: {[0,1], [0.2,0.8], [0.3,0.7]}
- [ ] Test λ values: {1, 2, 4}
- [ ] Test advantage types: {standard, LOO}

### Phase 4: Training
- [ ] Run coupled-GRPO for 1 epoch on 21K samples
- [ ] Monitor reward curves (should be smooth)
- [ ] Track AR-ness evolution (should decrease)
- [ ] Evaluate at temperatures: {0.2, 0.3, 0.4}

### Phase 5: Analysis
- [ ] Compare with baselines on HumanEval/MBPP
- [ ] Test at 2× speed (half timesteps)
- [ ] Measure inference latency
- [ ] Visualize decoding trajectories

---

## Key Takeaways

1. **Problem**: Monte Carlo estimation in diffusion models has high variance

2. **Solution**: Complementary masks ensure each token evaluated exactly once per pair

3. **Theory**: Negative covariance via antithetic variates guarantees variance reduction

4. **Practice**: +4.3% absolute improvement with stable training

5. **Broader Impact**: First successful RL method for dLLMs that respects non-AR nature

**Quote from DiffuCoder** (Page 2):
> "Our work provides deeper insight into the machinery of dLLM generation and offers an effective, diffusion-native RL training framework."
