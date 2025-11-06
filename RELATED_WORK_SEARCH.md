# Related Work Search - Neural Codec Diffusion for Code

## Papers Found (User's Search)

### 1. CodecFake+ (2025) - Detection
- **Domain:** Audio deepfake detection
- **Relevance:** Low (detection, not generation)
- **Impact:** None - different problem

### 2. TaDiCodec (2025) - IMPORTANT
- **Domain:** Audio codec
- **Key Point:** "avoids multi-layer residual vector quantization structures, opting for a single-layer codebook"
- **Architecture:** Encoder → Single VQ → Diffusion Decoder
- **Relevance:** HIGH - but they explicitly AVOID what we propose
- **Impact:** POSITIVE - validates that multi-layer RVQ + diffusion is unexplored
- **Positioning:** "While TaDiCodec avoids multi-layer RVQ for audio, we show hierarchical RVQ is beneficial for code generation's structured nature"

### 3. StreamCodec (2025)
- **Domain:** Audio codec with RSVQ
- **Relevance:** Low (no diffusion)
- **Impact:** None - orthogonal codec optimization

### 4. MBCodec (2025)
- **Domain:** Audio codec with RVQ
- **Relevance:** Low (no diffusion)
- **Impact:** None - just codec design

### 5. Diffusion + Quantization for Image Compression
- **Domain:** Image compression
- **Relevance:** Medium (different modality)
- **Impact:** Low - images ≠ code

## Critical Gaps (Our Novelty)

✅ **Gap 1:** No work applies neural codec + diffusion to CODE generation
✅ **Gap 2:** No work uses multi-layer RVQ + separate diffusion model for generation
✅ **Gap 3:** TaDiCodec explicitly avoids multi-layer RVQ (we embrace it)
✅ **Gap 4:** Existing work is audio/image, not text/code

## Additional Keywords to Search

### High Priority (Search These!)
1. **"neural codec language model"**
2. **"diffusion language model code generation"**
3. **"residual quantization transformer language"**
4. **"discrete diffusion code synthesis"**
5. **"hierarchical latent language model"**

### Medium Priority
6. **"vector quantization language model"**
7. **"codec-based text generation"**
8. **"diffusion autoregressive code"**
9. **"quantized latent language generation"**
10. **"multi-level codebook language model"**

### Low Priority (But Check)
11. **"neural codec natural language"**
12. **"residual VQ BERT"** (BERT uses discrete codes internally)
13. **"hierarchical quantization GPT"**
14. **"diffusion based program synthesis"**

## Papers to Specifically Look Up

### Codec + Language Models
- [ ] "MQTTS" - Multi-level quantization for TTS (might have language modeling)
- [ ] "SoundStorm" - Uses RVQ + parallel decoding (not diffusion, but related)
- [ ] Any work citing Encodec + GPT/Transformer together

### Diffusion Language Models (No Codec)
- [ ] DiffuSeq (2022) - Diffusion for text, but on tokens not latents
- [ ] Diffusion-LM (2022) - Continuous diffusion for text
- [ ] GENIE (2024) - If it uses diffusion + discrete codes

### RVQ Language Models (No Diffusion)
- [ ] VALL-E (2023) - Uses RVQ codes for speech, but AR not diffusion
- [ ] AudioLM (2022) - RVQ + language model, but AR not diffusion
- [ ] SpeechGPT - If it uses neural codec

## Search Strategy

### Step 1: Google Scholar
```
"neural codec" AND "language model"
"residual vector quantization" AND "transformer"
"diffusion" AND "code generation" AND "latent"
```

### Step 2: arXiv
```
cat:cs.CL AND ("neural codec" OR "residual quantization") AND ("diffusion" OR "latent")
cat:cs.LG AND "code generation" AND "diffusion"
```

### Step 3: Semantic Scholar
- Papers citing: Encodec (2022)
- Papers citing: DiffuCoder (2024)
- Papers citing: LLaDA (2024)
- Intersection of codec + diffusion papers

### Step 4: GitHub Code Search
```
"residual vector quantization" language:Python
"encodec" AND "transformer" AND "language"
"neural codec" AND "code generation"
```

## Key Papers We Already Know (Baselines)

### Audio Codecs
1. ✅ **Encodec** (Meta, 2022) - RVQ codec we'll adapt
2. ✅ **SoundStream** (Google, 2021) - Original RVQ codec
3. ✅ **DAC** (Descript, 2023) - Improved RVQ codec

### Code Diffusion
4. ✅ **DiffuCoder** (Apple, 2024) - Coupled-GRPO, token-level (in our repo)
5. ✅ **LLaDA** (2024) - Masked diffusion, token-level (in our repo)

### Audio Generation with Codecs
6. ✅ **AudioLM** (Google, 2022) - RVQ + AR (not diffusion)
7. ✅ **VALL-E** (Microsoft, 2023) - RVQ + AR for speech
8. ✅ **MusicGen** (Meta, 2023) - Uses Encodec + AR

### Text Diffusion
9. ✅ **Diffusion-LM** (2022) - Continuous diffusion on embeddings
10. ✅ **DiffuSeq** (2022) - Diffusion on token sequences

## What Would Kill Our Novelty?

### 🚨 Red Flags (If Found, Need New Angle)
- Paper that uses multi-level RVQ + diffusion for CODE generation
- Paper that applies Encodec-style codec to code/text + diffusion model
- Paper explicitly titled "Neural Codec Language Models" or similar

### ⚠️ Yellow Flags (Need to Differentiate)
- RVQ for text (but no diffusion) → We add diffusion
- Diffusion for code (but on tokens) → We use latent space
- Codec + AR for text → We use diffusion instead of AR

### ✅ Green Flags (Strengthens Our Work)
- More papers avoiding multi-layer RVQ (like TaDiCodec)
- Papers saying "discrete tokens are hard for diffusion" → We solve it with codecs
- Papers showing RVQ works for audio but not trying text/code

## Current Assessment

**Novelty Status:** ✅ **STRONG**

**Reasoning:**
1. No existing work on RVQ + diffusion for CODE
2. TaDiCodec AVOIDS what we propose (validates gap)
3. Audio/image work doesn't transfer directly to code
4. Unique combination of 3 elements

**Recommendation:**
1. Do the additional keyword searches above
2. Specifically check for "neural codec language" and related
3. If nothing found → PROCEED with confidence
4. Update related work section in paper with proper positioning

## Related Work Section (Draft)

```markdown
### Neural Audio Codecs
Recent work on neural audio codecs [Encodec, SoundStream, DAC] uses residual
vector quantization (RVQ) to achieve high-quality compression. However, these
are designed for audio waveforms, not discrete token sequences.

### Diffusion for Code Generation
DiffuCoder and LLaDA apply diffusion models directly to token sequences. While
effective, they operate in the discrete token space, limiting generation speed
and requiring many denoising steps (100-128).

### Codecs Meet Language Models
VALL-E and AudioLM use RVQ codes with autoregressive language models for speech.
TaDiCodec (2025) integrates diffusion into the codec decoder but explicitly
avoids multi-layer RVQ, using single-layer quantization instead.

### Our Contribution
We propose the first application of multi-level RVQ + separate diffusion modeling
for code generation. Unlike TaDiCodec which avoids hierarchical quantization for
audio, we show that multi-level RVQ naturally captures code's hierarchical
structure (syntax, semantics, style). Unlike AudioLM/VALL-E which use
autoregressive models, we use diffusion for non-autoregressive generation.
```

## Next Actions

- [ ] Search new keywords (30 minutes)
- [ ] Check specific papers (VALL-E, AudioLM, SoundStorm)
- [ ] Look for any 2024-2025 papers on "latent language models"
- [ ] If clear → Update IDEA_3 document with related work
- [ ] If red flags found → Discuss differentiation strategy
