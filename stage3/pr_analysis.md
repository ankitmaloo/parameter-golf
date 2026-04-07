# PR Analysis Summary — Parameter Golf Competition

**Date**: 2026-04-06
**Our current score**: 1.1631 BPB (rank ~20)
**Merged SOTA**: 1.1147 (PR #1019)
**Open no-TTT frontier**: 1.0856 (PR #1394)
**Open TTT frontier**: 1.0795 (PR #1416)
**Open SLOT frontier**: 0.7094 (PR #1376)

---

## Merged Records (chronological)

| PR | Score | Date | Author | Stack |
|----|-------|------|--------|-------|
| #1019 | 1.1147 | 03-30 | abaybektursun | AR Self-Gen GPTQ, XSA-all, BigramHash 3072×112, Parallel Muon, LeakyReLU², EMA |
| #549 | 1.1194 | 03-24 | (LeakyReLU² + Legal Score-First TTT + Parallel Muon) |
| #414 | 1.1233 | 03-23 | signalrush | 11L, GPTQ-lite, EMA, XSA4, Partial RoPE, LN Scale, VE128, Late QAT@0.15 |
| #315 | 1.1248 | 03-23 | jfprincz | 11L, Partial RoPE 16/64, LN Scale, EMA, XSA4 |
| #287 | 1.1271 | 03-23 | jfprincz | 11L, XSA4, EMA, Int6 MLP3x, WD=0.04 |
| #265 | 1.1307 | 03-23 | unnir | 11L, Efficient Partial XSA (3 layers), FA3, SWA120 |
| #363 | N/A | 03-25 | evangelinehelsinki | Non-record: Depth Recurrence research — what works, what doesn't |
| #180 | 1.1428 | 03-20 | thwu1 | 10L, Mixed int5/int6, BigramHash(10240), SWA(0.4), WD=0.04 |
| #162 | 1.1458 | 03-20 | raahilshah | Int6 MLP3x, SmearGate, BigramHash, OrthoInit, MuonWD, SWA |

---

## Open PRs — Tier A (sub-1.09, real contenders)

### PR #1420 — Triple Loop + Fused Kernels + Parallel Residuals (1.0801, 5-seed)
**Author**: abaybektursun | **TTT**: Yes (Pre-Quant) | **SLOT**: No
**What's new**:
- Triple loop (NUM_LOOPS=3) on layers 4-5 → 17 virtual layers from 11 physical
- Activate looping at 0.35 instead of 0.50 (earlier helps)
- Fused MLP kernels: Triton TMA forward + CUTLASS EVT backward → +10% throughput, +127 steps
- Parallel residuals (GPT-J style) on layers 7-10 → +68 steps from faster forward
- N-gram Tilt (details in appendix)
**Takeaway**: Engineering depth. Fused kernels are high-effort but high-reward. Triple loop > double loop > no loop.

### PR #1416 — SP8192 + Pre-Quant TTT (1.0795, 3-seed)
**Author**: erichroepke | **TTT**: Yes (Pre-Quant) | **SLOT**: No
**What's new**:
- Simple combo: PR #1394 base + PR #1364 pre-quant TTT
- TTT: 6 epochs AdamW on EMA model before GPTQ, freeze first 2 blocks
- TTT gives -0.034 BPB (post-EMA 1.1019 → post-TTT 1.0682)
- Built by a filmmaker with Claude Opus 4.6
**Takeaway**: Pre-quant TTT is orthogonal to base architecture. Just bolt on top.

### PR #1423 — SP8192 + Pre-Quant TTT + QK-Gain 5.0 (1.0791, 3-seed)
**Author**: aryanbhosale | **TTT**: Yes (Pre-Quant) | **SLOT**: No
**What's new**:
- Same as #1416 but QK-Gain 5.0 (up from 4.0)
- One hyperparameter change: -0.0004 BPB
**Takeaway**: QK-Gain 5.0 is free and helps.

### PR #1394 — SP8192 + GPTQ Embeddings + Depth Recurrence + MuonEq-R + SDClip (1.0856, 5-seed)
**Author**: clarkkev | **TTT**: No | **SLOT**: No
**What's new**:
- SP8192 vocab (up from SP4096)
- GPTQ on embedding matrix (int8) — saves space
- Depth recurrence: loop layers 4-5 twice
- SDClip: std-dev based clip threshold instead of percentile search
- Removed value embeddings (not needed with SP8192)
- ShuffledSequenceLoader replaces coprime-stride loader
- Row-normalized Muon (MuonEq-R)
- Brotli compression
**Takeaway**: The strongest no-TTT base. This is the template to adopt.

### PR #1408 — dTTT + BigramHash 3072×112 (1.0800, 3-seed)
**Author**: aamodbhatt | **TTT**: Yes (dTTT) | **SLOT**: No
**What's new**:
- Discriminative TTT: per-block adaptive LR (0.3×-1.0×)
- BigramHash 3072×112 (up from 2048×128)
- 10 epochs dTTT, AdamW LR=0.0005, GPTQ damp=0.005
- QK-Gain 5.0, WARMDOWN=4000, XSA all-layers
**Takeaway**: dTTT (discriminative TTT) with per-block LR groups is the refined TTT approach.

### PR #1415 — SP4096 + 3-Layer Recurrence + GPTQ Embeddings + SDClip + ETLB (1.0913, 3-seed)
**Author**: bigbag | **TTT**: No | **SLOT**: No
**What's new**:
- SP4096 (not SP8192)
- 3-layer depth recurrence (layers 3,4,5)
- ETLB: Eval-Time Logit Bias — optimizes vocab bias during eval
- GPTQ on embeddings
- SDClip
- QK-Gain 5.0
- WD=0.095, MLR=0.022
- LZMA code wrapper (18KB code saves ~40KB artifact)
**Takeaway**: ETLB is a cheap eval-time trick worth ~-0.001 BPB. Higher WD=0.095 works with MuonEq-R.

### PR #1399 — Pre-Quant TTT + ETLB (1.0898, 3-seed)
**Author**: AnubhavBharadwaaj | **TTT**: Yes (Pre-Quant) | **SLOT**: No
**What's new**:
- ETLB: bias vector b ∈ R^vocab optimized during sliding window eval
- Pre-quant TTT: freeze first 9 of 11 blocks, adapt last 2
- ETLB gives ~-0.002 BPB on top of sliding window
**Takeaway**: ETLB is a free add-on for eval. Stacks with everything.

### PR #1392 — SP4096 + Depth Recurrence + Parallel Residuals + Brotli (1.1020, 3-seed)
**Author**: Its-Just-Crump | **TTT**: No | **SLOT**: No
**What's new**:
- SP4096 native architecture
- MLP 4x (wider with SP4096)
- Depth recurrence + parallel residuals
- QK-Gain (higher than default)
- Brotli compression
- Replaces BigramHash with "SP4096 gets bigram info natively from larger vocab"
**Takeaway**: SP4096 + no BigramHash shows vocab size can substitute for explicit bigram features.

---

## Open PRs — Tier B (1.09-1.12, interesting techniques)

### PR #1410 — LatentMask TTT + Product-Key Bigram + Brotli (1.1158, 3-seed)
**Author**: izlley | **TTT**: Yes (LatentMask) | **SLOT**: No
**What's new**:
- LatentMask TTT: per-channel sigmoid masks + biases on MLP/attention outputs, trained per-chunk with sign-based Muon-lite optimizer
- Product-Key Bigram: factored `embed_prev(1024,512) * embed_cur(1024,512)`, zero hash collisions
- Alternating GatedAttention (every other layer)
- Brotli-11 + uint8 log-scale serialization
**Takeaway**: Product-Key Bigram is cleaner than hash-based BigramHash. LatentMask TTT is lightweight (~-0.002 BPB).

### PR #1384 — Progressive Depth + Hedge Mixer (1.1441, 3-seed)
**Author**: iverbovoy | **TTT**: No | **SLOT**: No
- Progressive depth training with hedge-based layer mixing

### PR #1421 — Depth Recurrence + EMA Tuning 0.9965 (1.0925)
**Author**: X-Abhishek-X | **TTT**: Implied | **SLOT**: No
- Higher EMA decay (0.9965 vs 0.997) — longer averaging window

### PR #1422 — Depth Recurrence + GPTQ + SGD TTT on 1xH100 (1.1172)
**Author**: swapp1990 | **TTT**: Yes (SGD) | **SLOT**: No
- Shows depth recurrence + GPTQ works on single H100 too

---

## Open PRs — Tier S (SLOT / exotic, sub-0.80)

### PR #1376 — SLOT-24 + Pre-quant TTT (0.7094, 3-seed)
**Author**: stukenov | **TTT**: Yes | **SLOT**: Yes
**What's new**:
- SLOT (Sparse Latent Optimization at Test-time, arXiv:2505.12392v2)
- Per-sample delta [bsz,1,512] + logit_bias [bsz,1,1024]
- 24 AdamW steps per sample (cosine LR 0.024→0.001, stride=96)
- Model weights frozen during SLOT — only delta+bias optimized
- Combined with pre-quant TTT (6 epochs on EMA before GPTQ)
**Takeaway**: SLOT is a game-changer (-0.37 BPB!). But fundamentally different paradigm. Competition may split.

---

## Merged Research — Key Papers

### PR #363 — Depth Recurrence: What Works, What Doesn't (merged non-record)
**Author**: evangelinehelsinki | **35 runs across 8xH100**
**Key findings**:
1. "Flat 11L 512d: 1.1648 vs Looped 3x3 640d: 1.1894" — looped was WORSE initially
2. **Noisy QAT** collapses recurrence quantization gap from 0.37 to 0.002 BPB
3. 3x3 > 2x5 loops (more unique blocks with fewer repeats)
4. Quantization error compounds through N repeats superlinearly
5. **12 negative results** documented: XSA all layers (+0.001 worse), cyclic momentum (catastrophic), QuadgramHash (unclear), factored embeddings (worse), Value Residual (+0.14 catastrophic), progressive unrolling (DDP crash)

**Why later PRs succeeded**: Better quantization (GPTQ+SDClip instead of simple int6) neutralizes the quantization compounding problem.

---

## Technique Evolution Timeline

```
Era 1 (Mar 18-19): Baseline + env vars
  fp16 embed, sliding window, 10L, MuonWD → 1.17-1.16

Era 2 (Mar 19-20): Module additions
  SmearGate, BigramHash, OrthoInit, SWA, int5/int6 → 1.15-1.14

Era 3 (Mar 20-22): Architecture + quantization
  XSA, EMA, Partial RoPE, LN Scale, VE128, FA3, GPTQ-lite → 1.13-1.12

Era 4 (Mar 22-24): Full GPTQ + LeakyReLU²
  Full Hessian GPTQ, LeakyReLU², Parallel Muon, VRL, TTT → 1.12-1.11

Era 5 (Mar 25-30): Depth recurrence + bigger vocab
  Depth recurrence, MuonEq-R, AR Self-Gen GPTQ, XSA-all → 1.11

Era 6 (Mar 30 - Apr 6): SP8192 + compression-aware quant
  SP4096/SP8192, SDClip, GPTQ embeddings, Brotli → 1.09-1.08
  Pre-Quant TTT, dTTT, ETLB, Parallel Residuals → 1.08
  SLOT → 0.71
```

We are in Era 2-3. The frontier is in Era 6.

---

## What We Need (ordered by impact)

### Must-Have (each is >0.01 BPB)
1. **SP4096 or SP8192 tokenizer** — SP1024 is ~0.02-0.03 BPB behind
2. **Full Hessian GPTQ** — ~0.02 BPB from quantization improvement
3. **Depth Recurrence** — ~0.015 BPB from virtual depth
4. **11 Layers** — ~0.005-0.01 BPB

### High Priority (each is 0.003-0.005 BPB)
5. **MuonEq-R** (row-normalized Muon)
6. **XSA on all 11 layers**
7. **LeakyReLU(0.5)²**
8. **EMA(0.997)**
9. **SDClip** (replaces percentile clip)
10. **GPTQ on embeddings**
11. **Brotli compression** (replaces zstd/lzma)

### Medium Priority (each is 0.001-0.003 BPB)
12. **QK-Gain 5.0**
13. **BigramHash 3072×112** (or Product-Key Bigram)
14. **Parallel Residuals** (GPT-J style, deep layers)
15. **MuonWD=0.04** (or higher: 0.095 in PR #1415)
16. **Warmdown 3500-4000**
17. **ETLB** (eval-time logit bias)
18. **Partial RoPE 16/64**
19. **LN Scale**

### TTT Lane (if pursuing)
20. **Pre-Quant TTT** (6 epochs, freeze first 2 blocks) — ~-0.034 BPB
21. **dTTT** (discriminative, per-block adaptive LR)
22. **SLOT** (24 steps per sample) — ~-0.37 BPB

### Estimated Stack Composite
```
Our current:       1.1631
+ SP4096/8192:     ~1.14   (tokenizer alone gives ~0.02)
+ 11L:             ~1.13
+ MuonWD + EMA:    ~1.12
+ GPTQ + SDClip:   ~1.10   (quantization is massive)
+ Depth Recurrence: ~1.085
+ XSA-all + LeakyReLU² + MuonEq-R: ~1.080
+ Pre-Quant TTT:   ~1.050
```

---

## Lineage of Key PRs

```
PR #162 (SmearGate/BigramHash/MuonWD/SWA, 1.1458)
  └→ PR #265 (XSA, FA3, 1.1307)
     └→ PR #287 (EMA, 1.1271)
        └→ PR #315 (Partial RoPE, LN Scale, Late QAT, 1.1248)
           └→ PR #414 (GPTQ-lite, VE128, warmdown3500, 1.1233)
              └→ PR #549 (Parallel Muon, Score-First TTT, 1.1194)
                 └→ PR #1019 (AR Self-Gen GPTQ, XSA-all, BH3072, 1.1147) ← CURRENT MERGED SOTA

PR #593 (Full GPTQ, Parameter Banking, LeakyReLU², 1.1171) [independent line]

PR #1204 (Depth Recurrence, omrigotlieb) [new technique]
  └→ PR #1217 (MuonEq-R, QK-Gain, bigbag)
     └→ PR #1394 (SP8192, SDClip, GPTQ embed, clarkkev, 1.0856)
        └→ PR #1416 (+ Pre-Quant TTT, erichroepke, 1.0795)
           └→ PR #1420 (Triple Loop + Fused Kernels, abaybektursun, 1.0801)
              └→ PR #1423 (+ QK-Gain 5.0, aryanbhosale, 1.0791)

PR #1364 (Pre-Quant TTT, stukenov) [new technique]
  └→ PR #1376 (+ SLOT-24, stukenov, 0.7094)
```

---

## PR #1019 Code-Level Diff vs Our Baseline

Full diff of `train_gpt.py`: 2135 lines (PR #1019) vs 1372 lines (baseline). Every subsystem was rewritten.

### Architecture Changes

| Component | Baseline | PR #1019 |
|-----------|----------|----------|
| Layers | 9 | **11** |
| MLP activation | relu² with 2× width | **LeakyReLU(0.5)²** with **3× width** |
| Seq len | 1024 | **2048** |
| Batch tokens | 524K | **786K** |
| Attention backend | PyTorch SDPA | **Flash Attention 3** (Hopper kernels) |
| RoPE | Full (all head dims) | **Partial RoPE** (16 of 64 dims) |
| LN Scale | None | **1/√(layer+1)** per-layer scaling on norm outputs |
| XSA | None | **All 11 layers** — subtracts self-value projection: `y_out = y - (y·v̂)×v̂` |
| SmearGate | None | **Yes** — sigmoid gate mixing current + previous position |
| BigramHash | None | **3072×112** — hash-based bigram embeddings (+ optional trigram) |
| Value Embedding | None | **VE128** on layers 9,10 — shared embedding table + per-layer scale |
| U-Net skips | Yes | Yes (unchanged) |

### Optimizer Changes

| Component | Baseline | PR #1019 |
|-----------|----------|----------|
| Muon | Standard all-reduce flat | **Parallel Muon** — async reduce-scatter → local NS5 on shard → async all-gather. No DDP wrapper. |
| Weight decay | 0.0 | **MuonWD=0.04, AdamWD=0.04** |
| NS5 steps | 10 | **5** (but batched 3D via bank tensors) |
| Momentum | 0.95, warmup 500 steps | **0.99**, warmup from **0.92** over **1500** steps |
| Learning rates | matrix=0.04, scalar=0.04 | matrix=**0.025**, scalar=**0.025** |
| Grad clip | 0.0 (off) | **0.3** |
| Warmdown | 1200 iters | **3500** iters |
| EMA | None | **EMA(0.997)** — shadow state_dict updated every step, applied before export |
| SWA | None | **SWA every 50 steps** during warmdown (scale < 0.2) |
| Late QAT | None | **STE quantization-aware training** activated when LR scale < 0.15 |
| Parameter banking | None | **3D bank tensors** — `qo_bank[2*N, dim, dim]`, `kv_bank[2*N, kv_dim, dim]`, `mlp_up_bank[N, mlp_dim, dim]`, `mlp_down_bank[N, dim, mlp_dim]`. All block matrix weights stacked into contiguous 3D tensors for batched Muon NS5. |

### Quantization & Export Changes

| Component | Baseline | PR #1019 |
|-----------|----------|----------|
| Quant format | int8 per-row (percentile clip) | **int6** per-row for attn+MLP matrices, int8 for embed/small tensors |
| Quant method | Simple percentile clip | **Full Hessian GPTQ** — Cholesky inverse of H=X^TX, block-wise error propagation (block_size=128), actorder column reordering |
| Calibration data | N/A | **AR self-gen** — model generates 64 sequences × 2048 tokens at temp=0.8, no external data |
| Hessian collection | N/A | Separate `_HessianGPT` model (non-banked copy with CastedLinear layers for forward hooks) |
| Compression | zlib level=9 | **LZMA preset=9** |
| Size fitting | None | **Selective ±1 pruning** — sort all ±1 quantized values by reconstruction error (scale²), binary search prune count to fit TARGET_MB |
| Unbanking | N/A | 3D bank tensors split into individual 2D `blocks.{i}.attn.c_q.weight` etc. before quantization, re-banked after dequantization |

### Eval Changes

| Component | Baseline | PR #1019 |
|-----------|----------|----------|
| Eval method | Standard full-sequence | **Sliding window** — stride=64, each token scored with max context |
| Eval implementation | model(x, y) | **compiled `forward_logits`** — separate method returning logits without loss, torch.compiled |
| Eval seq len | =train_seq_len (1024) | **2048** (separate `EVAL_SEQ_LEN` parameter) |
| Post-EMA diagnostic | None | **Yes** — runs eval on EMA weights before quantization to measure pre-quant baseline |

### Key Implementation Details

**Parallel Muon pipeline** (3-phase overlapped):
1. After backward: launch async reduce-scatter for all banks (biggest first)
2. While RS in-flight: all-reduce non-bank grads + step Adam on small params
3. Wait for each RS, run local NS5 on shard, launch async all-gather (overlaps with next bank's NS5)

**XSA implementation** (GQA-aware, no repeat_interleave):
```python
# y: [B,T,H,D], v: [B,T,Hkv,D]
y_g = y.reshape(B, T, Hkv, group, D)
vn = F.normalize(v, dim=-1).unsqueeze(-2)
proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
return (y_g - proj).reshape(B, T, H, D)
```

**GPTQ column reordering** (actorder):
```python
perm = torch.argsort(torch.diag(H), descending=True)  # most-sensitive columns first
W = t32[:, perm].clone()
H = H[perm][:, perm]
Hinv = torch.linalg.cholesky(H)
Hinv = torch.cholesky_inverse(Hinv)
Hinv = torch.linalg.cholesky(Hinv, upper=True)
# block-wise error propagation with Cholesky inverse
```

**Quantization gap**: Pre-quant 1.1354 → Post-GPTQ 1.1377 (+0.0023) → Sliding window **1.1147** (sliding window recovers -0.023 BPB)
