# Meta-Prompt: Single-Shot LM Training Script Optimization

## Usage

Feed this prompt to a frontier LLM along with the baseline `train_gpt.py` as input. The output should be a fully rewritten, competition-ready training script.

---

## The Prompt

```
You are an expert ML systems engineer competing in a constrained language model training challenge. Your task is to rewrite the provided baseline training script to achieve the lowest possible validation bits-per-byte (val_bpb).

## Constraints

- Hardware: 8× H100 SXM GPUs, 80GB each
- Wall clock: 600 seconds (10 minutes) hard cap
- Artifact size: 16 MB maximum (model weights + code)
- Metric: val_bpb (bits per byte) on a held-out FineWeb validation set, lower is better
- The script must be self-contained: training, quantization, export, and evaluation in one file
- No external data access after training starts (no val/train data for calibration)
- Tokenizer can be changed but evaluation is tokenizer-agnostic (BPB, not loss)

## Optimization Axes

You must simultaneously optimize across four independent axes. Each axis contributes independently to the final score:

### Axis 1: Training Efficiency (steps × quality per step)
The score is determined by how many gradient steps you complete AND how much each step improves the model. Optimize both:

- **Throughput**: Minimize ms/step. Use kernel-level optimizations (Flash Attention 3, fused ops), overlap communication with computation, eliminate synchronization barriers. Every saved millisecond = more training steps.
- **Model capacity**: Maximize learnable capacity within the throughput budget. More layers, wider MLPs, larger sequence lengths — but only if ms/step stays competitive.
- **Optimizer quality**: The optimizer must extract maximum improvement per step. Consider Muon (Newton-Schulz orthogonalized updates) with momentum warmup, weight decay, gradient clipping, and learning rate warmdown schedules tuned for the short-horizon regime.
- **Auxiliary signals**: Add cheap training signals that improve generalization without slowing the loop. Bigram/n-gram hash embeddings, position-mixing gates, value embeddings at specific layers, residual connections across the U-Net encoder-decoder structure.
- **Weight averaging**: EMA and/or SWA to smooth the optimization trajectory and reduce generalization gap.

### Axis 2: Architecture Design
Design choices that improve the model's expressiveness per parameter:

- **Attention**: GQA (grouped query attention), QK normalization with learnable gain, partial RoPE (apply rotary to subset of head dims), cross-sequence attention (XSA — subtract self-value projection to force cross-position information flow).
- **MLP**: LeakyReLU with large negative slope (0.5) squared — preserves gradient flow through negative activations while maintaining the sparsity benefit of squaring.
- **Normalization**: Per-layer scaling on norm outputs (1/√(layer+1)) to stabilize deep networks.
- **Depth**: U-Net skip connections (encoder stores, decoder reuses in reverse). Consider depth recurrence (looping a subset of layers 2-3× to create virtual depth from shared parameters).
- **Embeddings**: Larger vocab tokenizers (SP4096/SP8192) compress text into fewer tokens, improving per-byte modeling. Hash-based bigram embeddings inject local context at zero attention cost.

### Axis 3: Quantization & Export
The 16MB artifact limit means aggressive quantization is mandatory. The quantization method is as important as the model architecture:

- **Format**: int6 (6-bit, ±31 range) for large matrices, int8 for embeddings, fp16 passthrough for small control tensors.
- **Method**: Full Hessian GPTQ — collect H = X^T X via forward hooks, Cholesky decomposition, block-wise error propagation with column reordering (actorder). This is strictly superior to naive percentile clipping.
- **Calibration**: The model must generate its own calibration data autoregressively (no external data access). Generate ~64 sequences of ~2048 tokens at temperature 0.8.
- **Size fitting**: After quantization, if the artifact exceeds the size limit, selectively prune ±1 quantized values sorted by reconstruction error (scale²). Binary search for the minimum pruning needed.
- **Compression**: LZMA or Brotli compression on the serialized checkpoint.
- **Late QAT**: Enable quantization-aware training (straight-through estimator) during the warmdown phase to pre-adapt weights to quantization grid.

### Axis 4: Evaluation Policy
The evaluation method itself affects the score. Optimize how the artifact is scored:

- **Sliding window**: Instead of evaluating each sequence independently, use overlapping windows with a stride (e.g., 64 tokens). Each token is scored with the maximum available context. This recovers ~0.02 BPB over standard evaluation.
- **Sequence length**: Evaluate at the training sequence length or longer. Separate EVAL_SEQ_LEN from TRAIN_SEQ_LEN.
- **Compilation**: torch.compile the inference path (forward_logits) for faster eval.

## Engineering Requirements

- **No DDP wrapper**: For the main model, handle gradient communication manually. Use async reduce-scatter → local computation → async all-gather to overlap communication with optimizer work.
- **Parameter banking**: Stack all block-level matrix weights into contiguous 3D tensors (e.g., qo_bank[2*N, dim, dim]). This enables batched Newton-Schulz and efficient gradient communication.
- **Separate Hessian model**: For GPTQ, create a parallel non-banked model with standard nn.Linear layers that can attach forward hooks. Load the trained weights into it, collect Hessians, then quantize.
- **All configuration via environment variables**: The script should be fully configurable without code changes.

## Output Format

Produce a single, complete, self-contained Python training script. It must:
1. Train the model for up to 600 seconds on 8× H100
2. Apply EMA/SWA weight averaging
3. Run full Hessian GPTQ with self-generated calibration
4. Export a compressed artifact under 16MB
5. Evaluate with sliding window and report val_bpb
6. Be runnable via `torchrun --standalone --nproc_per_node=8 train_gpt.py`

Do not add comments explaining what you did. Do not include unused code paths. Do not include features that are disabled by default unless they are genuinely useful as env-var toggles. Maximize density and correctness.

## Baseline Script

<paste baseline train_gpt.py here>
```

---

## Why This Works

The prompt is structured around four key insights:

1. **Explicit axis decomposition**: Most people think "make the model better." This prompt forces the LLM to think about training efficiency, architecture, quantization, and eval as independent optimization surfaces. Each axis has ~0.01-0.03 BPB of headroom, and they stack multiplicatively.

2. **Concrete technique enumeration**: Instead of saying "use good quantization," the prompt names GPTQ, Cholesky, actorder, ±1 pruning. This eliminates the search over *what* to do and focuses the LLM on *how* to implement it correctly.

3. **Engineering constraints as design drivers**: The 16MB limit isn't just a constraint — it drives the entire quantization pipeline design. The 600s limit drives the throughput-first optimizer design. The prompt makes these causal links explicit.

4. **Anti-patterns excluded**: "Do not add comments," "do not include unused code paths" — prevents the LLM from padding the output with explanatory text instead of optimizing.

## Limitations

- This prompt assumes knowledge of specific techniques (Flash Attention 3, Muon, GPTQ). A truly general version would need to describe the mathematical foundations.
- The technique list is frozen to ~Apr 2026 SOTA. New techniques (depth recurrence details, SLOT, discriminative TTT) would need to be added as they emerge.
- Single-shot generation of 2000+ lines of correct CUDA-aware distributed training code is at the frontier of current LLM capability. Expect to need 1-2 rounds of debugging.
- The prompt encodes the *winning* stack. It does not encode the search process that discovered it. For novel competition settings, you'd need a different meta-prompt focused on exploration rather than exploitation.
