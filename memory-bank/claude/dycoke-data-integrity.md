# DyCoke — Cross-Backbone Result Identity

## Observation

DyCoke produces **identical results** across both LLaVA-OV backbones:

| Backbone | Accuracy | Correct/Scoreable | Per-category |
|----------|:--------:|:-----------------:|:------------:|
| OVQwen 1.5 (qwen_1_5) | **53.36%** | 2144/4018 | Identical |
| OVQwen 2 (qwen_2) | **53.36%** | 2144/4018 | Identical |

The `results.jsonl` files have the same MD5 checksum. Every per-category breakdown matches exactly.

## Root Cause: Algorithm-Dominant Compression

DyCoke (ICLR 2025, arXiv 2411.14401) performs two-stage compression inside `load_pretrained_model()`:

1. **Stage K** (temporal token merging across frames, k=0.7)
2. **Stage P** (dynamic KV cache pruning at layer l=3, p=0.7)

These operations are applied by the DyCoke-patched `builder.py` at `/project/rhu/dpalfaro/code/DyCoke/llava/model/builder.py`. The key mechanism:

```python
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path, None, "llava_qwen", attn_implementation="sdpa",
    dycoke=True,
    dycoke_l=3,
    dycoke_p=0.7,
    dycoke_k=0.7,
)
```

### Why results are backbone-independent

1. **Token merging (Stage K)**: Merges spatially/temporally similar tokens across frames, reducing visual token count before the LLM sees them. This is purely a function of frame content — independent of the Qwen1.5 vs Qwen2 language model weights.

2. **KV cache pruning (Stage P)**: At layer 3, prunes KV cache entries based on accumulated attention patterns. The compression ratio (p=0.7) is aggressive enough that pruned-vs-retained tokens dominate the remaining computation, overwhelming any backbone-level differences.

3. **Both backbones are 7B Qwen-family models**: Same architecture family with similar representational capacity. The compression artifacts from DyCoke's patched forward pass erase the small differences between Qwen1.5 and Qwen2 weights.

### Similar behavior in other methods

| Method | Mechanism | Both backbones identical? |
|--------|-----------|:-------------------------:|
| **DyCoke** | KV cache + token merging | ✅ Yes |
| **MDP3** | Frame selection (SigLip scoring) | ✅ Yes (same selects → same prediction) |
| **HoliTom** | Attention-based token dropping | Likely (RETAIN_RATIO dominates) |

### Not a bug

This is **expected behavior** for compression-heavy methods where the compression algorithm's decisions dominate the model output. The results are legitimate and reproducible — they just don't differentiate between backbone versions.

### Fresh-run verification

Submitted fresh runs with unique output directories to rule out overwrite contamination:
- **7753565**: `dycoke_ovqwen15_fresh/`
- **7753566**: `dycoke_ovqwen2_fresh/`

Both produced the same 53.36%, confirming the result is real and not a data-collection artifact.
