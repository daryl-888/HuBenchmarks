# Carya patches

These files patch source repos that cannot be modified via pip.
Apply them on Carya after cloning/resetting the relevant repo.

## PruneVid

Source repo: `/project/rhu/dpalfaro/code/PruneVid`

| Patch file | Target on Carya |
|---|---|
| `prunevid/llama.py` | `PruneVid/models/pllava/llama.py` |
| `prunevid/modeling_pllava.py` | `PruneVid/models/pllava/modeling_pllava.py` |

### `prunevid/llama.py`
- Added custom `CausalLMOutputWithPast` dataclass to avoid import from transformers internals
- Fixed `getattr(config, 'mlp_bias', False)` for newer transformers that dropped the attribute

### `prunevid/modeling_pllava.py`
- Fixed shape mismatch in VTP (Video Token Pruning) token selection loop
- Pruning index computation was off-by-one when `cluster_ratio < 1.0`

Apply:
```bash
cp /project/rhu/dpalfaro/code/holitom-motionbenc/patches/prunevid/llama.py \
   /project/rhu/dpalfaro/code/PruneVid/models/pllava/llama.py

cp /project/rhu/dpalfaro/code/holitom-motionbenc/patches/prunevid/modeling_pllava.py \
   /project/rhu/dpalfaro/code/PruneVid/models/pllava/modeling_pllava.py
```

---

## HoliTom

Source repo: `/project/rhu/dpalfaro/code/HoliTom`

| Patch file | Target on Carya |
|---|---|
| `holitom/modeling_qwen2.py` | `HoliTom/holitom/modeling_qwen2.py` |
| `holitom/builder.py` | `HoliTom/LLaVA-NeXT/llava/model/builder.py` |
| `holitom/siglip_encoder.py` | `HoliTom/LLaVA-NeXT/llava/model/multimodal_encoder/siglip_encoder.py` |
| `holitom/llava_init.py` | `HoliTom/LLaVA-NeXT/llava/__init__.py` |

### `holitom/modeling_qwen2.py`
Stubs for symbols absent in transformers 4.45.2 that HoliTom's code references:
- `FlashAttentionKwargs` — replaced with empty dataclass
- `dynamic_rope_update` — replaced with identity decorator
- `ALL_ATTENTION_FUNCTIONS` — replaced with empty dict
- `Unpack` — replaced with `typing.Any`
- `can_return_tuple` — replaced with identity decorator
- `replace_return_docstrings` — replaced with identity decorator
- `deprecate_kwarg` — replaced with identity decorator
- `SlidingWindowCache`, `StaticCache` — replaced with `object`

### `holitom/builder.py`
- `low_cpu_mem_usage=True` must remain set (needed by `device_map="auto"` in the function)
- Nothing to change here — the key fix is removing `device_map="auto"` from `eval_holitom.py`

### `holitom/siglip_encoder.py`
- Removed `device_map=device_map` from inner `SigLipVisionModel.from_pretrained` call
- Prevents nested meta device context when outer model uses `device_map="auto"`

### `holitom/llava_init.py`
- Wrapped `LlavaLlamaForCausalLM` import in try/except
- LLaMA model not installed in holitom env; import crash prevented

Apply:
```bash
HOLITOM=/project/rhu/dpalfaro/code/HoliTom
PATCHES=/project/rhu/dpalfaro/code/holitom-motionbenc/patches/holitom

cp $PATCHES/modeling_qwen2.py $HOLITOM/holitom/modeling_qwen2.py
cp $PATCHES/builder.py        $HOLITOM/LLaVA-NeXT/llava/model/builder.py
cp $PATCHES/siglip_encoder.py $HOLITOM/LLaVA-NeXT/llava/model/multimodal_encoder/siglip_encoder.py
cp $PATCHES/llava_init.py     $HOLITOM/LLaVA-NeXT/llava/__init__.py
```

---

## Conda env patch — transformers 4.45.2 rope fix

Applied directly to the installed package in the holitom conda env.
**Cannot be stored as a file patch — must be re-applied if the env is recreated.**

File: `/project/rhu/dpalfaro/conda/envs/holitom/lib/python3.11/site-packages/transformers/models/qwen2/modeling_qwen2.py`

Line ~61 (inside `_compute_default_rope_parameters`):
```python
# Before:
rope_type = rope_scaling["rope_type"] if rope_scaling is not None else "default"
# After:
rope_type = (rope_scaling or {}).get("rope_type", "default")
```

Line ~90 (inside `_compute_llama3_parameters` or similar):
```python
# Before:
base = rope_scaling["rope_theta"]
# After:
base = (rope_scaling or {}).get("rope_theta", 1000000.0)
```

To re-apply:
```bash
/project/rhu/dpalfaro/conda/envs/holitom/bin/pip install "transformers==4.45.2"
# Then hand-edit the two lines above in the installed qwen2 modeling file
```

---

## Why transformers 4.45.2?

HoliTom's `holitom/modeling_qwen2.py` patches Qwen2's attention to read attention weights.
It was written for transformers ~4.45. The holitom conda env shipped with ≥4.47, which:
- Added `FlashAttentionKwargs`, `dynamic_rope_update`, `ALL_ATTENTION_FUNCTIONS`, etc.
- Changed `SlidingWindowCache` / `StaticCache` signatures
- Changed `can_return_tuple`, `replace_return_docstrings`, `deprecate_kwarg` locations

Downgrading to 4.45.2 fixes all of these cleanly. The conda env stubs in
`holitom/modeling_qwen2.py` provide backward-compat for any remaining differences.
