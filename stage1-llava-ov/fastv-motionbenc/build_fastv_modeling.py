#!/usr/bin/env python3
"""
build_fastv_modeling.py — generate a FastV-patched modeling_qwen2.py from DyCoke's.

Run this ON CARYA. It copies DyCoke's already-patched modeling_qwen2.py (which has
PrunableDynamicCache + the in-loop pruning hooks) and injects a FastV code path
that mirrors DyCoke's mechanism with FastV's simpler static policy:

  - prune ONCE, at layer fastv_k, during prefill
  - rank image tokens by mean received attention at that layer
  - keep the top (1 - fastv_r) fraction; set cache.kv_cache to the kept indices
  - PrunableDynamicCache.update() then gathers only kept tokens for all later
    layers and every decode step (no further work needed)

The output goes to a FastV-owned package dir so DyCoke's copy is untouched.

Usage (on Carya):
  python build_fastv_modeling.py \
    --src /project/rhu/dpalfaro/code/DyCoke/llava/model/language_model/modeling_qwen2.py \
    --out /project/rhu/dpalfaro/code/fastv_llava/llava/model/language_model/modeling_qwen2.py
"""
import argparse
import os
import re
import shutil
import sys


# enable_fastv() added to Qwen2Model; plus fastv_prune() on the cache class.
ENABLE_FASTV_METHOD = '''
    def enable_fastv(self, fastv_k, fastv_r, image_token_start_index):
        """FastV static pruning config. Set by eval_fastv.apply_fastv()."""
        self.fastv = True
        self.fastv_k = int(fastv_k)
        self.fastv_r = float(fastv_r)
        self.fastv_img_start = int(image_token_start_index)
        # Disable DyCoke path if present so the two don't fight.
        self.dycoke = None
'''

FASTV_PRUNE_METHOD = '''
    def fastv_prune(self, attn, config_start, config_img_len, fastv_r):
        """
        FastV: rank image tokens by mean received attention (last query row,
        averaged over heads) and keep the top (1 - fastv_r) fraction. Set
        self.kv_cache so update() gathers only kept tokens thereafter.
        Mirrors dycoke_pruning()/update_cache() but unconditional and one-shot.
        """
        attention_avg = attn.mean(1)[0, -1]                     # (kv_len,)
        image_attention = attention_avg[config_start:config_start + config_img_len]
        num_keep = max(1, int(config_img_len * (1 - fastv_r)))
        top_indices = torch.topk(image_attention, num_keep, sorted=False)[1] + config_start
        device = image_attention.device
        full_range = torch.arange(config_start + config_img_len + self._fastv_tail, device=device)
        keep_indexs = torch.cat([
            full_range[:config_start],
            top_indices.sort()[0],
            full_range[config_start + config_img_len:],
        ])
        self.kv_cache = keep_indexs.tolist()
'''


def patch(src_text: str) -> str:
    t = src_text

    # 1) Ensure Qwen2Model has a `fastv` attribute default so the loop check is safe.
    #    DyCoke sets self.dycoke in __init__; add self.fastv next to it.
    if "self.fastv = None" not in t:
        t = re.sub(
            r"(\n(\s+)self\.dycoke\s*=\s*None)",
            r"\1\n\2self.fastv = None\n\2self.fastv_k = 0\n\2self.fastv_r = 0.0"
            r"\n\2self.fastv_img_start = 14",
            t, count=1,
        )
        if "self.fastv = None" not in t:
            # Fall back: DyCoke may not init self.dycoke in this class; inject after
            # the Qwen2Model.__init__ super().__init__ call is too fragile, so require
            # the anchor and fail loudly rather than silently no-op.
            raise SystemExit("PATCH FAIL: could not find `self.dycoke = None` anchor "
                             "to add fastv defaults. Inspect the source.")

    # 2) Add enable_fastv() + fastv_prune(). Attach enable_fastv to Qwen2Model
    #    (anchor: its class body — reuse the dycoke method as a neighbor) and
    #    fastv_prune to PrunableDynamicCache (anchor: def dycoke_pruning).
    if "def enable_fastv(" not in t:
        # place enable_fastv right before the decoder loop method set — anchor on
        # the first occurrence of "    def forward(" inside Qwen2Model is risky;
        # instead anchor on the class's get_input_embeddings if present, else on
        # "class Qwen2Model".
        m = re.search(r"\n(class Qwen2Model\b.*?:\n)", t, re.S)
        if not m:
            raise SystemExit("PATCH FAIL: no `class Qwen2Model` found.")
        insert_at = m.end()
        t = t[:insert_at] + ENABLE_FASTV_METHOD + t[insert_at:]

    if "def fastv_prune(" not in t:
        anchor = "    def dycoke_pruning(self"
        idx = t.find(anchor)
        if idx == -1:
            raise SystemExit("PATCH FAIL: no `dycoke_pruning` anchor on cache class.")
        t = t[:idx] + FASTV_PRUNE_METHOD + "\n" + t[idx:]

    # 3) Inject the FastV branch into the decoder loop, right where DyCoke's
    #    branch lives. Anchor: the `if self.dycoke:` block start.
    if "if self.fastv:" not in t:
        anchor = "                if self.dycoke:"
        idx = t.find(anchor)
        if idx == -1:
            raise SystemExit("PATCH FAIL: no `if self.dycoke:` loop anchor.")
        fastv_branch = (
            "                if self.fastv:\n"
            "                    # FastV: at layer fastv_k, prune image tokens once\n"
            "                    # using attention captured from layer fastv_k-1's output.\n"
            "                    if layer_idx <= self.fastv_k:\n"
            "                        past_key_values.kv_cache = None\n"
            "                    if layer_idx == self.fastv_k and getattr(past_key_values, 'kv_cache', None) is None:\n"
            "                        _seq = seq_length + past_key_values_length\n"
            "                        _tail = _seq - self.fastv_img_start - self._fastv_img_len if hasattr(self, '_fastv_img_len') else 0\n"
            "                        past_key_values._fastv_tail = _tail\n"
            "                        if output_attentions and layer_outputs is not None and len(layer_outputs) > 1 and layer_outputs[1] is not None:\n"
            "                            past_key_values.fastv_prune(layer_outputs[1], self.fastv_img_start, self._fastv_img_len, self.fastv_r)\n"
            "                    layer_outputs = decoder_layer(\n"
            "                        hidden_states,\n"
            "                        attention_mask=None,\n"
            "                        position_ids=position_ids,\n"
            "                        past_key_value=past_key_values,\n"
            "                        output_attentions=output_attentions,\n"
            "                        use_cache=use_cache,\n"
            "                    )\n"
            "                elif self.dycoke:\n"
        )
        # replace the leading "if self.dycoke:" with fastv branch + "elif self.dycoke:"
        t = t[:idx] + fastv_branch + t[idx + len(anchor):]

    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.src) as f:
        src = f.read()
    out = patch(src)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(out)
    # sanity: it must at least compile
    import py_compile
    py_compile.compile(args.out, doraise=True)
    print(f"OK wrote + compiled {args.out}")
    print("Injected: enable_fastv, fastv_prune, FastV loop branch.")


if __name__ == "__main__":
    main()
