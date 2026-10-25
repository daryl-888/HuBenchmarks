#!/usr/bin/env python3
"""
Patch /project/rhu/dpalfaro/DYTO/dyto/llava/__init__.py
to wrap the LlavaLlamaForCausalLM import in try/except.

DYTO is cloned to /project/rhu/dpalfaro/DYTO (dpalfaro-owned).
/project/rhu/dpalfaro/code/DYTO is owned by mahern69 — do not use.

Run on Carya login node (no GPU needed):
    python3 /project/rhu/dpalfaro/code/dyto-motionbenc/patches/dyto/patch_dyto_init.py
"""
import sys

TARGET = "/project/rhu/dpalfaro/DYTO/dyto/llava/__init__.py"

OLD = "from .model import LlavaLlamaForCausalLM"
NEW = "try:\n    from .model import LlavaLlamaForCausalLM\nexcept ImportError:\n    pass"

with open(TARGET) as f:
    content = f.read()

if OLD not in content:
    if "LlavaLlamaForCausalLM" not in content:
        print("LlavaLlamaForCausalLM not found in file — nothing to patch.")
        sys.exit(0)
    # Already patched or different form
    if "try:" in content and "LlavaLlamaForCausalLM" in content:
        print("Already patched.")
        sys.exit(0)
    print(f"ERROR: Could not find exact string to replace. File contents:")
    print(content)
    sys.exit(1)

patched = content.replace(OLD, NEW, 1)

with open(TARGET, "w") as f:
    f.write(patched)

print(f"Patched {TARGET}")
print("Before:", repr(OLD))
print("After:", repr(NEW))
