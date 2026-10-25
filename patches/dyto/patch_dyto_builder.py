#!/usr/bin/env python3
"""
Patch DYTO/dyto/llava/model/builder.py to explicitly import LlavaLlamaForCausalLM.

The star import `from llava.model import *` in builder.py does not pull in
LlavaLlamaForCausalLM because llava/model/__init__.py doesn't re-export it.
This patch adds an explicit import immediately after the star import.

Run on Carya login node:
    python3 /project/rhu/dpalfaro/code/dyto-motionbenc/patches/dyto/patch_dyto_builder.py
or:
    python3 /project/rhu/dpalfaro/code/patches/dyto/patch_dyto_builder.py
"""

TARGET = "/project/rhu/dpalfaro/DYTO/dyto/llava/model/builder.py"
MARKER = "from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM"
OLD = "from llava.model import *"
NEW = (
    "from llava.model import *\n"
    "from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM"
)

with open(TARGET) as f:
    content = f.read()

if MARKER in content:
    print("Already patched.")
elif OLD not in content:
    print(f"ERROR: pattern not found in {TARGET} — check the file manually.")
else:
    with open(TARGET, "w") as f:
        f.write(content.replace(OLD, NEW, 1))
    print("Patched builder.py — added explicit LlavaLlamaForCausalLM import.")
