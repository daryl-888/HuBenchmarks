#!/usr/bin/env python3
"""
!!! DEPRECATED — THIS IS NOT THE METHOD !!!

This file is one of the original Stage-3 "ports". Every one of them was the SAME
Qwen3-VL baseline script applying NO compression — running it produces the plain
backbone number wearing a method's name (the exact silent-no-op failure this
project already had to correct for FastV and PruneVID-OV).

The real implementation lives beside this file as eval_<method>_qwen3vl.py.
Use that, via smoke_<method>_qwen3vl.sbatch.

Kept only so old job scripts fail loudly instead of silently producing a fake
result. See stage3-qwen3-vl/PORT_FEASIBILITY.md.
"""
import sys
print(
    "REFUSING TO RUN: this is the deprecated Stage-3 baseline stub, not the "
    "method. Use eval_<method>_qwen3vl.py instead (see PORT_FEASIBILITY.md)."
, file=sys.stderr)
raise SystemExit(2)   # non-zero: SLURM must see this as a FAILURE
