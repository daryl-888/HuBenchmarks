# Claude's Active Context — Post Memory-Bank Split (2026-07-09)

## Memory Bank Restructured

Memory bank split into shared reference + per-agent working files:

```
memory-bank/
├── shared/              Read-only reference (both agents)
│   ├── projectbrief.md
│   ├── productContext.md
│   ├── systemPatterns.md
│   ├── techContext.md
│   └── paper-audit.md
├── claude/              Claude's working state (THIS FILE)
│   ├── activeContext.md
│   └── progress.md
└── deepseek/            Other agent's working state
    └── (read their activeContext.md for status)
```

**Rule**: Only write to `claude/`. Read `deepseek/` files to understand what the other agent is doing. Rarely edit `shared/` (only for new models, new paths, new patterns).

## Current Focus

Repository restructured, all sbatch files corrected, 8 ported models need verification runs.

### Job Queue

| JobID | Model | Backbone | State | Notes |
|-------|-------|----------|-------|-------|
| 7662788 | DyTo full run | LLaVA-NeXT-Vicuna-7B | PENDING | Test: 3.7% (expected) |
| 7662787 | AIM full run | LLaVA-OV-7B-Qwen2 | PENDING | boundaries fix applied |
| 7662191 | MDP3 full run | LLaVA-OV-7B | RUNNING | compute-10-4, ~50% data |

### 8 Ported Models Need Verification Runs

- `llava-ov-7b/`: aim, fastvid, flashvid (ported from Qwen2)
- `llava-ov-7b-qwen2/`: dycoke, fastv, holitom, mdp3, videoitg (ported from Qwen 1.5)

### Open Fixes (from paper-audit.md)

1. 🔴 AIM conv_template: qwen_1_5 → qwen_2 (Qwen2 weights need Qwen2 template)
2. 🟡 DyTo default arg: image_seq_v3 → vicuna_v1 in eval_dyto.py line 98
3. 🟡 FastV: note LLaVA-1.5→LLaVA-OV extension in README
