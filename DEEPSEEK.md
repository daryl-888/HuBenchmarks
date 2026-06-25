# DeepSeek on HuBenchmarks

This project can be worked on by either Claude (via Cline) or DeepSeek (via Cline or any tool with DeepSeek API access). The rules, paths, and patches in `CLAUDE.md` apply equally to DeepSeek — treat the name "CLAUDE.md" as a convention, not a restriction.

---

## How to Set Up DeepSeek for This Project

### 1. Get Cline in VS Code
Install the Cline VS Code extension.

### 2. Configure the DeepSeek API key
In Cline settings, set:
- **API Provider:** DeepSeek
- **API Key:** Your DeepSeek API key
- **Model:** `deepseek-chat` (or `deepseek-reasoner` for heavier reasoning)

### 3. Open the repo
```bash
git clone https://github.com/daryl-888/HuBenchmarks.git -b claude/bold-knuth-YlJYI
```
Open the folder in VS Code. Cline will auto-detect the repo context.

### 4. Point DeepSeek at the rules
In your first prompt to DeepSeek, point it at CLAUDE.md:
> Read CLAUDE.md first — it has every rule, path, patch, and gotcha. Do not skip it.

DeepSeek will use CLAUDE.md as its source of truth for cluster paths, patches, and conventions.

---

## What DeepSeek Can Do

- Read/edit all files in this repo
- Follow the same workflow: edit locally → commit → SCP to Carya → sbatch
- Interpret job outputs and results
- Set up new models following the patterns in existing `*-motionbenc/` directories
- Debug inference failures by inspecting logs on Carya

## Limitations

- DeepSeek **cannot SSH** to Carya from the VS Code terminal (same as Claude). You'll need to run `squeue`, `sbatch`, `cat` results, and setup commands manually and paste output back.
- `DEEPSEEK.md` does not replace `CLAUDE.md` — it's a supplement. If your DeepSeek instance is from a different provider (e.g. Cline using DeepSeek backend), it reads both.

---

## Collaboration Flow

Both Claude and DeepSeek read the same CLAUDE.md. When one AI makes progress:
1. It updates CLAUDE.md with new results, patches, or gotchas
2. The changes get committed and pushed to the branch
3. The next AI (whichever it is) picks up the latest CLAUDE.md and continues

**Critical rule:** Do NOT modify completed model directories or their files.
Each `*-motionbenc/` directory that has been run and benchmarked is **frozen**.
New models get their own new directory. The only file you should update
in existing work is CLAUDE.md (results table). The `strict/` directory at
repo root contains reference snapshots of all scripts — never modify those either.
Always add new work on top, never rewrite completed work.

## Completed Models (Frozen — Do Not Modify)

These directories produced final results and must not be changed:
- `dycoke-motionbenc/` — 53.46% (l=3, p=0.7)
- `sttm-v2-motionbenc/` — 54.28% (LLaVA-Video backbone)
- `sttm-llavavid-motionbenc/` — reference
- `holitom-motionbenc/` — 53.11% (RETAIN_RATIO=0.15)
- `videoitg-motionbenc/` — 52.51%
- `prunevid-motionbenc/` — 44.00%
- `flashvid-motionbenc/` — 50.50% (retention=0.1, alpha=0.7)
- `flashvid-strict/` — reference snapshot
- `fastv-motionbenc/` — setup only
- `imove-motionbenc/` — setup only (no weights)
- `trajvit-motionbenc/` — setup only (no weights)

**Branch:** `claude/bold-knuth-YlJYI` (never push to `main`)

---

## Current State (last updated 2026-06-24)

- VisionZip full run: **running** (job 7455129)
- FastVID: **running** (job 7454924)
- MDP3: **setup in progress** — clone repo, create conda env, cache SigLip, then test

See CLAUDE.md for full results table and all patch documentation.
