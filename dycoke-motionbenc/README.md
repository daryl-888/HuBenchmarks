# DyCoke × MotionBench Evaluation

Evaluates DyCoke — a training-free KV cache compression method for video LLMs — on the MotionBench benchmark using LLaVA-OneVision-7B on the UH Carya HPC cluster.

---

## What is DyCoke?

DyCoke (Dynamic Token Compression for Video LLMs, arXiv 2411.14401) reduces inference cost by pruning redundant visual tokens from the KV cache without retraining the model. It has two stages:

- **Stage 1 (K):** Temporal token merging — merges similar tokens across frames to reduce redundancy
- **Stage 2 (P):** Dynamic KV cache pruning — drops low-attention tokens at a specified layer during inference

It is plug-and-play: enabled by passing flags to the model at inference time.

## What is MotionBench?

MotionBench is a video understanding benchmark focused on motion comprehension. It has 8,052 multiple-choice QA samples across 6 categories:

- Action Order
- Camera Motion
- Location-related Motion
- Motion-related Objects
- Motion Recognition
- Repetition Count

Videos are stored at `/project/rhu/MotionBench_Data/MotionBench/` in two subdirectories: `self-collected/` and `public-dataset/`.

---

## Why this combination?

The DyCoke paper evaluates on ActivityNet-QA, NextQA, PerceptionTest, VideoMME, MVBench, and VideoDetailCaption. MotionBench is not in the original paper. This project extends DyCoke's evaluation to MotionBench to test whether its token compression degrades performance on motion-specific understanding tasks.

---

## Repository Structure

```
dycoke-motionbenc/
├── run_dycoke.sbatch        # Main SLURM job — runs DyCoke inference on MotionBench
├── cache_dataset.sbatch     # One-time job — caches the dataset before first run
└── tasks/
    └── motionbench/
        ├── motionbench.yaml # lmms_eval task definition
        └── utils.py         # Dataset loading and scoring functions
```

---

## DyCoke Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `dycoke` | True | Enables DyCoke compression |
| `dycoke_l` | 3 | Layer used to evaluate attention for pruning decisions |
| `dycoke_p` | 0.8 | Stage 2 KV cache pruning rate |
| `dycoke_k` | 0.3 | Stage 1 temporal token merging rate |

These match the values used in the DyCoke paper for LLaVA-OV-7B.

---

## How the Code Runs

### 1. Dataset loading (`motionbench.yaml`)

`lmms_eval` loads the dataset from the raw JSONL file:
```
/project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl
```

Each record contains a video path and a QA pair. The dataset is cached to:
```
/project/rhu/dpalfaro/cache/huggingface/datasets/
```

### 2. Video loading (`utils.py`)

For each sample, `motionbench_doc_to_visual()` locates the video file by checking both subdirectories (`self-collected/`, `public-dataset/`).

### 3. Inference (`run_dycoke.sbatch`)

`lmms_eval` feeds each video + question to LLaVA-OV-7B with DyCoke enabled. The model generates a short answer (max 16 tokens). Greedy decoding is used (temperature=0, no sampling).

### 4. Scoring (`utils.py`)

`motionbench_process_results()` extracts the predicted letter (A-D) from the model output using regex and compares it to the ground truth. `motionbench_aggregate_results()` computes overall accuracy.

### 5. Output

Results are saved to `/project/rhu/dpalfaro/results/dycoke_run1/`.

---

## Cluster Setup (UH Carya)

Carya is a GPU cluster managed by SLURM. Jobs are submitted with `sbatch` and never run interactively on the login node.

### Key paths on Carya

| Path | Purpose |
|------|---------|
| `/project/rhu/dpalfaro/code/DyCoke` | DyCoke source code (added to PYTHONPATH) |
| `/project/rhu/dpalfaro/code/dycoke-motionbenc` | This repo |
| `/project/rhu/dpalfaro/weights/llava-ov-7b` | LLaVA-OneVision-7B model weights |
| `/project/rhu/dpalfaro/conda/envs/dycoke11` | Conda environment with all dependencies |
| `/project/rhu/dpalfaro/cache/huggingface` | Hugging Face model and dataset cache |
| `/project/rhu/dpalfaro/results` | Job output and evaluation results |
| `/project/rhu/MotionBench_Data/MotionBench` | MotionBench videos and metadata |

### Environment variables

| Variable | Value | Reason |
|----------|-------|--------|
| `PYTHONNOUSERSITE` | 1 | Prevents user site-packages from interfering with the conda env |
| `PYTHONPATH` | includes DyCoke root | lmms_eval inside DyCoke needs to be importable |
| `HF_HOME` | `/project/rhu/dpalfaro/cache/huggingface` | Points HF to project storage instead of home dir (10GB limit) |
| `HF_DATASETS_OFFLINE` | 1 | Prevents HF from trying to reach the internet (cluster has no outbound access) |
| `TRANSFORMERS_OFFLINE` | 1 | Same as above for transformers |

### Node pinning

Jobs are pinned to `compute-9-1` via `#SBATCH --nodelist=compute-9-1`. This is because other nodes on the cluster have an outdated NVIDIA driver (version 12090) that is incompatible with the PyTorch version in the `dycoke11` environment. `compute-9-1` has been verified to work and has a 45GB GPU.

---

## How to Run

### First time only — cache the dataset

The first time the dataset is used, Hugging Face must convert the raw JSONL file into its internal binary format. This is a one-time operation that takes a few minutes as a batch job.

```bash
sbatch cache_dataset.sbatch
# Note the job ID printed, e.g. 6967542
```

### Submit the main job (after cache completes)

```bash
sbatch --dependency=afterok:<CACHE_JOBID> run_dycoke.sbatch
```

Or if the cache is already built from a previous run:

```bash
sbatch run_dycoke.sbatch
```

### Monitor progress

```bash
squeue -u dpalfaro
tail -f /project/rhu/dpalfaro/results/dycoke_<JOBID>.out
```

### Expected runtime

- Dataset cache load: seconds (after first run)
- Inference: ~3.5 hours (8052 samples at ~1.53s/sample)
- Time limit set to 6 hours as buffer

---

## Uploading Updated Files to Carya

This project is edited locally and uploaded via `scp`. SSH key authentication is configured.

```bash
scp run_dycoke.sbatch dpalfaro@carya.rcdc.uh.edu:/project/rhu/dpalfaro/code/dycoke-motionbenc/
scp cache_dataset.sbatch dpalfaro@carya.rcdc.uh.edu:/project/rhu/dpalfaro/code/dycoke-motionbenc/
scp tasks/motionbench/utils.py dpalfaro@carya.rcdc.uh.edu:/project/rhu/dpalfaro/code/dycoke-motionbenc/tasks/motionbench/
scp tasks/motionbench/motionbench.yaml dpalfaro@carya.rcdc.uh.edu:/project/rhu/dpalfaro/code/dycoke-motionbenc/tasks/motionbench/
```

Run all `scp` commands from your **local machine**, not from inside the Carya SSH session.
