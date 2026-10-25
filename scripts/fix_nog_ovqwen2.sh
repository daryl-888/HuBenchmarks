#!/bin/bash
# Fix all NOG models in ovqwen2 to use Qwen2 weights and ovqwen2 job names
set -e
cd "/home/daryl/Projects/Video LLM/HuBenchMarks/ovqwen2"

# --- dyto NOG ---
for f in dyto-motionbenc_NOG/run_dyto.sbatch dyto-motionbenc_NOG/test_dyto.sbatch; do
  [ -f "$f" ] && sed -i \
    -e 's|/project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b|/project/rhu/dpalfaro/weights/llava-ov-7b-qwen2|g' \
    -e 's|#SBATCH -J dyto_|#SBATCH -J ovqwen2_dyto_|g' \
    -e 's|--conv-template vicuna_v1|--conv-template qwen_2|g' "$f"
done
echo "dyto NOG: weights→qwen2, conv→qwen_2, job→ovqwen2_dyto"

# --- sttm NOG (was on llava-ov-7b → now qwen2) ---
sed -i \
  -e 's|/project/rhu/dpalfaro/weights/llava-ov-7b |/project/rhu/dpalfaro/weights/llava-ov-7b-qwen2 |g' \
  -e 's|#SBATCH -J sttm_|#SBATCH -J ovqwen2_sttm_|g' \
  sttm-motionbenc_NOG/run_sttm_v2.sbatch
echo "sttm NOG: weights→qwen2, job→ovqwen2_sttm"

# --- sttm-llavavid NOG (was on llava-video-7b → now qwen2) ---
sed -i \
  -e 's|/project/rhu/dpalfaro/weights/llava-video-7b|/project/rhu/dpalfaro/weights/llava-ov-7b-qwen2|g' \
  -e 's|#SBATCH -J sttm_|#SBATCH -J ovqwen2_sttm_llavavid_|g' \
  sttm-llavavid-motionbenc_NOG/run_sttm_llavavid.sbatch
echo "sttm-llavavid NOG: weights→qwen2, job→ovqwen2_sttm_llavavid"

# --- prunevid NOG (was on pllava-7b → now qwen2) ---
sed -i \
  -e 's|ermu2001/pllava-7b|/project/rhu/dpalfaro/weights/llava-ov-7b-qwen2|g' \
  -e 's|#SBATCH -J prunevid_|#SBATCH -J ovqwen2_prunevid_|g' \
  prunevid-motionbenc_NOG/run_prunevid.sbatch
echo "prunevid NOG: weights→qwen2, job→ovqwen2_prunevid"

# --- visionzip NOG (was on llava-v1.5-7b → now qwen2) ---
for f in visionzip-motionbenc_NOG/run_visionzip_32f.sbatch visionzip-motionbenc_NOG/run_visionzip.sbatch; do
  [ -f "$f" ] && sed -i \
    -e 's|/project/rhu/dpalfaro/weights/llava-v1.5-7b|/project/rhu/dpalfaro/weights/llava-ov-7b-qwen2|g' \
    -e 's|#SBATCH -J visionzip_|#SBATCH -J ovqwen2_visionzip_|g' "$f"
done
echo "visionzip NOG: weights→qwen2, job→ovqwen2_visionzip"

# --- imove NOG (was on llava-ov-7b → now qwen2) ---
[ -f imove-motionbenc_NOG/run_imove.sbatch ] && sed -i \
  -e 's|/project/rhu/dpalfaro/weights/llava-ov-7b|/project/rhu/dpalfaro/weights/llava-ov-7b-qwen2|g' \
  -e 's|#SBATCH -J imove_|#SBATCH -J ovqwen2_imove_|g' \
  imove-motionbenc_NOG/run_imove.sbatch
echo "imove NOG: weights→qwen2, job→ovqwen2_imove"

# --- trajvit NOG (was on llava-ov-7b → now qwen2) ---
[ -f trajvit-motionbenc_NOG/run_trajvit.sbatch ] && sed -i \
  -e 's|/project/rhu/dpalfaro/weights/llava-ov-7b|/project/rhu/dpalfaro/weights/llava-ov-7b-qwen2|g' \
  -e 's|#SBATCH -J trajvit_|#SBATCH -J ovqwen2_trajvit_|g' \
  trajvit-motionbenc_NOG/run_trajvit.sbatch
echo "trajvit NOG: weights→qwen2, job→ovqwen2_trajvit"

echo ""
echo "All NOG models updated for ovqwen2"
