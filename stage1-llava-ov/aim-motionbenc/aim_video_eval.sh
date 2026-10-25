# system paths
# export PYTHONPATH=/AIM
# export HF_HOME=/huggingface_cache

# AIM with base model as LLaVA-OneVision-7B
accelerate launch --num_processes=2 --gpu_ids 0,1 --main_process_port 25666 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,attn_implementation=eager \
--tasks videomme \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/
# AIM pruning: attn_implementation=eager
# More sampled frames: max_frames_num=192
# LLaVA-Prumerge: mm_spatial_pool_mode=identity
# tokenizer_model_max_length: the max length to pad in llava_arch.py

# Task Names: 
# mlvu,nextqa_mc_test,egoschema,videomme,videomme_w_subtitle,perceptiontest_val_mc,mvbench

# Egoschema:
# submit json to server for full benchmark evaluation
# curl -X POST -H "Content-Type: application/json" -d @/PATH_to_RESULT_FILE.json https://validation-server.onrender.com/api/upload/


#############################################################################################
#############################################################################################

# AIM with base model as Qwen2-VL-7B-Instruct
# accelerate launch --num_processes=2 --gpu_ids 0,1 --main_process_port 25666 \
# -m lmms_eval \
# --model qwen2_vl \
# --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct,max_num_frames=32,max_pixels=200704 \
# --tasks videomme_w_subtitle \
# --batch_size 1 \
# --log_samples \
# --log_samples_suffix qwen2_vl \
# --output_path ./logs/
# # Qwen2-VL-7B-Instruct: lmms_eval/models/qwen2_vl.py 
# # https://github.com/EvolvingLMMs-Lab/lmms-eval/issues/436
# # https://github.com/QwenLM/Qwen2-VL/issues/137
# # https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/346/commits/350654b92d471f532550ac3dc698e2983427de5c