# 使用 0,1,2 三张卡
export HF_HOME=./
export HF_ENDPOINT=https://hf-mirror.com
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 eval_gsm8k_4shot_parallel.py \
#     --model_name GSAI-ML/LLaDA-8B-Instruct \
#     --split test \
#     --batch_size 1 \
#     --max_samples 100
CUDA_VISIBLE_DEVICES=1 python3 eval_c4_ppl.py --model_name /lpai/volumes/ad-vla-vol-ga/lipengxiang/code/LLaDA/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/9275bf8f5a5687507189baf4657e91c51b2be338 --max_samples 1000 --batch_size 8 --max_length 128 --mc_num 128