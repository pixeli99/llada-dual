# 使用 0,1,2 三张卡
export HF_HOME=/lpai/volumes/ad-vla-vol-ga/lipengxiang/code/LLaDA/huggingface
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0,1,2 python3 eval_gsm8k_4shot_parallel.py \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --split test \
    --batch_size 1 \
    --max_samples 100