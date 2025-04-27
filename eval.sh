# 使用 0,1,2 三张卡
export HF_HOME=./
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 eval_gsm8k_4shot_parallel.py \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --split test \
    --batch_size 1 \
    --max_samples 100