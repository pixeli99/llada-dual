#!/usr/bin/env python3
# eval_c4_ppl.py
# ---------------------------------------------------------
# Evaluate perplexity of LLaDA model on C4 dataset
# ---------------------------------------------------------

import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
from math import exp

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from get_log_likelihood import get_log_likelihood, get_logits, forward_process

# ----------------------------------------------------------------------
# Core evaluation loop (single GPU / single process)
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_c4_ppl_batched(
    model,
    tokenizer,
    batch_size: int = 8,
    max_samples: int | None = None,
    max_length: int = 512,
    rank: int = 0,
    world_size: int = 1,
    mc_num: int = 128,
    cfg_scale: float = 0.0,
):
    """Evaluate perplexity on C4 dataset slice handled by this rank."""
    # 加载C4数据集(验证集)
    ds = load_dataset("/lpai/dataset/lpx-hf-cache/0-1-0/hf_cache/c4/en", split="validation", streaming=True, trust_remote_code=True)
    if max_samples is not None:
        ds = ds.take(max_samples)
    
    # 处理属于当前rank的样本
    samples_processed = 0
    total_log_likelihood = 0.0
    total_tokens = 0
    
    # 为rank 0设置进度条
    iterator = ds
    if rank == 0:
        iterator = tqdm(iterator, desc=f"[GPU {rank}]", total=max_samples)
    
    # 处理批次
    batch_texts = []
    for i, example in enumerate(iterator):
        # 只处理属于本rank的样本
        if i % world_size != rank:
            continue
        
        text = example["text"].strip()
        batch_texts.append(text)
        
        # 当批次填满时处理
        if len(batch_texts) == batch_size:
            log_ll, tokens = process_batch(model, tokenizer, batch_texts, max_length, mc_num, cfg_scale)
            total_log_likelihood += log_ll
            total_tokens += tokens
            samples_processed += len(batch_texts)
            batch_texts = []
    
    # 处理剩余样本
    if batch_texts:
        log_ll, tokens = process_batch(model, tokenizer, batch_texts, max_length, mc_num, cfg_scale)
        total_log_likelihood += log_ll
        total_tokens += tokens
        samples_processed += len(batch_texts)
    
    # 计算平均对数似然和困惑度
    return total_log_likelihood, total_tokens, samples_processed

def process_batch(model, tokenizer, batch_texts, max_length, mc_num, cfg_scale):
    """处理一批文本并返回总对数似然和token数量"""
    total_log_ll = 0.0
    total_tokens = 0
    
    mask_id = 126336  # LLaDA的[MASK] token ID
    context_length = 64  # 每个块的上下文/prompt长度
    target_length = 64   # 每个块的目标/answer长度
    
    for text in batch_texts:
        # 截断到max_length进行tokenize
        tokens = tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
        tokens = torch.tensor(tokens).to(model.device)
        
        if len(tokens) <= context_length:
            continue  # 跳过太短的文本
        
        # 将文本分成块并计算对数似然
        # 第一个块：使用小前缀作为prompt
        prompt = tokens[:context_length]
        answer = tokens[context_length:context_length+target_length]
        
        if len(answer) == 0:
            continue
            
        log_ll = get_log_likelihood(
            model=model,
            prompt=prompt,
            answer=answer,
            mc_num=mc_num,
            batch_size=min(8, len(answer)),  # 基于answer长度调整batch size
            cfg_scale=cfg_scale,
            mask_id=mask_id
        )
        
        total_log_ll += log_ll
        total_tokens += len(answer)
        
        # 处理后续块
        offset = context_length + target_length
        while offset < len(tokens):
            # 使用前面的块作为上下文
            context_start = max(0, offset - context_length)
            prompt = tokens[context_start:offset]
            
            # 获取下一个块作为answer
            answer_end = min(offset + target_length, len(tokens))
            answer = tokens[offset:answer_end]
            
            if len(answer) == 0:
                break
                
            log_ll = get_log_likelihood(
                model=model,
                prompt=prompt,
                answer=answer,
                mc_num=mc_num,
                batch_size=min(8, len(answer)),
                cfg_scale=cfg_scale,
                mask_id=mask_id
            )
            
            total_log_ll += log_ll
            total_tokens += len(answer)
            
            offset = answer_end
    
    return total_log_ll, total_tokens

# ----------------------------------------------------------------------
# Worker进程(每个GPU一个)
# ----------------------------------------------------------------------
def worker(rank: int, world_size: int, args, queue: mp.SimpleQueue):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # -- 加载模型/tokenizer ------------------------------------------
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # -- 评估分片 -----------------------------------------------
    log_likelihood, tokens, samples = eval_c4_ppl_batched(
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_length=args.max_length,
        rank=rank,
        world_size=world_size,
        mc_num=args.mc_num,
        cfg_scale=args.cfg_scale,
    )

    # -- 返回结果 --------------------------------------------
    queue.put((log_likelihood, tokens, samples))

# ----------------------------------------------------------------------
# 入口点
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="多GPU的C4困惑度评估脚本，用于LLaDA模型"
    )
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--mc_num", type=int, default=128)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("没有可见的CUDA设备!")

    mp.set_start_method("spawn", force=True)
    queue: mp.SimpleQueue = mp.SimpleQueue()

    print(f"启动{world_size}个工作进程...")
    mp.spawn(
        fn=worker,
        args=(world_size, args, queue),
        nprocs=world_size,
        join=True,
    )

    # ---- 聚合结果 ---------------------------------------------
    total_log_likelihood = 0.0
    total_tokens = 0
    total_samples = 0
    
    while not queue.empty():
        ll, tokens, samples = queue.get()
        total_log_likelihood += ll
        total_tokens += tokens
        total_samples += samples

    # 计算困惑度 PPL = exp(-avg_log_likelihood)
    avg_log_likelihood = total_log_likelihood / total_tokens if total_tokens else 0.0
    perplexity = exp(-avg_log_likelihood)
    
    print(f"[所有GPU] C4评估结果:")
    print(f"总样本数: {total_samples}")
    print(f"总token数: {total_tokens}")
    print(f"每token平均对数似然: {avg_log_likelihood:.4f}")
    print(f"困惑度(PPL): {perplexity:.4f}")

if __name__ == "__main__":
    main()
