#!/usr/bin/env python3
# eval_gsm8k_4shot_parallel.py
# ---------------------------------------------------------------
# Multi-GPU evaluation of a HF model on GSM8K with 4-shot prompts
# ---------------------------------------------------------------

import argparse
import os
import re
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from generate import generate      # <-- 你自己的生成函数


# ----------------------------------------------------------------------
# Utility: extract the *last* number (int/float) from generated text
# ----------------------------------------------------------------------
def extract_numeric_answer(text: str):
    pattern = r"[-+]?\d+(?:\.\d+)?"
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


# ----------------------------------------------------------------------
# Build k-shot prefix from GSM8K train split
# ----------------------------------------------------------------------
def build_few_shot_prefix(k: int = 4, seed: int = 42) -> list[dict]:
    """
    Return a list of k dicts, each with 'question' and 'answer'.
    We choose them *once* per process to avoid cross-process pickling.
    """
    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(train_ds), k, replace=False)
    return [train_ds[int(i)] for i in indices]


def _format_prefix(examples: list[dict]) -> str:
    """Turn example list into multi-example string prefix."""
    parts = []
    for ex in examples:
        q = ex["question"].strip()
        a = ex["answer"].strip()
        parts.append(f"Question: {q}\nAnswer: {a}\n")
    return "\n".join(parts) + "\nQuestion: "  # 末尾留出“Question:”方便拼接


# ----------------------------------------------------------------------
# Core evaluation loop (single GPU / single process)
# ----------------------------------------------------------------------
def eval_gsm8k_batched(
    model,
    tokenizer,
    few_shot_prefix: str,
    batch_size: int = 8,
    split: str = "test",
    max_samples: int | None = None,
    rank: int = 0,
    world_size: int = 1,
):
    """Return (correct, total) on the slice handled by this rank."""
    ds = load_dataset("openai/gsm8k", "main", split=split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    # 下采样：仅保留属于当前 rank 的样本
    ds = ds.select(range(rank, len(ds), world_size))

    correct = total = 0
    iterator = range(0, len(ds), batch_size)
    iterator = tqdm(iterator, desc=f"[GPU {rank}]") if rank == 0 else iterator

    for start in iterator:
        batch = ds[start : start + batch_size]
        # --- 构建 prompts & gold numbers ---------------------------------
        prompts, gold_nums = [], []
        for q, gold in zip(batch["question"], batch["answer"]):
            m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", gold)
            gold_nums.append(m.group(1) if m else None)

            prompt = f"{few_shot_prefix}{q.strip()}\nAnswer:"
            prompts.append(prompt)

        # --- Tokenize (pad left) -----------------------------------------
        tokenizer.padding_side = "left"
        tok = tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(model.device)

        # --- Generate ----------------------------------------------------
        out = generate(
            model=model,
            prompt=tok["input_ids"],
            steps=64,
            gen_length=64,
            block_length=32,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="random",
            tokenizer=tokenizer,
        )

        # --- Decode only new tokens -------------------------------------
        in_lens = (tok["input_ids"].ne(tokenizer.pad_token_id)).sum(dim=1)
        preds = []
        for i, gen in enumerate(out):
            new_tokens = gen[in_lens[i] :]
            preds.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

        # --- Compare -----------------------------------------------------
        for gold, pred in zip(gold_nums, preds):
            if gold is None:
                total += 1
                continue
            pred_num = extract_numeric_answer(pred)
            if pred_num == gold:
                correct += 1
            total += 1

    return correct, total


# ----------------------------------------------------------------------
# Worker process (one per GPU)
# ----------------------------------------------------------------------
def worker(rank: int, world_size: int, args, queue: mp.SimpleQueue):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # -- 1. Load model/tokenizer -----------------------------------------
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # -- 2. Build 4-shot prefix (same across workers) --------------------
    examples = build_few_shot_prefix(k=4, seed=42)
    few_shot_prefix = _format_prefix(examples)

    # -- 3. Evaluate slice ----------------------------------------------
    correct, total = eval_gsm8k_batched(
        model=model,
        tokenizer=tokenizer,
        few_shot_prefix=few_shot_prefix,
        batch_size=args.batch_size,
        split=args.split,
        max_samples=args.max_samples,
        rank=rank,
        world_size=world_size,
    )

    # -- 4. Send back results -------------------------------------------
    queue.put((correct, total))


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU GSM8K 4-shot evaluation script"
    )
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices visible!")

    mp.set_start_method("spawn", force=True)
    queue: mp.SimpleQueue = mp.SimpleQueue()

    print(f"Spawning {world_size} worker processes …")
    mp.spawn(
        fn=worker,
        args=(world_size, args, queue),
        nprocs=world_size,
        join=True,
    )

    # ---- Aggregate results --------------------------------------------
    total_correct = total_seen = 0
    while not queue.empty():
        c, t = queue.get()
        total_correct += c
        total_seen += t

    acc = total_correct / total_seen if total_seen else 0.0
    print(
        f"[ALL GPU] GSM8K {args.split} 4-shot numeric match accuracy: {acc:.2%} "
        f"({total_correct}/{total_seen})"
    )


if __name__ == "__main__":
    main()