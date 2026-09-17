#!/usr/bin/env python3
"""用已有 vLLM 对 verl parquet 做 n=1 批量评测，避免逐条 HF generate。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."


def extract_last_boxed(text: str):
    if not text:
        return None
    last_start = text.rfind(r"\boxed{")
    if last_start == -1:
        return None
    depth = 0
    i = last_start + len(r"\boxed{")
    start = i
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
        i += 1
    return None


def verify_math(pred, gold):
    if pred is None or not gold:
        return None
    try:
        from math_verify import parse as mv_parse
        from math_verify import verify as mv_verify
        gold_parsed = mv_parse("\\boxed{" + str(gold) + "}", parsing_timeout=30)
        pred_parsed = mv_parse("\\boxed{" + pred + "}", parsing_timeout=30)
        if gold_parsed and pred_parsed:
            return bool(mv_verify(gold_parsed, pred_parsed, timeout_seconds=30))
    except Exception:
        return False
    return False


def ensure_instruction(text: str) -> str:
    text = str(text).strip()
    if r"\boxed{}" in text or "boxed{}" in text:
        return text
    return text + " " + INSTRUCTION


def load_jobs(parquet_path: str, max_samples: int):
    df = pd.read_parquet(parquet_path)
    jobs = []
    for rec in df.to_dict(orient="records"):
        prompt = rec.get("prompt")
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": ensure_instruction(prompt)}]
        else:
            messages = list(prompt)
            messages[-1] = dict(messages[-1])
            messages[-1]["content"] = ensure_instruction(str(messages[-1].get("content", "")))
        rm = rec.get("reward_model") or {}
        if hasattr(rm, "item"):
            rm = rm.item()
        gold = str(rm.get("ground_truth", "") or "") if isinstance(rm, dict) else ""
        jobs.append({"messages": messages, "gold": gold})
        if max_samples > 0 and len(jobs) >= max_samples:
            break
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--eval-parquet", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = p.parse_args()

    jobs = load_jobs(args.eval_parquet, args.max_samples)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = []
    for j in jobs:
        try:
            prompts.append(tok.apply_chat_template(
                j["messages"], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            ))
        except TypeError:
            prompts.append(tok.apply_chat_template(
                j["messages"], tokenize=False, add_generation_prompt=True,
            ))

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=min(args.max_new_tokens + 2048, 16384),
    )
    params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    outs = llm.generate(prompts, params)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    hit = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, (job, out) in enumerate(zip(jobs, outs)):
            text = out.outputs[0].text
            pred = extract_last_boxed(text)
            ok = verify_math(pred, job["gold"])
            if ok is True:
                hit += 1
            rec = {
                "index": i,
                "sample": 0,
                "pred": pred,
                "gold": job["gold"],
                "correct": ok,
                "generation": text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i}] pred={pred!r} gold={job['gold']!r} correct={ok}", flush=True)

    n = len(jobs)
    print(f"scored={n}  pass@1={hit}/{n}  acc={hit/n if n else 0:.3f}", flush=True)


if __name__ == "__main__":
    main()
