#!/usr/bin/env python3
"""AntiSD / 基座模型：数学推理最小可运行脚本。

AntiSD 本身是训练期算法（在 GRPO 优势上叠加 JSD 反蒸馏 token 信号），
推理时就是普通 Causal LM。本脚本覆盖两种用法：

1. 单题推理：加载 HF / 本地 checkpoint，打印思维链和 \\boxed{} 答案
2. parquet 评测：读 data/prepare_antisd.sh 产出的 test.parquet，批量生成并可选核分

输入格式
--------
- 单题: --prompt "题目文本"
- 文件: --prompt-file problem.txt
- 评测: --eval-parquet datasets/math/aime25/test.parquet
        每行需含 prompt (chat list) 与 reward_model.ground_truth

预期输出
--------
- 完整生成文本
- 从最后一个 \\boxed{...} 抽出的答案
- 若提供 ground_truth 且安装了 math-verify: correct=True/False

示例
----
    python inference_math.py --model Qwen/Qwen3-8B --prompt "1+1=?"
    python inference_math.py --model /path/to/hf_ckpt --eval-parquet datasets/math/aime25/test.parquet --n 4
    python inference_math.py --backend vllm --model Qwen/Qwen3-8B --eval-parquet datasets/math/aime25/test.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional


INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."


def extract_last_boxed(text: str) -> Optional[str]:
    """与 verl/utils/reward_score/math_feedback/__init__.py 相同的抽取逻辑。"""
    if not text:
        return None
    last_start = text.rfind(r"\boxed{")
    if last_start == -1:
        return None
    depth = 0
    i = last_start + len(r"\boxed{")
    content_start = i
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[content_start:i]
            depth -= 1
        i += 1
    return None


def verify_math(pred: Optional[str], gold: str) -> Optional[bool]:
    if pred is None or not gold:
        return None
    try:
        from math_verify import parse as mv_parse
        from math_verify import verify as mv_verify
    except ImportError:
        return None
    try:
        gold_parsed = mv_parse("\\boxed{" + gold + "}", parsing_timeout=30)
        pred_parsed = mv_parse("\\boxed{" + pred + "}", parsing_timeout=30)
        if gold_parsed and pred_parsed:
            return bool(mv_verify(gold_parsed, pred_parsed, timeout_seconds=30))
    except Exception:
        return None
    return False


def ensure_instruction(text: str) -> str:
    if r"\boxed{}" in text or "boxed{}" in text:
        return text
    return text.rstrip() + " " + INSTRUCTION


def load_parquet_rows(path: str, max_samples: int) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("评测 parquet 需要 pandas / pyarrow") from exc
    df = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        prompt = rec.get("prompt")
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": ensure_instruction(prompt)}]
        else:
            messages = list(prompt)
            if messages and isinstance(messages[-1], dict):
                messages[-1] = dict(messages[-1])
                messages[-1]["content"] = ensure_instruction(str(messages[-1].get("content", "")))
        rm = rec.get("reward_model") or {}
        if hasattr(rm, "item"):
            rm = rm.item()
        gold = ""
        if isinstance(rm, dict):
            gold = str(rm.get("ground_truth", "") or "")
        rows.append({"messages": messages, "gold": gold, "raw": rec})
        if max_samples > 0 and len(rows) >= max_samples:
            break
    return rows


class HFBackend:
    def __init__(self, model_path: str, dtype: str, max_new_tokens: int,
                 temperature: float, top_p: float, enable_thinking: bool):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "auto": "auto"}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, messages: list[dict[str, str]]) -> str:
        import torch

        kwargs = {}
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


class VLLMBackend:
    def __init__(self, model_path: str, dtype: str, max_new_tokens: int,
                 temperature: float, top_p: float, enable_thinking: bool,
                 tensor_parallel: int):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(
            model=model_path,
            dtype=dtype if dtype != "auto" else "bfloat16",
            tensor_parallel_size=tensor_parallel,
            trust_remote_code=True,
        )
        self.params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def generate(self, messages: list[dict[str, str]]) -> str:
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        outs = self.llm.generate([prompt], self.params)
        return outs[0].outputs[0].text


def build_backend(args):
    common = dict(
        model_path=args.model,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )
    if args.backend == "vllm":
        return VLLMBackend(**common, tensor_parallel=args.tensor_parallel)
    return HFBackend(**common)


def run_one(backend, messages: list[dict[str, str]], gold: str = "") -> dict[str, Any]:
    text = backend.generate(messages)
    pred = extract_last_boxed(text)
    correct = verify_math(pred, gold) if gold else None
    return {
        "prompt": messages[-1]["content"] if messages else "",
        "generation": text,
        "pred": pred,
        "gold": gold,
        "correct": correct,
    }


def parse_args():
    p = argparse.ArgumentParser(description="AntiSD 数学推理最小脚本")
    p.add_argument("--model", default="Qwen/Qwen3-8B",
                   help="HF Hub ID 或本地 actor/hf 目录")
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    p.add_argument("--prompt", default="", help="单题文本")
    p.add_argument("--prompt-file", default="", help="从文件读题目")
    p.add_argument("--eval-parquet", default="", help="verl 格式 test.parquet")
    p.add_argument("--max-samples", type=int, default=0, help="0=全部")
    p.add_argument("--n", type=int, default=1, help="每题采样次数 (pass@n 粗评)")
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.7,
                   help="论文评测温度 0.7；user.yaml 训练内评测默认 0.6")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "auto"])
    p.add_argument("--enable-thinking", action="store_true",
                   help="Qwen3-think / Olmo-Think 打开 think 模式")
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--output", default="", help="把 JSONL 结果写到该路径")
    p.add_argument("--quiet", action="store_true", help="只打印题号/答案，不打印全文")
    return p.parse_args()


def main():
    args = parse_args()
    jobs: list[tuple[list[dict[str, str]], str]] = []

    if args.eval_parquet:
        for row in load_parquet_rows(args.eval_parquet, args.max_samples):
            jobs.append((row["messages"], row["gold"]))
    else:
        text = args.prompt
        if args.prompt_file:
            text = Path(args.prompt_file).read_text(encoding="utf-8")
        if not text.strip():
            text = (
                "Find the number of integer pairs (a,b) with 1 <= a,b <= 100 "
                "such that gcd(a,b)+lcm(a,b)=a+b+50."
            )
            print("[info] 未给题目，使用论文 Appendix C 的示例题。", file=sys.stderr)
        jobs.append(([{"role": "user", "content": ensure_instruction(text)}], ""))

    backend = build_backend(args)
    out_f = open(args.output, "w", encoding="utf-8") if args.output else None
    n_correct = 0
    n_scored = 0

    try:
        for i, (messages, gold) in enumerate(jobs):
            best = None
            any_correct = False
            for k in range(args.n):
                rec = run_one(backend, messages, gold)
                rec["sample"] = k
                rec["index"] = i
                if rec["correct"] is True:
                    any_correct = True
                if best is None or rec["correct"] is True:
                    best = rec
                if out_f:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            assert best is not None
            print("=" * 72)
            print(f"[{i}] pred={best['pred']!r}  gold={gold!r}  correct={best['correct']}")
            if not args.quiet:
                print(best["generation"][:2000])
                if len(best["generation"]) > 2000:
                    print("...[truncated]...")
            if gold:
                n_scored += 1
                n_correct += int(any_correct)
        if n_scored:
            print("=" * 72)
            print(f"scored={n_scored}  pass@{args.n}={n_correct}/{n_scored}  "
                  f"acc={n_correct / n_scored:.3f}")
    finally:
        if out_f:
            out_f.close()


if __name__ == "__main__":
    main()
