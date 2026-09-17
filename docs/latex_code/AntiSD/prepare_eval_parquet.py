#!/usr/bin/env python3
"""把 /home1/cjl/ICLR2026/eval_datasets 转成 inference_math.py 用的 verl parquet。

优先用已经下好的本地文件（总共 <1MB），缺的再从 HuggingFace / ModelScope 补。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("EVAL_ROOT", "/home1/cjl/ICLR2026/eval_datasets"))
OUT = ROOT / "verl_parquet"
INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."


def ensure_instruction(text: str) -> str:
    text = str(text).strip()
    if r"\boxed{}" in text or "boxed{}" in text:
        return text
    return text + " " + INSTRUCTION


def rows_from(problems, answers, source: str, split: str = "test") -> list[dict]:
    out = []
    for i, (q, a) in enumerate(zip(problems, answers)):
        gold = a if not isinstance(a, float) else (str(int(a)) if a == int(a) else str(a))
        gold = str(gold).strip()
        out.append({
            "data_source": source,
            "prompt": [{"role": "user", "content": ensure_instruction(q)}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gold},
            "extra_info": {"split": split, "index": str(i)},
        })
    return out


def save(name: str, records: list[dict]) -> Path:
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "test.parquet"
    pd.DataFrame(records).to_parquet(path, index=False)
    print(f"[ok] {name:16s} n={len(records):4d}  {path}  {path.stat().st_size/1024:.1f}KB")
    return path


def try_download_hf(repo: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=str(dest),
            resume_download=True,
            max_workers=2,
        )
        print(f"[dl] HF {repo} -> {dest}")
        return True
    except Exception as e:
        print(f"[skip] HF {repo}: {e}")
        return False


def try_download_ms(repo: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        snapshot_download(repo, repo_type="dataset", local_dir=str(dest))
        print(f"[dl] MS {repo} -> {dest}")
        return True
    except Exception as e:
        print(f"[skip] MS {repo}: {e}")
        return False


def load_aime2024():
    p = ROOT / "HuggingFaceH4--aime_2024/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(p)
    return rows_from(df["problem"], df["answer"], "HuggingFaceH4/aime_2024")


def load_aime2025():
    p = ROOT / "yentinglin--aime_2025/data/train-00000-of-00001-243207c6c994e1bd.parquet"
    df = pd.read_parquet(p)
    return rows_from(df["problem"], df["answer"], "yentinglin/aime_2025")


def load_hmmt25():
    p = ROOT / "MathArena--hmmt_feb_2025/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(p)
    return rows_from(df["problem"], df["answer"], "MathArena/hmmt_feb_2025")


def load_minerva():
    p = ROOT / "math-ai--minervamath/test.jsonl"
    qs, ans = [], []
    with open(p) as f:
        for line in f:
            rec = json.loads(line)
            qs.append(rec["question"])
            ans.append(rec["answer"])
    return rows_from(qs, ans, "math-ai/minervamath")


def load_amc23():
    p = ROOT / "math-ai--amc23/test-00000-of-00001.parquet"
    df = pd.read_parquet(p)
    qcol = "question" if "question" in df.columns else "problem"
    return rows_from(df[qcol], df["answer"], "math-ai/amc23")


def load_math500():
    p = ROOT / "HuggingFaceH4--MATH-500/test.jsonl"
    qs, ans = [], []
    with open(p) as f:
        for line in f:
            rec = json.loads(line)
            qs.append(rec.get("problem") or rec.get("question"))
            ans.append(rec.get("answer"))
    return rows_from(qs, ans, "HuggingFaceH4/MATH-500")


def load_or_download_aime26():
    local_dir = ROOT / "math-ai--aime26"
    parquets = list(local_dir.rglob("*.parquet")) if local_dir.exists() else []
    if not parquets:
        try_download_hf("math-ai/aime26", local_dir) or try_download_ms("AI-ModelScope/aime26", local_dir)
        parquets = list(local_dir.rglob("*.parquet"))
    if not parquets:
        # 有些仓只有 json/jsonl
        jsonls = list(local_dir.rglob("*.jsonl")) + list(local_dir.rglob("*.json"))
        if jsonls:
            recs = []
            for jp in jsonls:
                if jp.name.startswith("dataset_infos"):
                    continue
                with open(jp) as f:
                    raw = f.read().strip()
                    if not raw:
                        continue
                    if raw[0] == "[":
                        data = json.loads(raw)
                    else:
                        data = [json.loads(x) for x in raw.splitlines() if x.strip()]
                for rec in data:
                    q = rec.get("problem") or rec.get("question")
                    a = rec.get("answer")
                    if q is not None and a is not None:
                        recs.append((q, a))
            if recs:
                qs, ans = zip(*recs)
                return rows_from(qs, ans, "math-ai/aime26")
        print("[warn] aime26 未找到可用文件")
        return []
    df = pd.read_parquet(parquets[0])
    qcol = "problem" if "problem" in df.columns else "question"
    return rows_from(df[qcol], df["answer"], "math-ai/aime26")


def load_or_download_official_aime25():
    """论文写的是 math-ai/aime25；本地已有 yentinglin 30 题同源集。再备一份官方。"""
    local_dir = ROOT / "math-ai--aime25"
    if local_dir.exists() and any(local_dir.rglob("*.parquet")):
        df = pd.read_parquet(next(local_dir.rglob("*.parquet")))
        qcol = "problem" if "problem" in df.columns else "question"
        return rows_from(df[qcol], df["answer"], "math-ai/aime25")
    ok = try_download_hf("math-ai/aime25", local_dir)
    if not ok:
        return []
    parquets = list(local_dir.rglob("*.parquet"))
    if not parquets:
        return []
    df = pd.read_parquet(parquets[0])
    qcol = "problem" if "problem" in df.columns else "question"
    return rows_from(df[qcol], df["answer"], "math-ai/aime25")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 先转已经在磁盘上的（无需联网）
    save("aime_2024", load_aime2024())
    save("aime25", load_aime2025())
    save("hmmt25", load_hmmt25())
    save("minervamath", load_minerva())
    save("amc23", load_amc23())
    save("math500", load_math500())
    # 再补论文里缺的官方仓，失败就跳过
    official = load_or_download_official_aime25()
    if official:
        save("aime25_official", official)
    aime26 = load_or_download_aime26()
    if aime26:
        save("aime26", aime26)
    print("\n[done] parquet root:", OUT)


if __name__ == "__main__":
    main()
