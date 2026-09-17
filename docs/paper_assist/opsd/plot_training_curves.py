#!/usr/bin/env python3
"""Plot OPSD 100-step training curves from trainer_state.json and train logs."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home1/cjl/ICLR2026")
OUT = ROOT / "analysis" / "training_curves"
N_GEN_PER_STEP = 32  # 4 processes * 8 gradient accumulation steps

RUNS = {
    "Qwen3-1.7B": {
        "state": ROOT / "outputs/qwen31b_a6000_100steps_retry2/checkpoint-100/trainer_state.json",
        "log": ROOT / "opsd_100steps_retry2.log",
        "run_id": "qwen31b_a6000_100steps_retry2",
    },
    "Qwen3-4B": {
        "state": ROOT / "outputs/qwen34b_a6000_100steps/checkpoint-100/trainer_state.json",
        "log": ROOT / "qwen34b_100steps.log",
        "run_id": "qwen34b_a6000_100steps",
    },
}


def parse_generation(log_path: Path):
    text = log_path.read_text(errors="ignore")
    gens = re.findall(
        r"vLLM generation done - elapsed time: ([0-9.]+)s, prompts: (\d+), "
        r"total tokens: (\d+), avg length: ([0-9.]+), speed: ([0-9.]+) tok/s",
        text,
    )
    step_gen = []
    usable = len(gens) // N_GEN_PER_STEP * N_GEN_PER_STEP
    for i in range(0, usable, N_GEN_PER_STEP):
        chunk = gens[i : i + N_GEN_PER_STEP]
        speeds = [float(x[4]) for x in chunk]
        lengths = [float(x[3]) for x in chunk]
        toks = [int(x[2]) for x in chunk]
        elapsed = [float(x[0]) for x in chunk]
        step_gen.append(
            {
                "step": i // N_GEN_PER_STEP + 1,
                "speed_tok_s_mean": float(np.mean(speeds)),
                "avg_completion_len_mean": float(np.mean(lengths)),
                "total_tokens_mean": float(np.mean(toks)),
                "elapsed_s_mean": float(np.mean(elapsed)),
            }
        )
    return step_gen


def series(hist, key):
    return [x["step"] for x in hist], [x[key] for x in hist]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, meta in RUNS.items():
        hist = json.loads(meta["state"].read_text())["log_history"]
        rec = {
            "run": name,
            "output_dir": str(ROOT / "outputs" / meta["run_id"]),
            "trainer_state": str(meta["state"]),
            "train_log": str(meta["log"]),
            "log_history": hist,
            "generation_per_step": parse_generation(meta["log"]),
        }
        records[name] = rec
        (OUT / f"{meta['run_id']}_metrics.json").write_text(json.dumps(rec, indent=2))
    (OUT / "opsd_training_metrics.json").write_text(json.dumps(records, indent=2))

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    ax = axes[0, 0]
    for name, rec in records.items():
        st, y = series(rec["log_history"], "loss")
        ax.plot(st, y, marker="o", markersize=3, linewidth=1.4, label=name)
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set_title("OPSD train loss vs step (logging_steps=2)")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Loss (clipped JSD / KL surrogate)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for name, rec in records.items():
        st, y = series(rec["log_history"], "grad_norm")
        ax.plot(st, y, marker="o", markersize=3, linewidth=1.4, label=name)
    ax.set_title("Gradient norm vs step")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("grad_norm")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for name, rec in records.items():
        st = [x["step"] for x in rec["generation_per_step"]]
        y = [x["avg_completion_len_mean"] for x in rec["generation_per_step"]]
        ax.plot(st, y, linewidth=1.3, label=name)
    ax.axhline(1024, color="0.5", linewidth=0.8, linestyle="--", label="max_completion_length=1024")
    ax.set_title("Mean student completion length per step")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Tokens")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for name, rec in records.items():
        st = [x["step"] for x in rec["generation_per_step"]]
        y = [x["speed_tok_s_mean"] for x in rec["generation_per_step"]]
        ax.plot(st, y, linewidth=1.3, label=name)
    ax.set_title("Mean vLLM generation speed per step")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Tokens / second")
    ax.legend(frameon=False)

    fig.suptitle("OPSD 100-step training on 4x A6000 (GPU 1-4)", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "opsd_training_overview.png", bbox_inches="tight")
    fig.savefig(OUT / "opsd_training_overview.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for name, rec in records.items():
        st, y = series(rec["log_history"], "loss")
        ax.plot(st, y, marker="o", markersize=3.5, linewidth=1.6, label=f"{name} loss")
    ax.axhline(0, color="0.45", linewidth=0.9)
    ax.set_title("Train loss (on-policy clipped KL/JSD)")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "opsd_loss_curve.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for name, rec in records.items():
        st, y = series(rec["log_history"], "learning_rate")
        ax.plot(st, y, linewidth=1.6, label=name)
    ax.set_title("Learning rate schedule")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Learning rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "opsd_lr_curve.png", bbox_inches="tight")
    plt.close(fig)

    print("WROTE", OUT)
    for p in sorted(OUT.iterdir()):
        print(p.name, p.stat().st_size)
    for name, rec in records.items():
        hist = rec["log_history"]
        print(
            name,
            "n_log",
            len(hist),
            "first_loss",
            hist[0]["loss"],
            "last_loss",
            hist[-1]["loss"],
            "n_gen_steps",
            len(rec["generation_per_step"]),
        )


if __name__ == "__main__":
    main()
