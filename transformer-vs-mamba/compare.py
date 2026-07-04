"""
compare.py
----------
Train BOTH a Transformer and a Mamba model on the same data with identical
settings, then plot their loss curves and report speed / parameter counts.

This is the payoff: a fair, apples-to-apples comparison on your own machine.

    python compare.py                 # shorter run, good for a quick look
    python compare.py --steps 3000    # longer run for better models

Outputs comparison.png and prints a summary table.
"""

import os
import time
import math
import argparse

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import LanguageModel, CharTokenizer
from train import pick_device, get_data, get_batch, estimate_loss, lr_at, CONFIG


def train_one(arch, splits, tok, cfg, device):
    """Train a single architecture, returning its loss history and timing."""
    model = LanguageModel(
        vocab_size=tok.vocab_size, arch=arch,
        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_state=cfg["d_state"],
        context_len=cfg["context_len"], dropout=cfg["dropout"],
    ).to(device)
    n_params = model.count_params()
    print(f"\n[{arch}] params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            betas=(0.9, 0.95), weight_decay=cfg["weight_decay"])

    history = {"step": [], "train": [], "val": []}
    t0 = time.time()
    for step in range(1, cfg["max_steps"] + 1):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(splits["train"], cfg, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()

        if step % cfg["eval_every"] == 0 or step == 1:
            m = estimate_loss(model, splits, cfg, device)
            history["step"].append(step)
            history["train"].append(m["train"])
            history["val"].append(m["val"])
            print(f"  [{arch}] step {step:5d} | train {m['train']:.4f} | "
                  f"val {m['val']:.4f} | {time.time()-t0:.0f}s")
    elapsed = time.time() - t0
    return dict(arch=arch, params=n_params, history=history,
                seconds=elapsed, best_val=min(history["val"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    cfg = dict(CONFIG)
    cfg["max_steps"] = args.steps

    device = pick_device()
    print(f"[device] {device}")

    text = get_data(args.data)
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(ids) * cfg["train_split"])
    splits = {"train": ids[:n], "val": ids[n:]}
    print(f"[data] {len(text):,} chars, vocab {tok.vocab_size}")

    results = []
    for arch in ["transformer", "mamba"]:
        results.append(train_one(arch, splits, tok, cfg, device))

    # ---- plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"transformer": "#7F77DD", "mamba": "#1D9E75"}
    for r in results:
        ax1.plot(r["history"]["step"], r["history"]["val"],
                 color=colors[r["arch"]], linewidth=2,
                 label=f"{r['arch']} (val)")
        ax1.plot(r["history"]["step"], r["history"]["train"],
                 color=colors[r["arch"]], linewidth=1, linestyle="--", alpha=0.5,
                 label=f"{r['arch']} (train)")
    ax1.set_xlabel("step"); ax1.set_ylabel("loss")
    ax1.set_title("Loss curves — Transformer vs Mamba")
    ax1.legend(); ax1.grid(alpha=0.15)

    # bar chart: speed and final loss
    archs = [r["arch"] for r in results]
    secs = [r["seconds"] for r in results]
    vals = [r["best_val"] for r in results]
    x = range(len(archs))
    ax2b = ax2.twinx()
    bars = ax2.bar([i - 0.2 for i in x], secs, width=0.4,
                   color="#BA7517", label="train time (s)")
    bars2 = ax2b.bar([i + 0.2 for i in x], vals, width=0.4,
                     color="#D85A30", label="best val loss")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(archs)
    ax2.set_ylabel("train time (s)", color="#BA7517")
    ax2b.set_ylabel("best val loss", color="#D85A30")
    ax2.set_title("Training time & final loss")
    fig.tight_layout()
    fig.savefig("comparison.png", dpi=130)
    print("\n[saved] comparison.png")

    # ---- summary table ----
    print("\n" + "=" * 60)
    print(f"  {'arch':<14}{'params':>12}{'time(s)':>10}{'best val':>12}")
    print("  " + "-" * 46)
    for r in results:
        print(f"  {r['arch']:<14}{r['params']:>12,}{r['seconds']:>10.0f}"
              f"{r['best_val']:>12.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
