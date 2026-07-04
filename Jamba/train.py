"""
train.py
--------
Train the Jamba-style model on text. Auto-uses Apple MPS / CUDA / CPU.

    python train.py                 # trains the full jamba (mamba+attn+MoE)
    python train.py --arch mamba    # pure mamba
    python train.py --arch hybrid   # mamba + attention, dense FFN
    python train.py --data mytext.txt

Works from any directory (see _bootstrap). The MoE load-balancing aux loss
is added automatically inside the model.
"""

import _bootstrap  # noqa: F401  (fixes imports regardless of working dir)

import os
import time
import math
import argparse

import torch
import torch.nn as nn

from model import JambaLM, CharTokenizer, MoE
from utils import (pick_device, get_data, get_batch, estimate_loss, lr_at)

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIG = dict(
    d_model=128, n_layers=8, n_heads=4, d_state=16, context_len=256, dropout=0.1,
    attn_every=4, moe_every=2, n_experts=8, top_k=2,
    batch_size=32, max_steps=3000, eval_every=250, eval_iters=50,
    lr=3e-3, min_lr=1e-4, weight_decay=0.1, grad_clip=1.0, warmup=100,
    train_split=0.9,
)


def train(arch, data_path):
    device = pick_device()
    print(f"[device] {device}")

    # optional TensorBoard logging (safe if not installed)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(HERE, "runs", arch))
        print(f"[tensorboard] logging to runs/{arch}  (tensorboard --logdir runs)")
    except ImportError:
        print("[tensorboard] not installed — skipping (pip install tensorboard)")

    text = get_data(data_path, script_dir=HERE)
    tok = CharTokenizer(text)
    print(f"[data] {len(text):,} chars, vocab {tok.vocab_size}")
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(ids) * CONFIG["train_split"])
    splits = {"train": ids[:n], "val": ids[n:]}

    model = JambaLM(
        vocab_size=tok.vocab_size, arch=arch,
        d_model=CONFIG["d_model"], n_layers=CONFIG["n_layers"],
        n_heads=CONFIG["n_heads"], d_state=CONFIG["d_state"],
        context_len=CONFIG["context_len"], dropout=CONFIG["dropout"],
        attn_every=CONFIG["attn_every"], moe_every=CONFIG["moe_every"],
        n_experts=CONFIG["n_experts"], top_k=CONFIG["top_k"]).to(device)

    print(f"[model] arch={arch}")
    print(f"        layout       {model.layer_summary()}")
    print(f"        total params {model.count_params():,}")
    print(f"        active/token {model.active_params_per_token():,}  "
          f"(MoE runs only top-{CONFIG['top_k']} of {CONFIG['n_experts']})")

    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                            betas=(0.9, 0.95), weight_decay=CONFIG["weight_decay"])

    ckpt_dir = os.path.join(HERE, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, f"{arch}.pt")
    best = float("inf")
    t0 = time.time()

    print("-" * 64)
    for step in range(1, CONFIG["max_steps"] + 1):
        lr = lr_at(step, CONFIG["warmup"], CONFIG["max_steps"],
                   CONFIG["lr"], CONFIG["min_lr"])
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(splits["train"], CONFIG["batch_size"],
                         CONFIG["context_len"], device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        opt.step()

        if step % CONFIG["eval_every"] == 0 or step == 1:
            m = estimate_loss(model, splits, CONFIG["batch_size"],
                              CONFIG["context_len"], CONFIG["eval_iters"], device)
            print(f"  step {step:5d} | train {m['train']:.4f} | val {m['val']:.4f} "
                  f"| lr {lr:.1e} | {time.time()-t0:.0f}s")
            if writer is not None:
                writer.add_scalar("loss/train", m["train"], step)
                writer.add_scalar("loss/val", m["val"], step)
                writer.add_scalar("lr", lr, step)
                aux = sum(float(b.ffn.aux_loss) for b in model.blocks
                          if isinstance(b.ffn, MoE))
                writer.add_scalar("moe/aux_loss", aux, step)
            if m["val"] < best:
                best = m["val"]
                torch.save({"model": model.state_dict(), "config": CONFIG,
                            "arch": arch, "vocab": (tok.stoi, tok.itos)}, ckpt)

    print("-" * 64)
    print(f"[done] {arch}: best val {best:.4f} (ppl {math.exp(best):.1f}) "
          f"in {(time.time()-t0)/60:.1f} min -> {ckpt}")
    if writer is not None:
        writer.close()

    print("\n[sample]\n")
    seed = torch.tensor([tok.encode("ROMEO:")], dtype=torch.long, device=device)
    print(tok.decode(model.generate(seed, 300, 0.8, 40)[0].tolist()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["mamba", "transformer", "hybrid", "jamba"],
                   default="jamba")
    p.add_argument("--data", default=None)
    args = p.parse_args()
    train(args.arch, args.data)
