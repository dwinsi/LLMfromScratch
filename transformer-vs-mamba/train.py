"""
train.py
--------
Train TinyGPT on a MacBook Air M2 (or any machine). Automatically uses the
Apple MPS GPU backend when available, otherwise CPU.

    python train.py                    # trains a Mamba model on Shakespeare
    python train.py --arch transformer # trains a Transformer instead
    python train.py --data mytext.txt  # use your own text

Both architectures share the same training loop, so comparisons are fair.
"""

import os
import time
import math
import argparse
import urllib.request

import torch
import torch.nn as nn

from model import LanguageModel, CharTokenizer


# ─────────────────────────────────────────────
#  Config tuned for an M2 MacBook Air (8GB unified memory)
#  These sizes train in a few minutes and fit comfortably.
# ─────────────────────────────────────────────
CONFIG = dict(
    d_model=128,
    n_layers=4,
    n_heads=4,          # transformer only
    d_state=16,         # mamba only
    context_len=256,
    dropout=0.1,

    batch_size=32,      # lower to 16 if you hit memory pressure
    max_steps=3000,
    eval_every=250,
    eval_iters=50,
    lr=3e-3,
    min_lr=1e-4,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup=100,

    train_split=0.9,
)

SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")


def pick_device():
    """MPS on Apple Silicon, CUDA on NVIDIA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_data(path):
    if path is None:
        path = "data/shakespeare.txt"
        if not os.path.exists(path):
            os.makedirs("data", exist_ok=True)
            print("[data] downloading tiny-shakespeare (~1MB) ...")
            urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def get_batch(data, cfg, device):
    L = cfg["context_len"]
    ix = torch.randint(0, len(data) - L - 1, (cfg["batch_size"],))
    x = torch.stack([data[i:i + L] for i in ix])
    y = torch.stack([data[i + 1:i + L + 1] for i in ix])
    # non-blocking transfer helps a little on MPS
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, splits, cfg, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            x, y = get_batch(data, cfg, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(step, cfg):
    """Linear warmup then cosine decay."""
    if step < cfg["warmup"]:
        return cfg["lr"] * step / cfg["warmup"]
    progress = (step - cfg["warmup"]) / (cfg["max_steps"] - cfg["warmup"])
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg["min_lr"] + coeff * (cfg["lr"] - cfg["min_lr"])


def train(arch, data_path):
    device = pick_device()
    print(f"[device] {device}")
    if device.type == "mps":
        print("[device] using Apple MPS GPU backend")

    # Optional TensorBoard logging — safe if tensorboard isn't installed.
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f"runs/{arch}")
        print(f"[tensorboard] logging to runs/{arch}  "
              f"(view with:  tensorboard --logdir runs)")
    except ImportError:
        print("[tensorboard] not installed — skipping (pip install tensorboard)")

    text = get_data(data_path)
    tok = CharTokenizer(text)
    print(f"[data] {len(text):,} chars, vocab {tok.vocab_size}")

    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(ids) * CONFIG["train_split"])
    splits = {"train": ids[:n], "val": ids[n:]}

    model = LanguageModel(
        vocab_size=tok.vocab_size, arch=arch,
        d_model=CONFIG["d_model"], n_layers=CONFIG["n_layers"],
        n_heads=CONFIG["n_heads"], d_state=CONFIG["d_state"],
        context_len=CONFIG["context_len"], dropout=CONFIG["dropout"],
    ).to(device)
    print(f"[model] arch={arch}  params={model.count_params():,}")

    # log the compute graph to TensorBoard (viewable in the GRAPHS tab)
    if writer is not None:
        try:
            dummy = torch.zeros(1, CONFIG["context_len"], dtype=torch.long, device=device)
            writer.add_graph(model, dummy)
        except Exception as e:
            print(f"[tensorboard] graph logging skipped: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                            betas=(0.9, 0.95), weight_decay=CONFIG["weight_decay"])

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{arch}.pt"
    best_val = float("inf")
    t0 = time.time()

    print("─" * 64)
    for step in range(1, CONFIG["max_steps"] + 1):
        lr = lr_at(step, CONFIG)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(splits["train"], CONFIG, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        opt.step()

        if step % CONFIG["eval_every"] == 0 or step == 1:
            m = estimate_loss(model, splits, CONFIG, device)
            dt = time.time() - t0
            print(f"  step {step:5d}/{CONFIG['max_steps']} | "
                  f"train {m['train']:.4f} | val {m['val']:.4f} | "
                  f"lr {lr:.1e} | {dt:.0f}s")

            # log to TensorBoard: loss curves, learning rate, weight histograms
            if writer is not None:
                writer.add_scalar("loss/train", m["train"], step)
                writer.add_scalar("loss/val", m["val"], step)
                writer.add_scalar("lr", lr, step)
                for name, p in model.named_parameters():
                    if p.requires_grad and p.numel() > 1:
                        writer.add_histogram(f"weights/{name}", p, step)

            if m["val"] < best_val:
                best_val = m["val"]
                torch.save({"model": model.state_dict(), "config": CONFIG,
                            "arch": arch, "vocab": (tok.stoi, tok.itos)}, ckpt_path)

    dt = time.time() - t0
    print("─" * 64)
    print(f"[done] {arch}: best val {best_val:.4f} (ppl {math.exp(best_val):.1f}) "
          f"in {dt/60:.1f} min -> {ckpt_path}")

    if writer is not None:
        writer.close()

    # quick sample
    print(f"\n[sample] {arch} generating...\n")
    seed = torch.tensor([tok.encode("ROMEO:")], dtype=torch.long, device=device)
    out = model.generate(seed, max_new=300, temperature=0.8, top_k=40)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["mamba", "transformer"], default="mamba")
    p.add_argument("--data", default=None)
    args = p.parse_args()
    train(args.arch, args.data)
