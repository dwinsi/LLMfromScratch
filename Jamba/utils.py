"""
utils.py
--------
Shared helpers used by train.py, demo.py, and the other scripts:
device selection, data download/loading, batching, and the LR schedule.

Kept separate so every script imports the same, tested versions.
"""

import os
import math
import urllib.request

import torch


SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")


def pick_device():
    """Apple MPS on Apple Silicon, CUDA on NVIDIA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # if torch.cuda.is_available():
    #     return torch.device("cuda")
    return torch.device("cpu")


def get_data(path=None, script_dir=None):
    """
    Load training text. If no path is given, download tiny-shakespeare into
    a `data/` folder NEXT TO THE SCRIPTS (not the current working directory),
    so it works no matter where you run from.
    """
    if path is None:
        base = script_dir or os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base, "data")
        path = os.path.join(data_dir, "shakespeare.txt")
        if not os.path.exists(path):
            os.makedirs(data_dir, exist_ok=True)
            print("[data] downloading tiny-shakespeare (~1MB) ...")
            urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def get_batch(data, batch_size, context_len, device):
    ix = torch.randint(0, len(data) - context_len - 1, (batch_size,))
    x = torch.stack([data[i:i + context_len] for i in ix])
    y = torch.stack([data[i + 1:i + context_len + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, batch_size, context_len, eval_iters, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, context_len, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(step, warmup, max_steps, lr, min_lr):
    """Linear warmup then cosine decay."""
    if step < warmup:
        return lr * step / warmup
    progress = (step - warmup) / max(max_steps - warmup, 1)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)
