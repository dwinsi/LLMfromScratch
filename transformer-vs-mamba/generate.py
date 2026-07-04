"""
generate.py
-----------
Load a trained checkpoint and generate text interactively.

    python generate.py                      # loads checkpoints/mamba.pt
    python generate.py --ckpt checkpoints/transformer.pt
    python generate.py --seed "HAMLET:" --temp 0.7
"""

import argparse
import torch
from model import LanguageModel, CharTokenizer
from train import pick_device


def load(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg, arch = ck["config"], ck["arch"]
    stoi, itos = ck["vocab"]
    tok = CharTokenizer.__new__(CharTokenizer)
    tok.stoi, tok.itos, tok.vocab_size = stoi, itos, len(stoi)
    model = LanguageModel(
        vocab_size=tok.vocab_size, arch=arch,
        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_state=cfg["d_state"],
        context_len=cfg["context_len"], dropout=0.0)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[load] {arch} model, {model.count_params():,} params")
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/mamba.pt")
    ap.add_argument("--seed", default=None)
    ap.add_argument("--max_new", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    device = pick_device()
    model, tok = load(args.ckpt)
    model = model.to(device)

    def gen(seed):
        ids = torch.tensor([tok.encode(seed)], dtype=torch.long, device=device)
        out = model.generate(ids, args.max_new, args.temp, args.top_k)
        return tok.decode(out[0].tolist())

    if args.seed is not None:
        print(gen(args.seed))
        return

    print("\nInteractive generation. Type a seed, or 'quit'.\n")
    while True:
        try:
            s = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if s.lower() == "quit":
            break
        if not s:
            s = "\n"
        bad = [c for c in s if c not in tok.stoi]
        if bad:
            print(f"  unknown chars {set(bad)} — try others")
            continue
        print("─" * 50)
        print(gen(s))
        print("─" * 50)


if __name__ == "__main__":
    main()
