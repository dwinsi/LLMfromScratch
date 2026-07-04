"""
smoke_test.py
-------------
Run FIRST on your machine. Confirms the device works and all four
architectures do a forward + backward pass with correct shapes. ~15 seconds.

    python smoke_test.py
"""

import _bootstrap  # noqa: F401

import torch
from model import JambaLM
from utils import pick_device


def main():
    print("=" * 60)
    print("  jamba smoke test")
    print("=" * 60)
    device = pick_device()
    print(f"\n  Device: {device}")
    if device.type == "mps":
        print("  Apple MPS GPU backend available")
    elif device.type == "cpu":
        print("  Running on CPU (works, just slower)")

    vocab, B, T = 65, 4, 64
    x = torch.randint(0, vocab, (B, T), device=device)
    y = torch.randint(0, vocab, (B, T), device=device)

    for arch in ["mamba", "transformer", "hybrid", "jamba"]:
        model = JambaLM(vocab_size=vocab, arch=arch, d_model=128,
                        n_layers=8, context_len=256).to(device)
        logits, loss = model(x, y)
        loss.backward()
        assert logits.shape == (B, T, vocab), f"bad shape {logits.shape}"
        print(f"\n  [{arch}]")
        print(f"    total params : {model.count_params():,}")
        print(f"    active/token : {model.active_params_per_token():,}")
        print(f"    loss         : {loss.item():.4f}  (backward OK)")
        print(f"    layout       : {model.layer_summary()}")

    print("\n" + "=" * 60)
    print("  All good! Try:  python demo.py   then   python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
