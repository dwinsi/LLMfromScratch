"""
smoke_test.py
-------------
Run this FIRST on your Mac to confirm everything works before a full
training run. It checks MPS availability, builds both models, runs one
forward+backward step each, and confirms output shapes. Takes ~10 seconds.

    python smoke_test.py
"""
import torch
from model import LanguageModel
from train import pick_device


def main():
    print("=" * 56)
    print("  mamba-vs-transformer smoke test")
    print("=" * 56)

    device = pick_device()
    print(f"\n  Device: {device}")
    if device.type == "mps":
        print("  ✓ Apple MPS GPU backend is available and will be used")
    elif device.type == "cpu":
        print("  ! No GPU backend — will run on CPU (slower but fine)")

    vocab, B, T = 65, 4, 64
    x = torch.randint(0, vocab, (B, T), device=device)
    y = torch.randint(0, vocab, (B, T), device=device)

    for arch in ["transformer", "mamba"]:
        model = LanguageModel(vocab_size=vocab, arch=arch,
                              d_model=128, n_layers=4, context_len=256).to(device)
        logits, loss = model(x, y)
        loss.backward()
        assert logits.shape == (B, T, vocab), f"bad shape {logits.shape}"
        print(f"\n  [{arch}]")
        print(f"    params      : {model.count_params():,}")
        print(f"    logits shape: {tuple(logits.shape)}  ✓")
        print(f"    loss        : {loss.item():.4f}")
        print(f"    backward    : ✓ gradients flow")

    print("\n" + "=" * 56)
    print("  All good! You're ready to run:  python compare.py")
    print("=" * 56)


if __name__ == "__main__":
    main()
