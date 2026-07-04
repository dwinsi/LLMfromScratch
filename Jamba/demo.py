"""
demo.py
-------
Show the four architectures side by side WITHOUT training — layer layouts,
total parameters, and active-params-per-token — then inspect how the MoE
router distributes tokens across experts.

    python demo.py

The fastest way to see what Jamba buys you: far more total parameters
(capacity) at only a small increase in active compute per token.
"""

import _bootstrap  # noqa: F401

import torch
from model import JambaLM, MoE


def architecture_table():
    vocab = 65
    print("=" * 78)
    print("  FOUR ARCHITECTURES — same width/depth, different block mix")
    print("=" * 78)
    print(f"  {'arch':<12}{'total params':>14}{'active/token':>14}{'ratio':>8}   layout")
    print("  " + "-" * 74)
    for arch in ["mamba", "transformer", "hybrid", "jamba"]:
        m = JambaLM(vocab_size=vocab, arch=arch, d_model=128, n_layers=8,
                    n_experts=8, top_k=2)
        total = m.count_params()
        active = m.active_params_per_token()
        print(f"  {arch:<12}{total:>14,}{active:>14,}{active/total:>7.0%}")
        print(f"  {'':12}{m.layer_summary()}")
    print("=" * 78)
    print("\n  Notice: 'jamba' has the MOST total parameters (MoE adds capacity)")
    print("  but its active-params-per-token stays close to the others, because")
    print("  each token only runs top-2 of 8 experts. That is the MoE bargain:")
    print("  more knowledge capacity, nearly the same compute per token.\n")


def inspect_moe_routing():
    print("=" * 78)
    print("  MoE ROUTING — how are tokens spread across experts?")
    print("=" * 78)
    torch.manual_seed(0)
    d_model, E, k = 128, 8, 2
    moe = MoE(d_model, n_experts=E, top_k=k)
    moe.eval()

    x = torch.randn(1, 200, d_model)  # 200 tokens
    with torch.no_grad():
        logits = moe.router(x.reshape(-1, d_model))
        probs = torch.softmax(logits, -1)
        _, idx = probs.topk(k, dim=-1)
    counts = torch.bincount(idx.reshape(-1), minlength=E)
    print(f"\n  200 tokens x top-{k} = {200*k} assignments across {E} experts")
    print(f"  ideal uniform: {200*k//E} per expert\n")
    for e in range(E):
        bar = "#" * int(counts[e].item() / 4)
        print(f"    expert {e}: {counts[e].item():4d}  {bar}")
    print("\n  (Untrained router -> roughly uniform. During training the")
    print("   load-balancing aux loss keeps it from collapsing onto few experts.)\n")


if __name__ == "__main__":
    architecture_table()
    inspect_moe_routing()
