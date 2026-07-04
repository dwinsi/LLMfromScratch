"""
visualize_model.py
------------------
Visualize the Jamba architectures several ways:

  1. torchinfo   — text summary table (params + shapes per layer). Fastest.
  2. torchview   — architecture diagram (module tree) as PNG.
  3. Netron      — export to ONNX, open interactively (zoom/click each layer).
  4. MoE routing — a heatmap of how tokens spread across experts.

Usage:
    python visualize_model.py --arch jamba --tool all
    python visualize_model.py --arch jamba --tool moe        # the interesting one
    python visualize_model.py --arch hybrid --tool torchview

Install what you need:
    pip install -r requirements-viz.txt
    # torchview also needs the graphviz binary:  brew install graphviz  (macOS)
"""

import _bootstrap  # noqa: F401

import os
import argparse

import torch
from model import JambaLM, MoE
from utils import pick_device

HERE = os.path.dirname(os.path.abspath(__file__))

# small, visualization-friendly config (big models make unreadable diagrams)
VIZ = dict(vocab_size=65, d_model=64, n_layers=4, n_heads=4,
           d_state=16, context_len=64, n_experts=8, top_k=2,
           attn_every=2, moe_every=2)
SEQ_LEN = 32
BATCH = 1


def build(arch, device="cpu"):
    model = JambaLM(arch=arch, **VIZ).to(device)
    model.eval()
    return model


def show_torchinfo(arch):
    try:
        from torchinfo import summary
    except ImportError:
        print("  torchinfo not installed. Run: pip install torchinfo")
        return
    model = build(arch)
    print(f"\n{'='*70}\n  torchinfo summary — {arch}\n{'='*70}")
    print(f"  layout: {model.layer_summary()}\n")
    summary(model, input_size=(BATCH, SEQ_LEN), dtypes=[torch.long],
            depth=4, col_names=("input_size", "output_size", "num_params"))


def show_torchview(arch):
    try:
        from torchview import draw_graph
    except ImportError:
        print("  torchview not installed. Run: pip install torchview")
        print("  (also needs graphviz:  brew install graphviz)")
        return
    model = build(arch, device="meta")
    dummy = torch.zeros(BATCH, SEQ_LEN, dtype=torch.long)
    draw_graph(model, input_data=dummy, graph_name=f"jamba-{arch}",
               depth=3, expand_nested=True, save_graph=True,
               filename=f"arch_{arch}", directory=HERE)
    print(f"  saved arch_{arch}.png")


def show_netron(arch):
    model = build(arch)
    dummy = torch.zeros(BATCH, SEQ_LEN, dtype=torch.long)
    onnx_path = os.path.join(HERE, f"{arch}.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["token_ids"], output_names=["logits"],
        dynamic_axes={"token_ids": {0: "batch", 1: "seq"},
                      "logits": {0: "batch", 1: "seq"}},
        opset_version=17)
    print(f"  saved {onnx_path}")
    print(f"  explore:  pip install netron && netron {onnx_path}")
    print(f"       or:  open https://netron.app and drag in the .onnx file")
    try:
        import netron
        netron.start(onnx_path)
    except ImportError:
        pass


def show_moe_routing(arch):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed. Run: pip install matplotlib")
        return

    model = build(arch)
    moe_layers = [(i, b.ffn) for i, b in enumerate(model.blocks)
                  if isinstance(b.ffn, MoE)]
    if not moe_layers:
        print(f"  arch '{arch}' has no MoE layers — try --arch jamba")
        return

    torch.manual_seed(0)
    x = torch.randint(0, VIZ["vocab_size"], (1, 128))
    E, k = VIZ["n_experts"], VIZ["top_k"]
    counts_per_layer = {}

    def make_hook(layer_idx, moe):
        def hook(module, inp, out):
            flat = inp[0].reshape(-1, inp[0].shape[-1])
            probs = torch.softmax(moe.router(flat), -1)
            _, idx = probs.topk(k, dim=-1)
            counts_per_layer[layer_idx] = torch.bincount(
                idx.reshape(-1), minlength=E).detach().float()
        return hook

    handles = [moe.register_forward_hook(make_hook(i, moe)) for i, moe in moe_layers]
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    layers = sorted(counts_per_layer)
    mat = torch.stack([counts_per_layer[l] for l in layers]).numpy()

    fig, ax = plt.subplots(figsize=(1.2 * E, 0.8 * len(layers) + 1.5))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(E)); ax.set_xticklabels([f"E{e}" for e in range(E)])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"layer {l}" for l in layers])
    ax.set_xlabel("expert"); ax.set_ylabel("MoE layer")
    ax.set_title(f"MoE token routing — {arch}\n(128 tokens x top-{k}, "
                 f"ideal = {128*k//E} per expert)")
    for i in range(len(layers)):
        for j in range(E):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] < mat.max() * 0.6 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, label="tokens routed")
    fig.tight_layout()
    out = os.path.join(HERE, f"moe_routing_{arch}.png")
    fig.savefig(out, dpi=130)
    print(f"  saved {out}")
    print(f"  (untrained router -> roughly uniform; after training the")
    print(f"   load-balancing loss keeps it from collapsing onto few experts)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mamba", "transformer", "hybrid", "jamba"],
                    default="jamba")
    ap.add_argument("--tool",
                    choices=["torchinfo", "torchview", "netron", "moe", "all"],
                    default="all")
    args = ap.parse_args()

    if args.tool in ("torchinfo", "all"):
        show_torchinfo(args.arch)
    if args.tool in ("torchview", "all"):
        show_torchview(args.arch)
    if args.tool in ("netron", "all"):
        show_netron(args.arch)
    if args.tool in ("moe", "all"):
        show_moe_routing(args.arch)


if __name__ == "__main__":
    main()
