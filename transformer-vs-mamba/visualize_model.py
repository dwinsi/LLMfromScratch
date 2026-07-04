"""
visualize_model.py
------------------
Visualize the TinyGPT architectures (Transformer and Mamba) several ways.
Pick whichever fits what you want to see:

  1. torchinfo   — a text summary table (params + shapes per layer). Fastest.
  2. torchview   — an architecture diagram (module tree + tensor shapes) as PNG/SVG.
  3. Netron      — export to ONNX, then open interactively in the Netron app/website.

Usage:
    python visualize_model.py --arch mamba --tool all
    python visualize_model.py --arch transformer --tool torchview
    python visualize_model.py --arch mamba --tool netron   # writes mamba.onnx

Install what you need (see requirements-viz.txt):
    pip install torchinfo torchview netron
    # torchview also needs graphviz:  brew install graphviz   (on macOS)
"""

import argparse
import torch

from model import LanguageModel
from train import pick_device


# a small, visualization-friendly config (big models make huge diagrams)
VIZ = dict(vocab_size=65, d_model=64, n_layers=2, n_heads=4,
           d_state=16, context_len=64)
SEQ_LEN = 32
BATCH = 1


def build(arch, device="cpu"):
    model = LanguageModel(arch=arch, **VIZ).to(device)
    model.eval()
    return model


# ------------------------------------------------------------------
#  1. torchinfo — text summary
# ------------------------------------------------------------------
def show_torchinfo(arch):
    try:
        from torchinfo import summary
    except ImportError:
        print("  torchinfo not installed. Run: pip install torchinfo")
        return
    model = build(arch)
    print(f"\n{'='*70}\n  torchinfo summary — {arch}\n{'='*70}")
    # dtypes=[torch.long] because our input is token IDs, not floats
    summary(model, input_size=(BATCH, SEQ_LEN), dtypes=[torch.long],
            depth=4, col_names=("input_size", "output_size", "num_params"))


# ------------------------------------------------------------------
#  2. torchview — architecture diagram
# ------------------------------------------------------------------
def show_torchview(arch):
    try:
        from torchview import draw_graph
    except ImportError:
        print("  torchview not installed. Run: pip install torchview")
        print("  (also needs graphviz:  brew install graphviz)")
        return
    model = build(arch, device="meta")   # 'meta' = no memory used for viz
    dummy = torch.zeros(BATCH, SEQ_LEN, dtype=torch.long)
    g = draw_graph(
        model, input_data=dummy,
        graph_name=f"TinyGPT-{arch}",
        depth=3,                      # how deep to expand modules
        expand_nested=True,
        save_graph=True,
        filename=f"arch_{arch}",
        directory=".",
    )
    print(f"  saved arch_{arch}.png  (open it to see the module diagram)")
    return g


# ------------------------------------------------------------------
#  3. Netron — interactive ONNX export
# ------------------------------------------------------------------
def show_netron(arch):
    model = build(arch)
    dummy = torch.zeros(BATCH, SEQ_LEN, dtype=torch.long)
    onnx_path = f"{arch}.onnx"
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["token_ids"], output_names=["logits"],
        dynamic_axes={"token_ids": {0: "batch", 1: "seq"},
                      "logits": {0: "batch", 1: "seq"}},
        opset_version=17,
    )
    print(f"  saved {onnx_path}")
    print(f"  To explore interactively:")
    print(f"    option A:  pip install netron && netron {onnx_path}")
    print(f"    option B:  open https://netron.app and drag in {onnx_path}")
    # optionally launch the local server if netron is installed
    try:
        import netron
        print(f"  launching Netron server (Ctrl-C to stop)...")
        netron.start(onnx_path)
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mamba", "transformer"], default="mamba")
    ap.add_argument("--tool", choices=["torchinfo", "torchview", "netron", "all"],
                    default="all")
    args = ap.parse_args()

    if args.tool in ("torchinfo", "all"):
        show_torchinfo(args.arch)
    if args.tool in ("torchview", "all"):
        show_torchview(args.arch)
    if args.tool in ("netron", "all"):
        show_netron(args.arch)


if __name__ == "__main__":
    main()
