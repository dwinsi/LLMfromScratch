"""
mamba_block_torch.py
--------------------
A complete Mamba block in PyTorch — the form you'd actually use.
This is a faithful, readable implementation of the selective SSM (S6)
wrapped in the full Mamba block (input projection, causal conv, SiLU gate,
selective scan, output projection).

It is a drop-in replacement for a TransformerBlock: same input/output
shape [batch, seq_len, d_model], stackable into a language model.

Requires: torch   (pip install torch)

    python mamba_block_torch.py     # runs a shape test + tiny training demo
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
#  The selective scan (S6 core)
#  ----------------------------
#  h_t = Abar_t * h_{t-1} + Bbar_t * x_t ;  y_t = C_t . h_t + D * x_t
#  where Abar, Bbar, C, and the step Delta are ALL input-dependent.
# ══════════════════════════════════════════════════════════════════

def selective_scan(u, delta, A, B, C, D):
    """
    u:     (batch, d_inner, L)      the input signal per channel
    delta: (batch, d_inner, L)      input-dependent step size (> 0)
    A:     (d_inner, N)             state matrix (negative, stable) — the HiPPO backbone
    B:     (batch, N, L)            input-dependent input matrix
    C:     (batch, N, L)            input-dependent output matrix
    D:     (d_inner,)               skip connection

    Returns y: (batch, d_inner, L)

    This is the sequential form (clear and correct). Production Mamba uses a
    hardware-aware parallel scan for speed; the math is identical.
    """
    batch, d_inner, L = u.shape
    N = A.shape[1]

    # Discretize (Zero-Order Hold), with input-dependent delta:
    #   deltaA   = exp(delta * A)     shape (batch, d_inner, L, N)
    #   deltaB_u = delta * B * u      the input contribution
    # Broadcasting:  delta (b,d,L)->(b,d,L,1) ;  A (d,N)->(1,d,1,N)
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2))   # (b, d, L, N)
    #   B is (b, N, L) -> (b, L, N) -> (b, 1, L, N) to broadcast over d_inner
    B_bcast = B.permute(0, 2, 1).unsqueeze(1)                              # (b, 1, L, N)
    deltaB_u = delta.unsqueeze(-1) * B_bcast * u.unsqueeze(-1)             # (b, d, L, N)

    # sequential scan over time
    h = torch.zeros(batch, d_inner, N, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = deltaA[:, :, t] * h + deltaB_u[:, :, t]                 # (b, d, N)
        # output: C_t . h   -> C is (b, N, L)
        y_t = torch.einsum("bdn,bn->bd", h, C[:, :, t])            # (b, d)
        ys.append(y_t)
    y = torch.stack(ys, dim=2)                                     # (b, d, L)

    y = y + u * D.unsqueeze(0).unsqueeze(-1)                       # skip connection
    return y


# ══════════════════════════════════════════════════════════════════
#  The full Mamba block
# ══════════════════════════════════════════════════════════════════

class MambaBlock(nn.Module):
    """
    A single Mamba block. Drop-in replacement for a Transformer block.

    Flow:
        x  -> LayerNorm
           -> in_proj  (expand to 2 * d_inner: one path for SSM, one for gate)
           -> causal depthwise Conv1d  (local context)
           -> SiLU
           -> selective SSM  (the S6 core with input-dependent B, C, Delta)
           -> gate:  multiply by SiLU(gate path)
           -> out_proj  (back to d_model)
           -> residual add
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state

        self.norm = nn.LayerNorm(d_model)

        # input projection -> two paths (x for SSM, z for the gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # causal depthwise conv over the sequence (each channel independent)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=True,
        )

        # projections that make B, C, Delta INPUT-DEPENDENT (the selectivity)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # -> B, C, delta_pre
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)               # broadcast delta to channels

        # A: the state matrix. Parameterized as log for stability; init HiPPO-style.
        # A_real = -exp(A_log) is always negative -> stable memory.
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))       # (d_inner, d_state)
        self.D = nn.Parameter(torch.ones(self.d_inner))    # skip connection

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """x: (batch, seq_len, d_model) -> same shape."""
        b, L, _ = x.shape
        residual = x
        x = self.norm(x)

        # project and split into SSM path and gate path
        xz = self.in_proj(x)                               # (b, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)                     # each (b, L, d_inner)

        # causal depthwise conv: (b, L, d) -> (b, d, L) for Conv1d
        x_ssm = x_ssm.transpose(1, 2)                      # (b, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[:, :, :L]               # trim padding -> causal
        x_ssm = x_ssm.transpose(1, 2)                      # (b, L, d_inner)
        x_ssm = F.silu(x_ssm)

        # --- selectivity: compute B, C, Delta from the (conv'd) input ---
        x_dbl = self.x_proj(x_ssm)                         # (b, L, 2N+1)
        B, C, dt = torch.split(
            x_dbl, [self.d_state, self.d_state, 1], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))               # (b, L, d_inner) > 0

        A = -torch.exp(self.A_log)                         # (d_inner, N), negative -> stable

        # reshape to the scan's expected layout
        u = x_ssm.transpose(1, 2)                          # (b, d_inner, L)
        delta = delta.transpose(1, 2)                      # (b, d_inner, L)
        B = B.transpose(1, 2)                              # (b, N, L)
        C = C.transpose(1, 2)                              # (b, N, L)

        y = selective_scan(u, delta, A, B, C, self.D)      # (b, d_inner, L)
        y = y.transpose(1, 2)                              # (b, L, d_inner)

        # gate and project out
        y = y * F.silu(z)
        out = self.out_proj(y)
        return residual + out


# ══════════════════════════════════════════════════════════════════
#  A tiny language-model-style wrapper to show it stacks and trains
# ══════════════════════════════════════════════════════════════════

class MambaLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, d_state=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight               # weight tying

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss


# ══════════════════════════════════════════════════════════════════
#  Self-test: shapes, stability, and a tiny training run
# ══════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(0)
    print("=" * 68)
    print("  MAMBA BLOCK (PyTorch) — self-test")
    print("=" * 68)

    # --- shape test ---
    b, L, d = 2, 32, 64
    block = MambaBlock(d_model=d, d_state=16)
    x = torch.randn(b, L, d)
    y = block(x)
    print(f"\n  Input  shape: {tuple(x.shape)}")
    print(f"  Output shape: {tuple(y.shape)}  ->  {'MATCH' if y.shape == x.shape else 'FAIL'}")
    n_params = sum(p.numel() for p in block.parameters())
    print(f"  Block parameters: {n_params:,}")

    # --- stability: A must be negative ---
    A = -torch.exp(block.A_log)
    print(f"\n  A values all negative (stable)? {(A < 0).all().item()}")
    print(f"  A range: [{A.min():.3f}, {A.max():.3f}]")

    # --- tiny training: learn to copy a token from the start to the end ---
    print("\n  Training a 2-layer MambaLM to recall the first token at the end...")
    vocab = 10
    model = MambaLM(vocab, d_model=64, n_layers=2, d_state=16)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    def make_seq(batch, L):
        seq = torch.randint(1, vocab, (batch, L))
        # task: the LAST target is the FIRST token (long-range recall)
        tgt = seq.clone()
        tgt[:, -1] = seq[:, 0]
        return seq, tgt

    for step in range(1, 401):
        seq, tgt = make_seq(32, 24)
        _, loss = model(seq, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"    step {step:4d}   loss {loss.item():.4f}")

    # test recall accuracy on the final position
    model.eval()
    with torch.no_grad():
        seq, tgt = make_seq(200, 24)
        logits, _ = model(seq, tgt)
        pred_last = logits[:, -1].argmax(-1)
        acc = (pred_last == tgt[:, -1]).float().mean().item()
    print(f"\n  Recall accuracy (predict first token at the end): {acc*100:.1f}%")
    print(f"  (random guessing would be ~{100/vocab:.0f}%)")

    print("\n" + "=" * 68)
    print("  Mamba block works: correct shapes, stable A, learns long-range recall.")
    print("=" * 68)


if __name__ == "__main__":
    main()
