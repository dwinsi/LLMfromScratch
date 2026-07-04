"""
model.py
--------
Two sequence models sharing one interface, so you can train and compare
them head-to-head on the same data:

    * TransformerLM  — standard GPT-style decoder (attention)
    * MambaLM        — selective state space model (Mamba block)

Both take token IDs [batch, seq_len] and return (logits, loss).
Optimized to run comfortably on a MacBook Air M2 via the MPS backend.

Nothing here is a black box: attention, the selective scan, and the
HiPPO-style A initialization are all written out explicitly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
#  SHARED: character tokenizer
# ══════════════════════════════════════════════════════════════════

class CharTokenizer:
    """Character-level tokenizer — simple and dependency-free."""
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


# ══════════════════════════════════════════════════════════════════
#  TRANSFORMER PATH
# ══════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # reshape into heads: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # scaled dot-product attention with a causal mask.
        # F.scaled_dot_product_attention is fused & fast on MPS.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ══════════════════════════════════════════════════════════════════
#  MAMBA PATH  (selective SSM)
# ══════════════════════════════════════════════════════════════════

def selective_scan(u, delta, A, B, C, D):
    """
    Sequential selective scan (clear and correct; MPS-friendly).
    u, delta: (b, d_inner, L)   A: (d_inner, N)
    B, C:     (b, N, L)         D: (d_inner,)
    Returns y: (b, d_inner, L)
    """
    b, d_inner, L = u.shape
    # discretize with input-dependent delta
    # deltaA:   (b, d_inner, L, N)
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2))
    B_bcast = B.permute(0, 2, 1).unsqueeze(1)             # (b, 1, L, N)
    deltaB_u = delta.unsqueeze(-1) * B_bcast * u.unsqueeze(-1)   # (b, d_inner, L, N)

    h = torch.zeros(b, d_inner, A.shape[1], device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = deltaA[:, :, t] * h + deltaB_u[:, :, t]
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, :, t]))
    y = torch.stack(ys, dim=2)
    return y + u * D.unsqueeze(0).unsqueeze(-1)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        # A initialized small-magnitude & negative (stable, persistent memory)
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, L, _ = x.shape
        residual = x
        x = self.norm(x)
        x_ssm, z = self.in_proj(x).chunk(2, dim=-1)

        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv1d(x_ssm)[:, :, :L]        # causal trim
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        x_dbl = self.x_proj(x_ssm)
        B, C, dt = torch.split(x_dbl, [self.d_state, self.d_state, 1], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)

        u = x_ssm.transpose(1, 2)
        delta = delta.transpose(1, 2)
        B = B.transpose(1, 2)
        C = C.transpose(1, 2)

        y = selective_scan(u, delta, A, B, C, self.D).transpose(1, 2)
        y = y * F.silu(z)
        return residual + self.drop(self.out_proj(y))


# ══════════════════════════════════════════════════════════════════
#  SHARED LANGUAGE MODEL WRAPPER
# ══════════════════════════════════════════════════════════════════

class LanguageModel(nn.Module):
    """
    A GPT-style language model that can use EITHER transformer or mamba blocks.
    Pick with arch="transformer" or arch="mamba".
    """
    def __init__(self, vocab_size, arch="mamba", d_model=128, n_layers=4,
                 n_heads=4, d_state=16, context_len=256, dropout=0.1):
        super().__init__()
        self.arch = arch
        self.context_len = context_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # transformer needs positional embeddings; mamba does not (recurrence is ordered)
        self.pos_emb = nn.Embedding(context_len, d_model) if arch == "transformer" else None
        self.drop = nn.Dropout(dropout)

        if arch == "transformer":
            self.blocks = nn.ModuleList(
                [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        elif arch == "mamba":
            self.blocks = nn.ModuleList(
                [MambaBlock(d_model, d_state=d_state, dropout=dropout) for _ in range(n_layers)])
        else:
            raise ValueError("arch must be 'transformer' or 'mamba'")

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight        # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new):
            idx_cond = idx[:, -self.context_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
