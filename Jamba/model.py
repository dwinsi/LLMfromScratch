"""
model.py
--------
A Jamba-style language model combining the three ideas the modern LLM
frontier converged on:

    * Mamba blocks        — selective SSM, O(L) memory, fixed-size state
    * Attention blocks    — exact recall, O(L^2), inserted sparingly
    * Mixture-of-Experts  — many expert FFNs, only the top-k run per token

Build a pure version of any one, or a hybrid, or the full "jamba" mix.
Everything is written from scratch and heavily commented.

This file has no dependencies beyond torch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
#  Tokenizer
# ══════════════════════════════════════════════════════════════════

class CharTokenizer:
    """Character-level tokenizer — simple and dependency-free."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


# ══════════════════════════════════════════════════════════════════
#  Mixture-of-Experts feed-forward layer
#  --------------------------------------
#  Instead of one dense FFN, keep E independent expert FFNs. A small
#  router scores the experts for each token and sends the token to its
#  top-k experts. Only k of E experts run per token -> more capacity
#  (parameters) at a fixed compute budget per token.
# ══════════════════════════════════════════════════════════════════

class Expert(nn.Module):
    """A single SwiGLU feed-forward network (one expert)."""

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.gate = nn.Linear(d_model, d_hidden, bias=False)
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        # SwiGLU: (silu(gate(x)) * up(x)) -> down
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts layer with top-k routing and load-balancing loss.

    Args:
        d_model   : model width
        n_experts : total number of experts (E)
        top_k     : how many experts each token uses (k)
        d_hidden  : hidden width inside each expert
    """

    def __init__(self, d_model, n_experts=8, top_k=2, d_hidden=None):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        # keep active FLOPs roughly equal to one dense FFN
        d_hidden = d_hidden or 4 * d_model // top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_hidden) for _ in range(n_experts)])
        self.aux_loss = 0.0  # set each forward, read by the training loop

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)  # (B*L, D)

        # --- routing: score every expert for every token ---
        logits = self.router(x_flat)          # (n_tok, E)
        probs = F.softmax(logits, dim=-1)     # (n_tok, E)

        # pick the top-k experts per token, renormalize their weights
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)  # (n_tok, k)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # --- dispatch tokens to their experts and combine ---
        out = torch.zeros_like(x_flat)
        for slot in range(self.top_k):
            idx = topk_idx[:, slot]                     # which expert
            weight = topk_probs[:, slot].unsqueeze(-1)  # its gate weight
            for e in range(self.n_experts):
                mask = (idx == e)
                if mask.any():
                    out[mask] += weight[mask] * self.experts[e](x_flat[mask])

        # --- auxiliary load-balancing loss (Switch-Transformer style) ---
        # fraction of tokens dispatched to each expert...
        one_hot = F.one_hot(topk_idx, self.n_experts).sum(dim=1).float()  # (n_tok,E)
        tokens_per_expert = one_hot.mean(dim=0)          # (E,)
        mean_prob = probs.mean(dim=0)                    # (E,)
        # ...times mean router prob; minimized when both are uniform (1/E)
        self.aux_loss = self.n_experts * torch.sum(tokens_per_expert * mean_prob)

        return out.reshape(B, L, D)


# ══════════════════════════════════════════════════════════════════
#  Attention block mixer (exact recall, O(L^2))
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
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # fused scaled-dot-product attention (fast on MPS/CUDA), causal
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


# ══════════════════════════════════════════════════════════════════
#  Mamba block mixer (selective SSM, O(L))
# ══════════════════════════════════════════════════════════════════

def selective_scan(u, delta, A, B, C, D):
    """
    Sequential selective scan (clear and correct; MPS-friendly).
    u, delta: (b, d_inner, L)   A: (d_inner, N)
    B, C:     (b, N, L)         D: (d_inner,)
    Returns y: (b, d_inner, L)
    """
    b, d_inner, L = u.shape
    # discretize with input-dependent delta:  deltaA = exp(delta * A)
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2))
    B_bcast = B.permute(0, 2, 1).unsqueeze(1)                  # (b,1,L,N)
    deltaB_u = delta.unsqueeze(-1) * B_bcast * u.unsqueeze(-1)  # (b,d,L,N)
    h = torch.zeros(b, d_inner, A.shape[1], device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = deltaA[:, :, t] * h + deltaB_u[:, :, t]
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, :, t]))
    y = torch.stack(ys, dim=2)
    return y + u * D.unsqueeze(0).unsqueeze(-1)


class MambaMixer(nn.Module):
    """The selective-SSM mixer (no norm/residual — the Block adds those)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        # A initialized small-magnitude & negative -> stable, persistent memory
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        b, L, _ = x.shape
        x_ssm, z = self.in_proj(x).chunk(2, dim=-1)
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv1d(x_ssm)[:, :, :L]         # causal trim
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = F.silu(x_ssm)
        B, C, dt = torch.split(self.x_proj(x_ssm),
                               [self.d_state, self.d_state, 1], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)                   # always negative -> stable
        u = x_ssm.transpose(1, 2)
        delta = delta.transpose(1, 2)
        B = B.transpose(1, 2)
        C = C.transpose(1, 2)
        y = selective_scan(u, delta, A, B, C, self.D).transpose(1, 2)
        return self.out_proj(y * F.silu(z))          # gated output


# ══════════════════════════════════════════════════════════════════
#  Generic block: (sequence mixer) + (channel mixer)
#  -------------------------------------------------
#  mix over time (mamba OR attention), then mix over features
#  (dense FFN OR MoE). Each sub-layer has its own norm + residual.
# ══════════════════════════════════════════════════════════════════

class DenseFFN(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))
        self.aux_loss = 0.0

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model, mixer_kind, ffn_kind, n_heads, d_state,
                 n_experts, top_k, dropout):
        super().__init__()
        self.mixer_kind = mixer_kind
        self.ffn_kind = ffn_kind

        self.norm1 = nn.LayerNorm(d_model)
        if mixer_kind == "mamba":
            self.mixer = MambaMixer(d_model, d_state=d_state)
        elif mixer_kind == "attn":
            self.mixer = MultiHeadAttention(d_model, n_heads, dropout)
        else:
            raise ValueError(f"unknown mixer_kind: {mixer_kind}")

        self.norm2 = nn.LayerNorm(d_model)
        if ffn_kind == "moe":
            self.ffn = MoE(d_model, n_experts=n_experts, top_k=top_k)
        elif ffn_kind == "dense":
            self.ffn = DenseFFN(d_model, dropout)
        else:
            raise ValueError(f"unknown ffn_kind: {ffn_kind}")

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ══════════════════════════════════════════════════════════════════
#  The full model
# ══════════════════════════════════════════════════════════════════

class JambaLM(nn.Module):
    """
    A configurable Jamba-style language model.

    arch presets:
        "mamba"        — all mamba mixers, dense FFN
        "transformer"  — all attention mixers, dense FFN
        "hybrid"       — mostly mamba + attention every `attn_every`, dense FFN
        "jamba"        — hybrid mixers + MoE every `moe_every` layer (the full thing)
    """

    def __init__(self, vocab_size, arch="jamba", d_model=128, n_layers=8,
                 n_heads=4, d_state=16, context_len=256, dropout=0.1,
                 attn_every=4, moe_every=2, n_experts=8, top_k=2):
        super().__init__()
        self.arch = arch
        self.context_len = context_len
        self.token_emb = nn.Embedding(vocab_size, d_model)

        uses_attn = arch in ("transformer", "hybrid", "jamba")
        self.pos_emb = nn.Embedding(context_len, d_model) if uses_attn else None
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        self.layer_kinds = []
        for i in range(n_layers):
            # sequence mixer for this layer
            if arch == "transformer":
                mk = "attn"
            elif arch == "mamba":
                mk = "mamba"
            else:  # hybrid or jamba
                mk = "attn" if ((i + 1) % attn_every == 0) else "mamba"

            # channel mixer for this layer
            if arch == "jamba":
                fk = "moe" if ((i + 1) % moe_every == 0) else "dense"
            else:
                fk = "dense"

            self.blocks.append(Block(d_model, mk, fk, n_heads, d_state,
                                     n_experts, top_k, dropout))
            self.layer_kinds.append((mk, fk))

        # guarantee at least one attention layer for hybrid/jamba
        if arch in ("hybrid", "jamba") and not any(mk == "attn" for mk, _ in self.layer_kinds):
            mid = n_layers // 2
            _, fk = self.layer_kinds[mid]
            self.blocks[mid] = Block(d_model, "attn", fk, n_heads, d_state,
                                     n_experts, top_k, dropout)
            self.layer_kinds[mid] = ("attn", fk)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, aux_weight=0.01):
        B, T = idx.shape
        x = self.token_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # add the MoE load-balancing auxiliary loss from every MoE layer
            aux = sum((b.ffn.aux_loss for b in self.blocks if isinstance(b.ffn, MoE)),
                      start=torch.tensor(0.0, device=idx.device))
            if isinstance(aux, torch.Tensor) and aux.requires_grad:
                loss = loss + aux_weight * aux
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
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    # -------- introspection --------
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def active_params_per_token(self):
        """Params actually used per token (MoE runs only top_k of n_experts)."""
        total = 0
        for name, p in self.named_parameters():
            if ".experts." not in name:
                total += p.numel()
        # add the active fraction of expert params
        for b in self.blocks:
            if isinstance(b.ffn, MoE):
                exp_params = sum(pp.numel() for e in b.ffn.experts for pp in e.parameters())
                total += int(exp_params * b.ffn.top_k / b.ffn.n_experts)
        return total

    def layer_summary(self):
        parts = []
        for mk, fk in self.layer_kinds:
            m = "A" if mk == "attn" else "m"
            f = "E" if fk == "moe" else "."
            parts.append(m + f)
        n_attn = sum(1 for mk, _ in self.layer_kinds if mk == "attn")
        n_mamba = len(self.layer_kinds) - n_attn
        n_moe = sum(1 for _, fk in self.layer_kinds if fk == "moe")
        return (f"[{' '.join(parts)}]  "
                f"({n_mamba} mamba, {n_attn} attn; {n_moe} MoE)   "
                f"legend: m=mamba A=attn .=dense E=MoE")
