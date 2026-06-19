"""
Ablation Study: Dense SwiGLU FFN vs Mixture of Experts (MoE) FFN
Both models use RMSNorm, RoPE, and GQA (4Q/2KV) for attention.
MoE uses 8 experts with top-2 routing and an auxiliary load-balancing loss.
"""

import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 8
BATCH_SIZE = 32
EMBEDDING_DIM = 64
FF_HIDDEN_DIM = 128
NUM_BLOCKS = 4
DROPOUT_RATE = 0.1
LEARNING_RATE = 0.001
NUM_EPOCHS = 500
GRAD_CLIP = 1.0
VOCAB_SIZE = 256
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = EMBEDDING_DIM // NUM_Q_HEADS  # 16
NUM_EXPERTS = 8
TOP_K = 2
AUX_LOSS_WEIGHT = 0.01
SEED = 42

HERE = pathlib.Path(__file__).parent
CORPUS_PATH = HERE / ".." / "09-mini_llm_llama_style" / "weather_corpus_v2.txt"
OUTPUT_PNG = HERE / "dense_vs_moe.png"

# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------

def build_tokenizer(corpus_path: pathlib.Path) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        min_frequency=1,
    )
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)
    return tokenizer


def build_dataset(corpus_path: pathlib.Path, tokenizer: Tokenizer, seq_len: int):
    text = corpus_path.read_text(encoding="utf-8")
    encoding = tokenizer.encode(text)
    ids = torch.tensor(encoding.ids, dtype=torch.long)
    X, Y = [], []
    for i in range(0, len(ids) - seq_len, 1):
        X.append(ids[i : i + seq_len])
        Y.append(ids[i + 1 : i + seq_len + 1])
    X = torch.stack(X)
    Y = torch.stack(Y)
    return TensorDataset(X, Y)

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))

# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def precompute_rope_freqs(head_dim: int, max_len: int, device: torch.device):
    half = head_dim // 2
    theta = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32, device=device) * 2 / head_dim))
    positions = torch.arange(max_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, theta)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, num_heads, T, head_dim)"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_ = cos[:x.size(2)].unsqueeze(0).unsqueeze(0)
    sin_ = sin[:x.size(2)].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos_ - x2 * sin_, x1 * sin_ + x2 * cos_], dim=-1)

# ---------------------------------------------------------------------------
# GQA Attention (4Q / 2KV) with RoPE
# ---------------------------------------------------------------------------

class GQAttention(nn.Module):
    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int, dropout: float):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.groups = num_q_heads // num_kv_heads
        self.head_dim = embed_dim // num_q_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        Hq, Hkv, D = self.num_q_heads, self.num_kv_heads, self.head_dim
        q = self.q_proj(x).view(B, T, Hq, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn = (q @ k.transpose(-2, -1)) * self.scale + causal
        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

# ---------------------------------------------------------------------------
# Mixture of Experts FFN
# ---------------------------------------------------------------------------

class MixtureOfExperts(nn.Module):
    """
    8 SwiGLU expert networks with top-2 routing.
    Returns (output, aux_loss) where aux_loss is the load-balancing loss.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_experts: int, top_k: int, dropout: float):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Each expert is a SwiGLU FFN
        self.experts = nn.ModuleList(
            [SwiGLUFFN(embed_dim, hidden_dim, dropout) for _ in range(num_experts)]
        )
        self.router = nn.Linear(embed_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        flat = x.view(-1, C)  # (B*T, C)
        router_logits = self.router(flat)  # (B*T, num_experts)
        router_probs = torch.softmax(router_logits, dim=-1)  # (B*T, num_experts)

        # Top-k selection
        topk_vals, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)  # (B*T, top_k)
        # Renormalize selected weights
        topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # (B*T, top_k)

        # Compute expert outputs for each selected expert
        output = torch.zeros_like(flat)  # (B*T, C)
        for k in range(self.top_k):
            expert_indices = topk_idx[:, k]   # (B*T,)
            weights = topk_weights[:, k]       # (B*T,)
            for e in range(self.num_experts):
                mask = (expert_indices == e)   # bool (B*T,)
                if mask.any():
                    expert_out = self.experts[e](flat[mask])  # (n_e, C)
                    output[mask] += weights[mask].unsqueeze(-1) * expert_out

        output = output.view(B, T, C)

        # Auxiliary load-balancing loss
        # fraction_i = fraction of tokens routed to expert i
        # mean_prob_i = mean router probability for expert i
        # aux_loss = num_experts * sum(fraction_i * mean_prob_i)
        one_hot = torch.zeros(flat.size(0), self.num_experts, device=x.device)
        for k in range(self.top_k):
            one_hot.scatter_add_(1, topk_idx[:, k].unsqueeze(1), torch.ones(flat.size(0), 1, device=x.device))
        one_hot = one_hot / self.top_k  # normalise so fractions sum to 1

        fraction = one_hot.mean(dim=0)           # (num_experts,)
        mean_prob = router_probs.mean(dim=0)     # (num_experts,)
        aux_loss = self.num_experts * (fraction * mean_prob).sum()

        return output, aux_loss

# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class DenseBlock(nn.Module):
    def __init__(self, embed_dim, num_q_heads, num_kv_heads, ff_hidden, dropout):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = GQAttention(embed_dim, num_q_heads, num_kv_heads, dropout)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLUFFN(embed_dim, ff_hidden, dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x, 0.0  # no aux loss


class MoEBlock(nn.Module):
    def __init__(self, embed_dim, num_q_heads, num_kv_heads, ff_hidden, num_experts, top_k, dropout):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = GQAttention(embed_dim, num_q_heads, num_kv_heads, dropout)
        self.norm2 = RMSNorm(embed_dim)
        self.moe = MixtureOfExperts(embed_dim, ff_hidden, num_experts, top_k, dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        ffn_out, aux_loss = self.moe(self.norm2(x))
        x = x + ffn_out
        return x, aux_loss

# ---------------------------------------------------------------------------
# Full models
# ---------------------------------------------------------------------------

class DenseModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_q_heads, num_kv_heads, ff_hidden, num_blocks, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_q_heads
        self.blocks = nn.ModuleList(
            [DenseBlock(embed_dim, num_q_heads, num_kv_heads, ff_hidden, dropout) for _ in range(num_blocks)]
        )
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        T = x.size(1)
        device = x.device
        cos, sin = precompute_rope_freqs(self.head_dim, T, device)
        h = self.drop(self.embedding(x))
        total_aux = 0.0
        for block in self.blocks:
            h, aux = block(h, cos, sin)
            total_aux = total_aux + aux
        return self.lm_head(self.norm(h)), total_aux


class MoEModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_q_heads, num_kv_heads, ff_hidden,
                 num_blocks, num_experts, top_k, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_q_heads
        self.blocks = nn.ModuleList(
            [MoEBlock(embed_dim, num_q_heads, num_kv_heads, ff_hidden, num_experts, top_k, dropout)
             for _ in range(num_blocks)]
        )
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        T = x.size(1)
        device = x.device
        cos, sin = precompute_rope_freqs(self.head_dim, T, device)
        h = self.drop(self.embedding(x))
        total_aux = torch.tensor(0.0, device=device)
        for block in self.blocks:
            h, aux = block(h, cos, sin)
            total_aux = total_aux + aux
        return self.lm_head(self.norm(h)), total_aux

# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_dense(model: DenseModel, loader: DataLoader, device: torch.device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    epoch_losses = []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        scheduler.step()
        avg_loss = total_loss / batches
        epoch_losses.append(avg_loss)
        if epoch % 100 == 0:
            print(f"  [Dense] Epoch {epoch:4d}/{NUM_EPOCHS}  loss={avg_loss:.4f}")
    return epoch_losses


def train_moe(model: MoEModel, loader: DataLoader, device: torch.device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    epoch_ce_losses = []
    epoch_aux_losses = []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_ce = 0.0
        total_aux = 0.0
        batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, aux_loss = model(xb)
            ce_loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss = ce_loss + AUX_LOSS_WEIGHT * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            total_ce += ce_loss.item()
            total_aux += aux_loss.item()
            batches += 1
        scheduler.step()
        avg_ce = total_ce / batches
        avg_aux = total_aux / batches
        epoch_ce_losses.append(avg_ce)
        epoch_aux_losses.append(avg_aux)
        if epoch % 100 == 0:
            print(f"  [MoE]   Epoch {epoch:4d}/{NUM_EPOCHS}  CE={avg_ce:.4f}  aux={avg_aux:.4f}")
    return epoch_ce_losses, epoch_aux_losses

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building tokenizer...")
    tokenizer = build_tokenizer(CORPUS_PATH)
    actual_vocab = tokenizer.get_vocab_size()
    print(f"Vocab size: {actual_vocab}")

    print("Building dataset...")
    dataset = build_dataset(CORPUS_PATH, tokenizer, SEQUENCE_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"Dataset sequences: {len(dataset)}, batches per epoch: {len(loader)}")

    torch.manual_seed(SEED)
    dense_model = DenseModel(
        vocab_size=actual_vocab,
        embed_dim=EMBEDDING_DIM,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        ff_hidden=FF_HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT_RATE,
    )

    torch.manual_seed(SEED)
    moe_model = MoEModel(
        vocab_size=actual_vocab,
        embed_dim=EMBEDDING_DIM,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        ff_hidden=FF_HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        dropout=DROPOUT_RATE,
    )

    print(f"\nDense params: {sum(p.numel() for p in dense_model.parameters()):,}")
    print(f"MoE   params: {sum(p.numel() for p in moe_model.parameters()):,}")

    print("\n--- Training Dense Model ---")
    dense_losses = train_dense(dense_model, loader, device)

    print("\n--- Training MoE Model ---")
    moe_ce_losses, moe_aux_losses = train_moe(moe_model, loader, device)

    # Plot
    epochs = list(range(1, NUM_EPOCHS + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, dense_losses, label="Dense SwiGLU", color="steelblue")
    axes[0].plot(epochs, moe_ce_losses, label="MoE (8 experts, top-2)", color="darkorange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training CE Loss: Dense vs MoE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, moe_aux_losses, color="seagreen", label="MoE Aux Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Aux (Load-Balancing) Loss")
    axes[1].set_title("MoE Load-Balancing Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Ablation: Dense SwiGLU FFN vs Mixture of Experts", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_PNG), dpi=150)
    print(f"\nSaved plot to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    main()
