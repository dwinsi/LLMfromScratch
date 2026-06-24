"""
Ablation Study: Grouped Query Attention (GQA) vs Multi-Head Attention (MHA)
Both models use RMSNorm, RoPE, and SwiGLU FFN. Attention is implemented from scratch.
"""

import pathlib
import torch
import torch.nn as nn
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
NUM_KV_HEADS_GQA = 2
HEAD_DIM = EMBEDDING_DIM // NUM_Q_HEADS  # 16
SEED = 42

HERE = pathlib.Path(__file__).parent
CORPUS_PATH = HERE / ".." / "09-mini_llm_llama_style" / "weather_corpus_v2.txt"
OUTPUT_PNG = HERE / "gqa_vs_mha.png"

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
        gate = torch.nn.functional.silu(self.gate_proj(x))
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
    B, H, T, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_ = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, half)
    sin_ = sin[:T].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos_ - x2 * sin_, x1 * sin_ + x2 * cos_], dim=-1)

# ---------------------------------------------------------------------------
# MHA from scratch (4Q / 4K / 4V)
# ---------------------------------------------------------------------------

class MHAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn = (q @ k.transpose(-2, -1)) * self.scale + causal
        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

# ---------------------------------------------------------------------------
# GQA (4Q / 2KV)
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
        q = self.q_proj(x).view(B, T, Hq, D).transpose(1, 2)   # (B, Hq, T, D)
        k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)  # (B, Hkv, T, D)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        # Apply RoPE to all Q heads and KV heads
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # Expand KV heads to match Q heads
        k = k.repeat_interleave(self.groups, dim=1)  # (B, Hq, T, D)
        v = v.repeat_interleave(self.groups, dim=1)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn = (q @ k.transpose(-2, -1)) * self.scale + causal
        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class MHABlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MHAttention(embed_dim, num_heads, dropout)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLUFFN(embed_dim, ff_hidden, dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class GQABlock(nn.Module):
    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int, ff_hidden: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = GQAttention(embed_dim, num_q_heads, num_kv_heads, dropout)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLUFFN(embed_dim, ff_hidden, dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x

# ---------------------------------------------------------------------------
# Full models
# ---------------------------------------------------------------------------

class MHAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden, num_blocks, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        self.blocks = nn.ModuleList(
            [MHABlock(embed_dim, num_heads, ff_hidden, dropout) for _ in range(num_blocks)]
        )
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        device = x.device
        cos, sin = precompute_rope_freqs(self.head_dim, T, device)
        h = self.drop(self.embedding(x))
        for block in self.blocks:
            h = block(h, cos, sin)
        return self.lm_head(self.norm(h))


class GQAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_q_heads, num_kv_heads, ff_hidden, num_blocks, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_q_heads
        self.blocks = nn.ModuleList(
            [GQABlock(embed_dim, num_q_heads, num_kv_heads, ff_hidden, dropout) for _ in range(num_blocks)]
        )
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        device = x.device
        cos, sin = precompute_rope_freqs(self.head_dim, T, device)
        h = self.drop(self.embedding(x))
        for block in self.blocks:
            h = block(h, cos, sin)
        return self.lm_head(self.norm(h))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(model: nn.Module, loader: DataLoader, device: torch.device, label: str):
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
            logits = model(xb)
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
            print(f"  [{label}] Epoch {epoch:4d}/{NUM_EPOCHS}  loss={avg_loss:.4f}")
    return epoch_losses

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    device = torch.device("cpu")
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
    mha_model = MHAModel(
        vocab_size=actual_vocab,
        embed_dim=EMBEDDING_DIM,
        num_heads=NUM_Q_HEADS,
        ff_hidden=FF_HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT_RATE,
    )

    torch.manual_seed(SEED)
    gqa_model = GQAModel(
        vocab_size=actual_vocab,
        embed_dim=EMBEDDING_DIM,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS_GQA,
        ff_hidden=FF_HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT_RATE,
    )

    print(f"\nMHA params: {sum(p.numel() for p in mha_model.parameters()):,}")
    print(f"GQA params: {sum(p.numel() for p in gqa_model.parameters()):,}")

    print("\n--- Training MHA Model ---")
    mha_losses = train_model(mha_model, loader, device, "MHA")

    print("\n--- Training GQA Model ---")
    gqa_losses = train_model(gqa_model, loader, device, "GQA")

    # Plot
    epochs = list(range(1, NUM_EPOCHS + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, mha_losses, label="MHA (4Q/4KV)", color="steelblue")
    ax.plot(epochs, gqa_losses, label="GQA (4Q/2KV)", color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Ablation: GQA vs Standard MHA")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_PNG), dpi=150)
    print(f"\nSaved plot to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    main()
