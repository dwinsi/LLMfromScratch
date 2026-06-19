"""
Project 9: RMSNorm + RoPE + SwiGLU

Three architectural changes that together produce a LLaMA-style Transformer.
Each change is small in isolation. Together they make a meaningful difference.

Changes from Project 8:
  RMSNorm    replaces nn.LayerNorm      - simpler normalisation, no mean subtraction
  RoPE       replaces sinusoidal encoding - positional rotation inside attention
  SwiGLU     replaces GELU FFN          - gated feed forward, more expressive

What stays the same:
  BPE tokenisation from Project 8
  Four Transformer blocks
  Multi-head attention (64 dim, 4 heads)
  Batching, cosine annealing, gradient clipping
  DataLoader for memory-efficient training

Install requirements:
  pip install torch tokenizers matplotlib
"""

import json
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from torch.utils.data import TensorDataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

_cfg = json.loads((pathlib.Path(__file__).parent / "config.json").read_text())
_model_cfg = _cfg["model"]
_train_cfg = _cfg["training"]

torch.manual_seed(_train_cfg["seed"])

# ---- Device setup ----
try:
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
except Exception:
    device = torch.device('cpu')

print(f"Using device: {device}")


# ---- Load corpus and train BPE tokeniser ----

corpus_file_path    = 'weather_corpus_v2.txt'
tokenizer_save_path = 'weather_bpe_tokenizer.json'

with open(corpus_file_path, 'r') as f:
    corpus_text = f.read().lower()

bpe_tokenizer             = Tokenizer(BPE(unk_token="[UNK]"))
bpe_tokenizer.pre_tokenizer = ByteLevel()
bpe_tokenizer.decoder       = ByteLevelDecoder()

bpe_trainer = BpeTrainer(
    vocab_size=_model_cfg["vocab_size"],
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    min_frequency=1
)

bpe_tokenizer.train(files=[corpus_file_path], trainer=bpe_trainer)
bpe_tokenizer.save(tokenizer_save_path)

vocabulary_size = bpe_tokenizer.get_vocab_size()
print(f"Vocabulary size: {vocabulary_size}")


# ---- Build training sequences ----

batch_size         = _train_cfg["batch_size"]
sequence_length    = _model_cfg["sequence_length"]
training_sequences = []
training_targets   = []

all_token_ids = bpe_tokenizer.encode(corpus_text).ids

split_idx = int(0.8 * len(all_token_ids))
train_ids = all_token_ids[:split_idx]
val_ids   = all_token_ids[split_idx:]

for i in range(len(train_ids) - sequence_length):
    training_sequences.append(train_ids[i : i + sequence_length])
    training_targets.append(train_ids[i + sequence_length])

val_sequences = []
val_targets   = []
for i in range(len(val_ids) - sequence_length):
    val_sequences.append(val_ids[i : i + sequence_length])
    val_targets.append(val_ids[i + sequence_length])

sequences_tensor = torch.tensor(training_sequences)
targets_tensor   = torch.tensor(training_targets)
val_seq_tensor   = torch.tensor(val_sequences)
val_tgt_tensor   = torch.tensor(val_targets)

training_dataset = TensorDataset(sequences_tensor, targets_tensor)
training_loader  = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
val_dataset      = TensorDataset(val_seq_tensor, val_tgt_tensor)
val_loader       = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Training sequences:   {len(training_sequences)}")
print(f"Validation sequences: {len(val_sequences)}")


# ============================================================
# NEW COMPONENTS
# ============================================================

# ---- 1. RMSNorm ----

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation.
    Simpler than LayerNorm: removes mean subtraction, keeps only RMS scaling.
    One fewer learned parameter (no bias/beta term).

    LayerNorm:  normalise by (x - mean) / sqrt(variance)  then scale + shift
    RMSNorm:    normalise by x / sqrt(mean(x^2))           then scale only
    """

    def __init__(self, embedding_dim, epsilon=1e-6):
        super(RMSNorm, self).__init__()
        self.epsilon      = epsilon
        self.learned_scale = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x):
        # Compute root mean square across embedding dimension
        rms         = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.epsilon)
        normalised  = x / rms
        return self.learned_scale * normalised


# ---- 2. RoPE ----

def compute_rope_frequencies(head_dim, max_seq_len, base=10000, device='cpu'):
    """
    Precompute RoPE rotation angles for all positions and dimension pairs.

    Returns cos and sin tables of shape (max_seq_len, head_dim/2).
    Each position gets a unique set of rotation angles across dimension pairs.
    """
    # Dimension pairs: (0,1), (2,3), (4,5) ... each pair shares one angle
    dimension_pair_indices = torch.arange(0, head_dim, 2, device=device).float()
    frequencies            = 1.0 / (base ** (dimension_pair_indices / head_dim))

    # One angle per position per frequency
    positions = torch.arange(max_seq_len, device=device).float()
    angles    = torch.outer(positions, frequencies)   # (max_seq_len, head_dim/2)

    return torch.cos(angles), torch.sin(angles)


def apply_rope(query_or_key, cos_table, sin_table):
    """
    Apply rotary positional encoding to query or key vectors.

    Instead of adding position to embeddings (sinusoidal),
    RoPE rotates Q and K vectors by position-dependent angles.
    The rotation happens inside each attention head.

    query_or_key shape: (batch, seq_len, num_heads, head_dim)
    """
    seq_len  = query_or_key.shape[1]
    cos_vals = cos_table[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim/2)
    sin_vals = sin_table[:seq_len].unsqueeze(0).unsqueeze(2)

    # Split into even and odd dimensions
    x_even = query_or_key[..., 0::2]   # dimensions 0, 2, 4, ...
    x_odd  = query_or_key[..., 1::2]   # dimensions 1, 3, 5, ...

    # Apply 2D rotation to each dimension pair
    x_rotated_even = x_even * cos_vals - x_odd * sin_vals
    x_rotated_odd  = x_even * sin_vals + x_odd * cos_vals

    # Interleave even and odd back together
    x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    x_rotated = x_rotated.flatten(-2)   # (batch, seq_len, num_heads, head_dim)

    return x_rotated


# ---- 3. SwiGLU Feed Forward ----

class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed Forward Network.
    Replaces the standard expand -> GELU -> compress with a gated mechanism.

    Standard FFN:
      hidden = GELU(x @ W_expand)
      output = hidden @ W_compress

    SwiGLU:
      gate   = SiLU(x @ W_gate)      <- learned gate (SiLU = x * sigmoid(x))
      value  = x @ W_value            <- learned values
      hidden = gate * value           <- gate controls what passes through
      output = hidden @ W_compress    <- compress back

    Uses three weight matrices instead of two.
    Hidden dim is set to 2/3 of the standard FFN dim to keep param count comparable.
    """

    def __init__(self, embedding_dim, feedforward_hidden_dim):
        super(SwiGLUFeedForward, self).__init__()

        self.gate_projection    = nn.Linear(embedding_dim, feedforward_hidden_dim, bias=False)
        self.value_projection   = nn.Linear(embedding_dim, feedforward_hidden_dim, bias=False)
        self.compress_projection = nn.Linear(feedforward_hidden_dim, embedding_dim, bias=False)
        self.dropout            = nn.Dropout(0.1)

    def forward(self, x):
        gate   = F.silu(self.gate_projection(x))    # SiLU = Swish activation
        value  = self.value_projection(x)
        hidden = gate * value                        # element-wise gating
        hidden = self.dropout(hidden)
        return self.compress_projection(hidden)


# ============================================================
# UPDATED TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    """
    LLaMA-style Transformer block.

    Changes from Project 8:
      LayerNorm  -> RMSNorm
      Sinusoidal -> RoPE (applied inside attention)
      GELU FFN   -> SwiGLU FFN
    """

    def __init__(self, embedding_dim, number_of_attention_heads,
                 feedforward_hidden_dim, dropout_rate, max_seq_len):
        super(TransformerBlock, self).__init__()

        self.number_of_attention_heads    = number_of_attention_heads
        self.head_dim                     = embedding_dim // number_of_attention_heads
        self.embedding_dim                = embedding_dim

        # RMSNorm replaces LayerNorm
        self.rms_norm_before_attention    = RMSNorm(embedding_dim)
        self.rms_norm_before_feedforward  = RMSNorm(embedding_dim)

        # Attention projections (manual so we can apply RoPE)
        self.query_projection  = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_projection    = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_projection  = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.attention_dropout = nn.Dropout(dropout_rate)

        # SwiGLU replaces GELU feed forward
        self.swiglu_feedforward = SwiGLUFeedForward(embedding_dim, feedforward_hidden_dim)

        # Precompute RoPE frequencies for this block
        cos_table, sin_table = compute_rope_frequencies(
            self.head_dim, max_seq_len
        )
        self.register_buffer('rope_cos', cos_table)
        self.register_buffer('rope_sin', sin_table)

    def forward(self, token_representations, causal_mask):
        batch_size = token_representations.shape[0]
        seq_len    = token_representations.shape[1]

        # ---- Pre-norm (RMSNorm before attention) ----
        normed = self.rms_norm_before_attention(token_representations)

        # ---- Compute Q, K, V ----
        Q = self.query_projection(normed)    # (batch, seq_len, embed_dim)
        K = self.key_projection(normed)
        V = self.value_projection(normed)

        # Reshape to (batch, seq_len, num_heads, head_dim) for RoPE
        Q = Q.view(batch_size, seq_len, self.number_of_attention_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.number_of_attention_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.number_of_attention_heads, self.head_dim)

        # ---- Apply RoPE to Q and K ----
        Q = apply_rope(Q, self.rope_cos, self.rope_sin)
        K = apply_rope(K, self.rope_cos, self.rope_sin)

        # Reshape to (batch, num_heads, seq_len, head_dim) for attention
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # ---- Scaled dot-product attention ----
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        attention_scores = attention_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float('-inf')
        )

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # Weighted sum of values
        attention_output = torch.matmul(attention_weights, V)

        # Reshape back to (batch, seq_len, embed_dim)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.embedding_dim)
        attention_output = self.output_projection(attention_output)

        # ---- Residual connection ----
        token_representations = token_representations + attention_output

        # ---- Pre-norm (RMSNorm before feed forward) ----
        normed = self.rms_norm_before_feedforward(token_representations)

        # ---- SwiGLU feed forward ----
        feedforward_output = self.swiglu_feedforward(normed)

        # ---- Residual connection ----
        token_representations = token_representations + feedforward_output

        return token_representations


# ============================================================
# MINI LANGUAGE MODEL
# ============================================================

class MiniLanguageModel(nn.Module):
    """
    LLaMA-style mini language model.
    RMSNorm + RoPE + SwiGLU throughout.
    No sinusoidal positional encoding: RoPE handles position inside attention.
    """

    def __init__(self, vocabulary_size, embedding_dim, number_of_attention_heads,
                 feedforward_hidden_dim, number_of_blocks, dropout_rate, max_sequence_length):
        super(MiniLanguageModel, self).__init__()

        self.word_embedding   = nn.Embedding(vocabulary_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate)

        # No positional encoding buffer needed: RoPE is inside each block

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                number_of_attention_heads=number_of_attention_heads,
                feedforward_hidden_dim=feedforward_hidden_dim,
                dropout_rate=dropout_rate,
                max_seq_len=max_sequence_length
            )
            for _ in range(number_of_blocks)
        ])

        # Final RMSNorm before output projection
        self.final_rms_norm    = RMSNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocabulary_size, bias=False)

    def _build_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        ).bool()

    def forward(self, token_indices):
        seq_len               = token_indices.shape[1]
        token_representations = self.word_embedding(token_indices)
        token_representations = self.embedding_dropout(token_representations)

        causal_mask = self._build_causal_mask(seq_len, token_indices.device)

        for transformer_block in self.transformer_blocks:
            token_representations = transformer_block(token_representations, causal_mask)

        token_representations     = self.final_rms_norm(token_representations)
        last_token_representation = token_representations[:, -1, :]
        return self.output_projection(last_token_representation)


# ---- Initialise model ----

embedding_dim             = _model_cfg["embedding_dim"]
number_of_attention_heads = _model_cfg["number_of_attention_heads"]
feedforward_hidden_dim    = _model_cfg["feedforward_hidden_dim"]
number_of_blocks          = _model_cfg["number_of_blocks"]
dropout_rate              = _model_cfg["dropout_rate"]
learning_rate             = _train_cfg["learning_rate"]
number_of_epochs          = _train_cfg["epochs"]

model = MiniLanguageModel(
    vocabulary_size=vocabulary_size,
    embedding_dim=embedding_dim,
    number_of_attention_heads=number_of_attention_heads,
    feedforward_hidden_dim=feedforward_hidden_dim,
    number_of_blocks=number_of_blocks,
    dropout_rate=dropout_rate,
    max_sequence_length=sequence_length
).to(device)

loss_function = nn.CrossEntropyLoss()
optimiser     = optim.Adam(model.parameters(), lr=learning_rate)
scheduler     = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=number_of_epochs)

total_parameters = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_parameters:,}")
print(f"Project 8 had:    167,040 (BPE + standard Transformer)")


# ---- Training loop ----

training_loss_history = []
val_loss_history      = []

for epoch in range(number_of_epochs):
    model.train()
    total_loss  = 0
    num_batches = 0

    for batch_sequences, batch_targets in training_loader:
        batch_sequences = batch_sequences.to(device)
        batch_targets   = batch_targets.to(device)

        optimiser.zero_grad()
        output_scores = model(batch_sequences)
        loss          = loss_function(output_scores, batch_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss  += loss.item()
        num_batches += 1

    scheduler.step()
    average_epoch_loss = total_loss / num_batches
    training_loss_history.append(average_epoch_loss)

    model.eval()
    val_total = 0
    val_batches = 0
    with torch.no_grad():
        for batch_seq, batch_tgt in val_loader:
            out = model(batch_seq.to(device))
            val_total   += loss_function(out, batch_tgt.to(device)).item()
            val_batches += 1
    val_loss_history.append(val_total / val_batches)

    if epoch % 400 == 0:
        print(f"Epoch {epoch:5d}  train: {average_epoch_loss:.4f}  val: {val_loss_history[-1]:.4f}  lr: {scheduler.get_last_lr()[0]:.6f}")


# ---- Text generation ----

def generate_text(seed_text, number_of_tokens_to_generate=16, temperature=0.8):
    model.eval()
    generated_ids = bpe_tokenizer.encode(seed_text.lower()).ids.copy()

    with torch.no_grad():
        for _ in range(number_of_tokens_to_generate):
            context_ids     = generated_ids[-sequence_length:]
            sequence_tensor = torch.tensor(context_ids).unsqueeze(0).to(device)
            output_scores   = model(sequence_tensor)

            if temperature == 0.0:
                predicted_id = torch.argmax(output_scores, dim=-1).item()
            else:
                probabilities = torch.softmax(output_scores / temperature, dim=-1)
                predicted_id  = torch.multinomial(probabilities, num_samples=1).item()

            generated_ids.append(predicted_id)

    decoded = bpe_tokenizer.decode(generated_ids)
    return ' '.join(decoded.split())


print()
print("Generated text (temperature=0.8):")
print(" ", generate_text("the sky is cloudy"))
print(" ", generate_text("bring your umbrella"))
print(" ", generate_text("dark clouds mean"))
print(" ", generate_text("the rain will"))
print(" ", generate_text("a clear sky"))

print(f"\nFinal loss: {training_loss_history[-1]:.4f}")


# ---- Plot loss curve ----

plt.figure(figsize=(10, 5))
plt.plot(training_loss_history, color='steelblue', linewidth=1.5,
         label=f'Training loss — RMSNorm + RoPE + SwiGLU ({total_parameters:,} params)')
plt.plot(val_loss_history,      color='tomato',    linewidth=1.5,
         label='Validation loss', linestyle='--')
plt.axhline(
    y=math.log(vocabulary_size),
    color='gray', linestyle=':', linewidth=1,
    label=f'Random baseline: {math.log(vocabulary_size):.2f}'
)
plt.title('Mini LLM: LLaMA-style Architecture Training vs Validation Loss', fontsize=13)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Cross-Entropy Loss', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('loss_curve_llama_style.png', dpi=150)
plt.show()

print("Loss curve saved to loss_curve_llama_style.png")