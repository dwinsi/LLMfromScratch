"""
Project 8: BPE Tokenisation with HuggingFace Tokenizers.

Replaces the word-level tokenisation from Project 7 with a custom
BPE tokenizer trained on the weather corpus using HuggingFace tokenizers.

What changes from Project 7:
  - Tokenisation: word-level -> BPE subword tokens
  - Vocabulary: 198 words -> ~256 BPE tokens
  - Tokenizer: hand-built vocab dict -> HuggingFace BPE trainer
  - Everything else: same architecture, same training loop

Install:
  pip install tokenizers torch matplotlib
"""

import json
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
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
# Catches CUDA errors gracefully and falls back to CPU
try:
    if torch.cuda.is_available():
        # Test if CUDA is actually usable, not just present
        torch.zeros(1).cuda()
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
except Exception:
    device = torch.device('cpu')

print(f"Using device: {device}")


# ---- Step 1: Train BPE tokenizer on the weather corpus ----

corpus_file_path    = 'weather_corpus_v2.txt'
tokenizer_save_path = 'weather_bpe_tokenizer.json'

bpe_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# ByteLevel pre-tokenizer encodes spaces as Ġ character
# This is what GPT-2 uses and ensures correct reconstruction when decoding
bpe_tokenizer.pre_tokenizer = ByteLevel()
bpe_tokenizer.decoder       = ByteLevelDecoder()

bpe_trainer = BpeTrainer(
    vocab_size=_model_cfg["vocab_size"],
    special_tokens=_model_cfg["special_tokens"],
    min_frequency=_model_cfg["min_frequency"]
)

bpe_tokenizer.train(files=[corpus_file_path], trainer=bpe_trainer)
bpe_tokenizer.save(tokenizer_save_path)

vocabulary_size = bpe_tokenizer.get_vocab_size()
print(f"BPE tokenizer trained and saved to {tokenizer_save_path}")
print(f"Vocabulary size: {vocabulary_size}")


# ---- Step 2: Encode the corpus into token sequences ----

with open(corpus_file_path, 'r') as f:
    corpus_text = f.read().lower()

encoded_corpus = bpe_tokenizer.encode(corpus_text)
all_token_ids  = encoded_corpus.ids

print(f"Total tokens in corpus: {len(all_token_ids)}")

# Show a sample encoding
sample_sentence = "the sky is cloudy today"
sample_encoded  = bpe_tokenizer.encode(sample_sentence)
print(f"\nSample: '{sample_sentence}'")
print(f"  Tokens:    {sample_encoded.tokens}")
print(f"  Token ids: {sample_encoded.ids}")


# ---- Step 3: Build training sequences ----

sequence_length    = _model_cfg["sequence_length"]
batch_size         = _train_cfg["batch_size"]
training_sequences = []
training_targets   = []

split_idx    = int(0.8 * len(all_token_ids))
train_ids    = all_token_ids[:split_idx]
val_ids      = all_token_ids[split_idx:]

for i in range(len(train_ids) - sequence_length):
    training_sequences.append(train_ids[i : i + sequence_length])
    training_targets.append(train_ids[i + sequence_length])

val_sequences = []
val_targets   = []
for i in range(len(val_ids) - sequence_length):
    val_sequences.append(val_ids[i : i + sequence_length])
    val_targets.append(val_ids[i + sequence_length])

from torch.utils.data import TensorDataset, DataLoader

sequences_tensor = torch.tensor(training_sequences)
targets_tensor   = torch.tensor(training_targets)
val_seq_tensor   = torch.tensor(val_sequences)
val_tgt_tensor   = torch.tensor(val_targets)

training_dataset = TensorDataset(sequences_tensor, targets_tensor)
training_loader  = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_seq_tensor, val_tgt_tensor)
val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"\nSequence length:      {sequence_length}")
print(f"Training sequences:   {len(training_sequences)}")
print(f"Validation sequences: {len(val_sequences)}")


# ---- Transformer block (identical to Project 7) ----

class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, number_of_attention_heads,
                 feedforward_hidden_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=number_of_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.layer_norm_after_attention   = nn.LayerNorm(embedding_dim)
        self.feedforward_expand           = nn.Linear(embedding_dim, feedforward_hidden_dim)
        self.feedforward_compress         = nn.Linear(feedforward_hidden_dim, embedding_dim)
        self.feedforward_activation       = nn.GELU()
        self.feedforward_dropout          = nn.Dropout(dropout_rate)
        self.layer_norm_after_feedforward = nn.LayerNorm(embedding_dim)
        self.attention_dropout            = nn.Dropout(dropout_rate)

    def forward(self, token_representations, causal_mask):
        attention_output, _ = self.multihead_attention(
            query=token_representations,
            key=token_representations,
            value=token_representations,
            attn_mask=causal_mask
        )
        attention_output      = self.attention_dropout(attention_output)
        token_representations = self.layer_norm_after_attention(
            token_representations + attention_output
        )
        feedforward_output = self.feedforward_expand(token_representations)
        feedforward_output = self.feedforward_activation(feedforward_output)
        feedforward_output = self.feedforward_dropout(feedforward_output)
        feedforward_output = self.feedforward_compress(feedforward_output)
        token_representations = self.layer_norm_after_feedforward(
            token_representations + feedforward_output
        )
        return token_representations


# ---- Mini Language Model (identical to Project 7) ----

class MiniLanguageModel(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, number_of_attention_heads,
                 feedforward_hidden_dim, number_of_blocks, dropout_rate,
                 max_sequence_length):
        super(MiniLanguageModel, self).__init__()
        self.word_embedding   = nn.Embedding(vocabulary_size, embedding_dim)
        pos_enc = self._compute_positional_encoding(max_sequence_length, embedding_dim)
        self.register_buffer('positional_encoding', pos_enc)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, number_of_attention_heads,
                             feedforward_hidden_dim, dropout_rate)
            for _ in range(number_of_blocks)
        ])
        self.final_layer_norm  = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocabulary_size)
        self.embedding_dropout = nn.Dropout(dropout_rate)

    def _compute_positional_encoding(self, max_seq_len, embedding_dim):
        position  = torch.arange(max_seq_len).unsqueeze(1).float()
        dimension = torch.arange(embedding_dim).unsqueeze(0).float()
        angles    = position / torch.pow(10000, (2 * (dimension // 2)) / embedding_dim)
        enc             = angles.clone()
        enc[:, 0::2]    = torch.sin(angles[:, 0::2])
        enc[:, 1::2]    = torch.cos(angles[:, 1::2])
        return enc.unsqueeze(0)

    def _build_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

    def forward(self, token_indices):
        seq_len             = token_indices.shape[1]
        token_representations = self.word_embedding(token_indices)
        token_representations = token_representations + self.positional_encoding[:, :seq_len, :]
        token_representations = self.embedding_dropout(token_representations)
        causal_mask           = self._build_causal_mask(seq_len, token_indices.device)
        for block in self.transformer_blocks:
            token_representations = block(token_representations, causal_mask)
        token_representations = self.final_layer_norm(token_representations)
        return self.output_projection(token_representations[:, -1, :])


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
print(f"\nTotal parameters: {total_parameters:,}")
print(f"Project 7 had:    159,558 (word-level tokenisation)")


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

    # ---- Validation pass ----
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
    """
    Generate text using BPE tokenisation.
    Encodes seed text to token ids, generates new tokens,
    then uses bpe_tokenizer.decode() to convert back to clean text.
    The tokenizer handles Ġ (space) and Ċ (newline) automatically.
    """
    model.eval()
    encoded_seed  = bpe_tokenizer.encode(seed_text.lower())
    generated_ids = encoded_seed.ids.copy()

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

    # The tokenizer handles all decoding including spaces and newlines
    decoded = bpe_tokenizer.decode(generated_ids)

    # Replace newlines with spaces for clean single-line output
    return ' '.join(decoded.split())


print()
print("Generated text (temperature=0.8):")
print(" ", generate_text("the sky is cloudy"))
print(" ", generate_text("bring your umbrella"))
print(" ", generate_text("dark clouds mean"))
print(" ", generate_text("the rain will stop"))
print(" ", generate_text("a clear sky means"))

print(f"\nFinal loss: {training_loss_history[-1]:.4f}")


# ---- Plot loss curve ----

plt.figure(figsize=(10, 5))
plt.plot(training_loss_history, color='steelblue', linewidth=1.5,
         label=f'Training loss ({number_of_blocks} blocks, {total_parameters:,} params)')
plt.plot(val_loss_history,      color='tomato',    linewidth=1.5,
         label='Validation loss', linestyle='--')
plt.axhline(
    y=math.log(vocabulary_size),
    color='gray', linestyle=':', linewidth=1,
    label=f'Random baseline: {math.log(vocabulary_size):.2f}'
)
plt.title('Mini LLM with BPE Tokenisation: Training vs Validation Loss', fontsize=13)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Cross-Entropy Loss', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('loss_curve_bpe.png', dpi=150)
plt.show()

print("Loss curve saved to loss_curve_bpe.png")
