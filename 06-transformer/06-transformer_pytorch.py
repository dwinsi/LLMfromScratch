"""
Project 6: Transformer block in PyTorch.

Rebuilds the numpy Transformer block from scratch using PyTorch.
Now all 43,149 parameters train properly through autograd.

What PyTorch handles:
  nn.MultiheadAttention  - Q, K, V projections, scaled dot-product, causal masking
  nn.LayerNorm           - layer normalisation
  nn.Linear              - feed forward layers
  nn.Dropout             - dropout
  nn.Embedding           - word embedding lookup
  autograd               - full backward pass through all parameters

What stays visible:
  The residual connections: x = x + sublayer(x)
  The overall block structure: embed -> attention -> FFN -> output
  The training loop and generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ---- Device setup ----
# device = torch.device(
#     'cuda' if torch.cuda.is_available()  else
#     'mps'  if torch.backends.mps.is_available() else
#     'cpu'
# )
device = torch.device('cpu')
print(f"Using device: {device}")


# ---- Load and preprocess corpus ----

with open('weather_corpus.txt', 'r') as corpus_file:
    corpus_text = corpus_file.read().lower()

all_words       = corpus_text.split()
unique_words    = sorted(set(all_words))
vocabulary_size = len(unique_words)

word_to_index = {word: idx for idx, word in enumerate(unique_words)}
index_to_word = {idx: word for idx, word in enumerate(unique_words)}

print(f"Vocabulary size: {vocabulary_size}")


# ---- Build training sequences ----

sequence_length    = 3
training_sequences = []
training_targets   = []

for i in range(len(all_words) - sequence_length):
    input_sequence = all_words[i : i + sequence_length]
    target_word    = all_words[i + sequence_length]
    training_sequences.append([word_to_index[w] for w in input_sequence])
    training_targets.append(word_to_index[target_word])

training_sequences_tensor = [torch.tensor(seq).to(device) for seq in training_sequences]
training_targets_tensor   = [torch.tensor([tgt]).to(device) for tgt in training_targets]

print(f"Training sequences: {len(training_sequences)}")


# ---- Hyperparameters ----

embedding_dim                 = 64
number_of_attention_heads     = 4
feedforward_hidden_dim        = 128
dropout_rate                  = 0.1
learning_rate                 = 0.001
number_of_epochs              = 1000

assert embedding_dim % number_of_attention_heads == 0, \
    "embedding_dim must be divisible by number_of_attention_heads"


# ---- Transformer Block definition ----

class TransformerBlock(nn.Module):
    """
    One complete Transformer decoder block.

    Structure:
      1. Word embedding + positional encoding
      2. Masked multi-head self-attention
      3. Residual connection + layer normalisation
      4. Feed forward network
      5. Residual connection + layer normalisation
      6. Output projection to vocabulary
    """

    def __init__(self, vocabulary_size, embedding_dim, number_of_attention_heads,
                 feedforward_hidden_dim, dropout_rate, sequence_length):
        super(TransformerBlock, self).__init__()

        self.embedding_dim    = embedding_dim
        self.sequence_length  = sequence_length

        # Word embedding: index -> dense vector
        self.word_embedding = nn.Embedding(vocabulary_size, embedding_dim)

        # Positional encoding: fixed, not learned
        # Registered as a buffer so it moves to the right device automatically
        positional_encoding = self._compute_positional_encoding(sequence_length, embedding_dim)
        self.register_buffer('positional_encoding', positional_encoding)

        # Multi-head self-attention
        # batch_first=True means input shape is (batch, sequence, features)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=number_of_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # Layer normalisation after attention
        self.layer_norm_after_attention = nn.LayerNorm(embedding_dim)

        # Feed forward network: expand then compress
        self.feedforward_expand   = nn.Linear(embedding_dim, feedforward_hidden_dim)
        self.feedforward_compress = nn.Linear(feedforward_hidden_dim, embedding_dim)
        self.feedforward_relu     = nn.ReLU()
        self.feedforward_dropout  = nn.Dropout(dropout_rate)

        # Layer normalisation after feed forward
        self.layer_norm_after_feedforward = nn.LayerNorm(embedding_dim)

        # Output projection: from embedding space to vocabulary
        self.output_projection = nn.Linear(embedding_dim, vocabulary_size)

    def _compute_positional_encoding(self, sequence_length, embedding_dim):
        """
        Sine/cosine positional encoding.
        Returns a tensor of shape (1, sequence_length, embedding_dim).
        The leading 1 is the batch dimension for broadcasting.
        """
        position_indices  = torch.arange(sequence_length).unsqueeze(1).float()
        dimension_indices = torch.arange(embedding_dim).unsqueeze(0).float()

        frequency_scaling = position_indices / torch.pow(
            10000, (2 * (dimension_indices // 2)) / embedding_dim
        )

        positional_encoding             = frequency_scaling.clone()
        positional_encoding[:, 0::2]    = torch.sin(frequency_scaling[:, 0::2])
        positional_encoding[:, 1::2]    = torch.cos(frequency_scaling[:, 1::2])

        return positional_encoding.unsqueeze(0)   # (1, sequence_length, embedding_dim)

    def _build_causal_mask(self, sequence_length):
        """
        Causal mask prevents each position from attending to future positions.
        Returns an upper triangular matrix of -inf above the diagonal.
        nn.MultiheadAttention expects True where attention should be blocked.
        """
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=self.positional_encoding.device),
            diagonal=1
        ).bool()
        return causal_mask

    def forward(self, word_indices_in_sequence):
        """
        Run one complete Transformer block forward pass.
        word_indices_in_sequence: (batch, sequence_length)
        """
        batch_size   = word_indices_in_sequence.shape[0]
        seq_len      = word_indices_in_sequence.shape[1]

        # ---- Step 1: Embed words and add positional encoding ----
        word_embeddings       = self.word_embedding(word_indices_in_sequence)   # (batch, seq_len, 64)
        token_representations = word_embeddings + self.positional_encoding[:, :seq_len, :]

        # ---- Step 2: Masked multi-head self-attention ----
        causal_mask = self._build_causal_mask(seq_len)

        # nn.MultiheadAttention takes (query, key, value) as separate arguments
        # For self-attention, all three are the same input
        attention_output, attention_weights = self.multihead_attention(
            query=token_representations,
            key=token_representations,
            value=token_representations,
            attn_mask=causal_mask
        )

        # ---- Step 3: Residual connection + layer normalisation ----
        token_representations = self.layer_norm_after_attention(
            token_representations + attention_output
        )

        # ---- Step 4: Feed forward network ----
        # Each token processed independently: expand to 128, ReLU, compress to 64
        feedforward_output = self.feedforward_expand(token_representations)
        feedforward_output = self.feedforward_relu(feedforward_output)
        feedforward_output = self.feedforward_dropout(feedforward_output)
        feedforward_output = self.feedforward_compress(feedforward_output)

        # ---- Step 5: Residual connection + layer normalisation ----
        token_representations = self.layer_norm_after_feedforward(
            token_representations + feedforward_output
        )

        # ---- Step 6: Project last token to vocabulary probabilities ----
        # Only the last token predicts the next word
        last_token_representation = token_representations[:, -1, :]          # (batch, 64)
        output_scores             = self.output_projection(last_token_representation)  # (batch, 77)

        return output_scores, attention_weights


# ---- Initialise model, loss and optimiser ----

model = TransformerBlock(
    vocabulary_size=vocabulary_size,
    embedding_dim=embedding_dim,
    number_of_attention_heads=number_of_attention_heads,
    feedforward_hidden_dim=feedforward_hidden_dim,
    dropout_rate=dropout_rate,
    sequence_length=sequence_length
).to(device)

loss_function = nn.CrossEntropyLoss()
optimiser     = optim.Adam(model.parameters(), lr=learning_rate)

total_parameters = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_parameters:,}")


# ---- Training loop ----

training_loss_history = []

for epoch in range(number_of_epochs):
    model.train()
    total_epoch_loss = 0

    for sequence_tensor, target_tensor in zip(
        training_sequences_tensor, training_targets_tensor
    ):
        sequence_input = sequence_tensor.unsqueeze(0)   # (1, seq_len)
        target_input   = target_tensor                   # (1,)

        optimiser.zero_grad()
        output_scores, _ = model(sequence_input)
        loss             = loss_function(output_scores, target_input)
        loss.backward()
        optimiser.step()

        total_epoch_loss += loss.item()

    average_epoch_loss = total_epoch_loss / len(training_sequences_tensor)
    training_loss_history.append(average_epoch_loss)

    if epoch % 200 == 0:
        print(f"Epoch {epoch:5d}  loss: {average_epoch_loss:.4f}")


# ---- Text generation ----

def generate_text(seed_words, number_of_words_to_generate=6):
    """Generate new words given a seed sequence."""
    model.eval()
    generated_words = seed_words.copy()

    with torch.no_grad():
        for _ in range(number_of_words_to_generate):
            context_words   = generated_words[-sequence_length:]
            context_indices = [word_to_index.get(w, 0) for w in context_words]
            sequence_tensor = torch.tensor(context_indices).unsqueeze(0).to(device)

            output_scores, _ = model(sequence_tensor)
            predicted_index  = torch.argmax(output_scores, dim=-1).item()
            generated_words.append(index_to_word[predicted_index])

    return ' '.join(generated_words)


print()
print("Generated text:")
print(" ", generate_text(['the', 'sky', 'is']))
print(" ", generate_text(['bring', 'your', 'umbrella']))
print(" ", generate_text(['dark', 'clouds', 'mean']))
print(" ", generate_text(['the', 'rain', 'will']))
print(" ", generate_text(['a', 'clear', 'sky']))


# ---- Plot loss curve ----

plt.figure(figsize=(10, 5))
plt.plot(training_loss_history, color='steelblue', linewidth=1.5,
         label='Transformer block (PyTorch)')
plt.axhline(
    y=torch.log(torch.tensor(vocabulary_size, dtype=torch.float)).item(),
    color='tomato', linestyle='--', linewidth=1,
    label=f'Random baseline (log {vocabulary_size} = {torch.log(torch.tensor(vocabulary_size, dtype=torch.float)).item():.2f})'
)
plt.title('Transformer Block Training Loss (PyTorch)', fontsize=13)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Cross-Entropy Loss', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('loss_curve_pytorch.png', dpi=150)
plt.show()

print("\nLoss curve saved to loss_curve_pytorch.png")