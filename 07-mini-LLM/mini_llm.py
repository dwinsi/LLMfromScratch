"""
Project 7: A Mini Language Model.

Stacks four Transformer blocks into a complete language model.
Trained on an expanded weather corpus (91 sentences, 198 vocabulary).

This is the capstone of the build series. The architecture is identical
to GPT at a much smaller scale. The same ideas, the same structure,
the same training loop. Just fewer parameters and a smaller dataset.

Architecture:
  Word embedding + positional encoding
  4 × Transformer blocks (multi-head attention + FFN + residual + layer norm)
  Output projection to vocabulary

New in this project:
  Stacking multiple blocks
  Larger corpus and vocabulary
  Sequence length of 4 (up from 3)
  Temperature-based text generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math

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

with open('weather_corpus_v2.txt', 'r') as corpus_file:
    corpus_text = corpus_file.read().lower()

all_words       = corpus_text.split()
unique_words    = sorted(set(all_words))
vocabulary_size = len(unique_words)

word_to_index = {word: idx for idx, word in enumerate(unique_words)}
index_to_word = {idx: word for idx, word in enumerate(unique_words)}

print(f"Vocabulary size:    {vocabulary_size}")
print(f"Total words:        {len(all_words)}")

# ---- Build training sequences ----
# Sequence length 4: four words predict the fifth

sequence_length    = 4
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


# ---- Single Transformer block ----

class TransformerBlock(nn.Module):
    """
    One complete Transformer decoder block.
    Identical to Project 6 but now used as a repeating unit.
    """

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
        self.feedforward_activation       = nn.ReLU()
        self.feedforward_dropout          = nn.Dropout(dropout_rate)
        self.layer_norm_after_feedforward = nn.LayerNorm(embedding_dim)
        self.attention_dropout            = nn.Dropout(dropout_rate)

    def forward(self, token_representations, causal_mask):

        # Multi-head self-attention with causal mask
        attention_output, _ = self.multihead_attention(
            query=token_representations,
            key=token_representations,
            value=token_representations,
            attn_mask=causal_mask
        )
        attention_output = self.attention_dropout(attention_output)

        # Residual connection + layer normalisation
        token_representations = self.layer_norm_after_attention(
            token_representations + attention_output
        )

        # Feed forward network: expand -> activate -> compress
        feedforward_output = self.feedforward_expand(token_representations)
        feedforward_output = self.feedforward_activation(feedforward_output)
        feedforward_output = self.feedforward_dropout(feedforward_output)
        feedforward_output = self.feedforward_compress(feedforward_output)

        # Residual connection + layer normalisation
        token_representations = self.layer_norm_after_feedforward(
            token_representations + feedforward_output
        )

        return token_representations


# ---- Mini Language Model ----

class MiniLanguageModel(nn.Module):
    """
    A mini GPT-style language model.

    Structure:
      1. Word embedding + sinusoidal positional encoding
      2. N stacked Transformer blocks
      3. Final layer normalisation
      4. Output projection to vocabulary
    """

    def __init__(self, vocabulary_size, embedding_dim, number_of_attention_heads,
                 feedforward_hidden_dim, number_of_blocks, dropout_rate, max_sequence_length):
        super(MiniLanguageModel, self).__init__()

        self.embedding_dim  = embedding_dim
        self.number_of_blocks = number_of_blocks

        # Word embedding
        self.word_embedding = nn.Embedding(vocabulary_size, embedding_dim)

        # Positional encoding (fixed, not learned)
        positional_encoding = self._compute_positional_encoding(
            max_sequence_length, embedding_dim
        )
        self.register_buffer('positional_encoding', positional_encoding)

        # Stack of Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                number_of_attention_heads=number_of_attention_heads,
                feedforward_hidden_dim=feedforward_hidden_dim,
                dropout_rate=dropout_rate
            )
            for _ in range(number_of_blocks)
        ])

        # Final layer normalisation
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        # Output projection: from embedding space to vocabulary probabilities
        self.output_projection = nn.Linear(embedding_dim, vocabulary_size)

        # Embedding dropout
        self.embedding_dropout = nn.Dropout(dropout_rate)

    def _compute_positional_encoding(self, max_sequence_length, embedding_dim):
        """Sinusoidal positional encoding, shape (1, max_seq_len, embedding_dim)."""
        position_indices  = torch.arange(max_sequence_length).unsqueeze(1).float()
        dimension_indices = torch.arange(embedding_dim).unsqueeze(0).float()

        frequency_scaling = position_indices / torch.pow(
            10000, (2 * (dimension_indices // 2)) / embedding_dim
        )

        positional_encoding             = frequency_scaling.clone()
        positional_encoding[:, 0::2]    = torch.sin(frequency_scaling[:, 0::2])
        positional_encoding[:, 1::2]    = torch.cos(frequency_scaling[:, 1::2])

        return positional_encoding.unsqueeze(0)   # (1, max_seq_len, embedding_dim)

    def _build_causal_mask(self, sequence_length):
        """Upper triangular boolean mask blocking future positions."""
        return torch.triu(
            torch.ones(sequence_length, sequence_length,
                       device=self.positional_encoding.device),
            diagonal=1
        ).bool()

    def forward(self, word_indices_in_sequence):
        """
        Forward pass through the full language model.
        word_indices_in_sequence: (batch, sequence_length)
        Returns output scores: (batch, vocabulary_size)
        """
        batch_size = word_indices_in_sequence.shape[0]
        seq_len    = word_indices_in_sequence.shape[1]

        # Embed words and add positional encoding
        word_embeddings       = self.word_embedding(word_indices_in_sequence)
        token_representations = word_embeddings + self.positional_encoding[:, :seq_len, :]
        token_representations = self.embedding_dropout(token_representations)

        # Build causal mask once and pass through all blocks
        causal_mask = self._build_causal_mask(seq_len)

        # Pass through all Transformer blocks sequentially
        for transformer_block in self.transformer_blocks:
            token_representations = transformer_block(token_representations, causal_mask)

        # Final layer normalisation
        token_representations = self.final_layer_norm(token_representations)

        # Use last token to predict next word
        last_token_representation = token_representations[:, -1, :]
        output_scores             = self.output_projection(last_token_representation)

        return output_scores


# ---- Initialise model ----

embedding_dim             = 64
number_of_attention_heads = 4
feedforward_hidden_dim    = 128
number_of_blocks          = 4
dropout_rate              = 0.1
learning_rate             = 0.001
number_of_epochs          = 1000

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
        sequence_input = sequence_tensor.unsqueeze(0)
        target_input   = target_tensor

        optimiser.zero_grad()
        output_scores = model(sequence_input)
        loss          = loss_function(output_scores, target_input)
        loss.backward()

        # Gradient clipping: prevents exploding gradients in deep networks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()
        total_epoch_loss += loss.item()

    average_epoch_loss = total_epoch_loss / len(training_sequences_tensor)
    training_loss_history.append(average_epoch_loss)

    if epoch % 200 == 0:
        print(f"Epoch {epoch:5d}  loss: {average_epoch_loss:.4f}")


# ---- Text generation with temperature ----

def generate_text(seed_words, number_of_words_to_generate=8, temperature=1.0):
    """
    Generate new words given a seed sequence.

    temperature controls randomness:
      temperature = 1.0  standard sampling
      temperature < 1.0  more conservative, sticks to likely words
      temperature > 1.0  more creative, explores less likely words
      temperature = 0.0  always picks the most likely word (greedy)
    """
    model.eval()
    generated_words = seed_words.copy()

    with torch.no_grad():
        for _ in range(number_of_words_to_generate):
            context_words   = generated_words[-sequence_length:]
            context_indices = [word_to_index.get(w, 0) for w in context_words]
            sequence_tensor = torch.tensor(context_indices).unsqueeze(0).to(device)

            output_scores = model(sequence_tensor)

            if temperature == 0.0:
                predicted_index = torch.argmax(output_scores, dim=-1).item()
            else:
                scaled_scores   = output_scores / temperature
                probabilities   = torch.softmax(scaled_scores, dim=-1)
                predicted_index = torch.multinomial(probabilities, num_samples=1).item()

            generated_words.append(index_to_word[predicted_index])

    return ' '.join(generated_words)


print()
print("Generated text (greedy, temperature=0.0):")
print(" ", generate_text(['the', 'sky', 'is', 'cloudy'], temperature=0.0))
print(" ", generate_text(['bring', 'your', 'umbrella', 'when'], temperature=0.0))
print(" ", generate_text(['dark', 'clouds', 'mean', 'heavy'], temperature=0.0))
print(" ", generate_text(['the', 'rain', 'will', 'stop'], temperature=0.0))
print(" ", generate_text(['a', 'clear', 'sky', 'means'], temperature=0.0))

print()
print("Generated text (temperature=0.8, slightly creative):")
print(" ", generate_text(['the', 'sky', 'is', 'cloudy'], temperature=0.8))
print(" ", generate_text(['bring', 'your', 'umbrella', 'when'], temperature=0.8))
print(" ", generate_text(['the', 'storm', 'is', 'moving'], temperature=0.8))


# ---- Plot loss curve ----

plt.figure(figsize=(10, 5))
plt.plot(training_loss_history, color='steelblue', linewidth=1.5,
         label=f'Mini LLM ({number_of_blocks} blocks, {total_parameters:,} params)')
plt.axhline(
    y=math.log(vocabulary_size),
    color='tomato', linestyle='--', linewidth=1,
    label=f'Random baseline: {math.log(vocabulary_size):.2f}'
)
plt.title('Mini Language Model Training Loss', fontsize=13)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Cross-Entropy Loss', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.show()

print(f"\nFinal loss: {training_loss_history[-1]:.4f}")
print("Loss curve saved to loss_curve.png")