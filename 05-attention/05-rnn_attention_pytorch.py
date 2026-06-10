"""
Project 5: RNN with Attention in PyTorch.

Rebuilds the numpy RNN with attention from Project 4 using PyTorch.
nn.RNN handles the recurrent loop. Attention stays hand-rolled so
the Q, K, V math remains visible.

Automatically uses GPU if available (CUDA on NVIDIA, MPS on Apple Silicon),
otherwise falls back to CPU.

New concepts:
  nn.Embedding        - learned word lookup table replacing one-hot encoding
  nn.RNN              - replaces the manual hidden state loop
  nn.CrossEntropyLoss - combines softmax and negative log likelihood
  Adam optimiser      - adaptive learning rate, faster than SGD
  torch.no_grad()     - disables gradient tracking during generation
  device management   - automatic GPU/CPU selection with .to(device)

This file also includes a side by side comparison of Adam vs SGD
using the same network and data to show the difference in convergence.
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
    sequence = all_words[i : i + sequence_length]
    target   = all_words[i + sequence_length]
    training_sequences.append([word_to_index[w] for w in sequence])
    training_targets.append(word_to_index[target])

training_sequences_tensor = [torch.tensor(seq).to(device) for seq in training_sequences]
training_targets_tensor   = [torch.tensor([tgt]).to(device) for tgt in training_targets]

print(f"Training sequences: {len(training_sequences)}")


# ---- Network definition ----

class RNNWithAttention(nn.Module):

    def __init__(self, vocabulary_size, hidden_size, attention_size):
        super(RNNWithAttention, self).__init__()

        self.hidden_size    = hidden_size
        self.attention_size = attention_size

        # Embedding layer: word index -> dense vector
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)

        # RNN layer: processes the sequence, returns all hidden states
        # batch_first=True means input shape is (batch, sequence, features)
        self.rnn = nn.RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Attention weight matrices: hand-rolled so the math stays visible
        self.weights_query = nn.Linear(hidden_size, attention_size, bias=False)
        self.weights_key   = nn.Linear(hidden_size, attention_size, bias=False)
        self.weights_value = nn.Linear(hidden_size, hidden_size,    bias=False)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, input_sequence):

        # Embed the input words: index -> dense vector
        embedded_input = self.embedding(input_sequence)      # (batch, seq_len, hidden_size)

        # RNN forward pass: get all hidden states at once
        rnn_output, _ = self.rnn(embedded_input)             # (batch, seq_len, hidden_size)

        # Attention: project each hidden state into Q, K, V spaces
        query_vectors = self.weights_query(rnn_output)       # (batch, seq_len, attention_size)
        key_vectors   = self.weights_key(rnn_output)         # (batch, seq_len, attention_size)
        value_vectors = self.weights_value(rnn_output)       # (batch, seq_len, hidden_size)

        # Use last hidden state query to attend over all keys
        last_query = query_vectors[:, -1:, :]                # (batch, 1, attention_size)

        # Scaled dot-product attention
        attention_scores  = torch.bmm(last_query, key_vectors.transpose(1, 2))
        attention_scores  = attention_scores / (self.attention_size ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Context vector: weighted sum of values
        context_vector = torch.bmm(attention_weights, value_vectors)   # (batch, 1, hidden_size)
        context_vector = context_vector.squeeze(1)                      # (batch, hidden_size)

        # Output layer
        output_scores = self.output_layer(context_vector)               # (batch, vocabulary_size)

        return output_scores, attention_weights.squeeze(1)


# ---- Reusable training function ----

def train_model(optimiser_name, number_of_epochs=1000):
    """
    Train a fresh RNNWithAttention model using the specified optimiser.
    Returns the loss history so both runs can be compared on the same plot.
    Both runs use the same random seed for a fair comparison.
    """
    torch.manual_seed(42)

    hidden_size    = 64
    attention_size = 32

    model         = RNNWithAttention(vocabulary_size, hidden_size, attention_size).to(device)
    loss_function = nn.CrossEntropyLoss()

    if optimiser_name == 'Adam':
        optimiser = optim.Adam(model.parameters(), lr=0.001)
    elif optimiser_name == 'SGD':
        optimiser = optim.SGD(model.parameters(), lr=0.01)

    loss_history = []

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
            loss              = loss_function(output_scores, target_input)
            loss.backward()
            optimiser.step()

            total_epoch_loss += loss.item()

        average_epoch_loss = total_epoch_loss / len(training_sequences_tensor)
        loss_history.append(average_epoch_loss)

        if epoch % 200 == 0:
            print(f"  [{optimiser_name}] Epoch {epoch:5d}  loss: {average_epoch_loss:.4f}")

    return model, loss_history


# ---- Run both experiments ----

total_parameters = sum(
    p.numel() for p in RNNWithAttention(vocabulary_size, 64, 32).parameters()
)
print(f"Total parameters: {total_parameters:,}")
print()

print("Training with Adam...")
model_adam, adam_loss_history = train_model('Adam', number_of_epochs=1000)

print()
print("Training with SGD...")
model_sgd,  sgd_loss_history  = train_model('SGD',  number_of_epochs=1000)

print()
print(f"Final loss  Adam: {adam_loss_history[-1]:.4f}")
print(f"Final loss   SGD: {sgd_loss_history[-1]:.4f}")


# ---- Text generation using Adam model ----

def generate_text(model, seed_words, number_of_words_to_generate=6):
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
print("Generated text (Adam model):")
print(" ", generate_text(model_adam, ['the', 'sky', 'is']))
print(" ", generate_text(model_adam, ['bring', 'your', 'umbrella']))
print(" ", generate_text(model_adam, ['dark', 'clouds', 'mean']))
print(" ", generate_text(model_adam, ['the', 'rain', 'will']))
print(" ", generate_text(model_adam, ['a', 'clear', 'sky']))

print()
print("Generated text (SGD model):")
print(" ", generate_text(model_sgd, ['the', 'sky', 'is']))
print(" ", generate_text(model_sgd, ['bring', 'your', 'umbrella']))
print(" ", generate_text(model_sgd, ['dark', 'clouds', 'mean']))


# ---- Plot Adam vs SGD comparison ----

plt.figure(figsize=(12, 5))

plt.plot(adam_loss_history, color='steelblue', linewidth=1.5, label=f'Adam  (final loss: {adam_loss_history[-1]:.4f})')
plt.plot(sgd_loss_history,  color='tomato',    linewidth=1.5, label=f'SGD   (final loss: {sgd_loss_history[-1]:.4f})',  linestyle='--')

plt.axhline(
    y=torch.log(torch.tensor(vocabulary_size, dtype=torch.float)).item(),
    color='gray', linestyle=':', linewidth=1,
    label=f'Random baseline: {torch.log(torch.tensor(vocabulary_size, dtype=torch.float)).item():.2f}'
)

plt.title('Adam vs SGD: RNN with Attention on Weather Corpus', fontsize=13)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Cross-Entropy Loss', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('sgd_vs_adam.png', dpi=150)
plt.show()

print("\nComparison plot saved to sgd_vs_adam.png")