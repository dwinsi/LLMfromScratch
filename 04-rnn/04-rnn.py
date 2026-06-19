"""
Project 4: A word-level RNN from scratch.

Moves beyond classification into sequence prediction.
Reads a sequence of words and predicts the next word.
Trained on a custom weather corpus written for this series.

Architecture: word embeddings -> hidden state (tanh) -> output (softmax)
New concepts: vocabulary building, sequence prediction, cross-entropy loss
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt

_cfg = json.loads((pathlib.Path(__file__).parent / "config.json").read_text())
_model = _cfg["model"]
_train = _cfg["training"]

np.random.seed(_train["seed"])


# ---- Helper functions ----

def softmax(scores):
    """
    Convert raw output scores into a probability distribution.
    Subtracting the max prevents numerical overflow.
    """
    shifted_scores    = scores - np.max(scores)
    exponentiated     = np.exp(shifted_scores)
    return exponentiated / np.sum(exponentiated)


def one_hot_encode(word_index, vocabulary_size):
    """
    Convert a word index into a one-hot vector.
    All zeros except a single 1 at the word's position.
    """
    one_hot_vector                    = np.zeros((1, vocabulary_size))
    one_hot_vector[0, word_index]     = 1
    return one_hot_vector


# ---- Load and preprocess the corpus ----

with open('weather_corpus.txt', 'r') as corpus_file:
    corpus_text = corpus_file.read().lower()

all_words       = corpus_text.split()
unique_words    = sorted(set(all_words))
vocabulary_size = len(unique_words)

word_to_index = {word: idx for idx, word in enumerate(unique_words)}
index_to_word = {idx: word for idx, word in enumerate(unique_words)}

print(f"Total words in corpus:     {len(all_words)}")
print(f"Unique words (vocabulary): {vocabulary_size}")


# ---- Build training sequences ----
# Each sequence of 3 words predicts the next word

sequence_length    = _model["sequence_length"]
training_sequences = []
training_targets   = []

for i in range(len(all_words) - sequence_length):
    sequence = all_words[i : i + sequence_length]
    target   = all_words[i + sequence_length]
    training_sequences.append([word_to_index[w] for w in sequence])
    training_targets.append(word_to_index[target])

split              = int(0.8 * len(training_sequences))
val_sequences      = training_sequences[split:]
val_targets        = training_targets[split:]
training_sequences = training_sequences[:split]
training_targets   = training_targets[:split]

print(f"Training sequences:        {len(training_sequences)}")
print(f"Validation sequences:      {len(val_sequences)}")


# ---- Initialise weights ----

hidden_size   = _model["hidden_size"]
learning_rate = _train["learning_rate"]
epochs        = _train["epochs"]

weights_input_to_hidden  = np.random.randn(vocabulary_size, hidden_size) * 0.01
weights_hidden_to_hidden = np.random.randn(hidden_size, hidden_size)     * 0.01
weights_hidden_to_output = np.random.randn(hidden_size, vocabulary_size) * 0.01
bias_hidden              = np.zeros((1, hidden_size))
bias_output              = np.zeros((1, vocabulary_size))

loss_history     = []
val_loss_history = []


# ---- Training loop ----

for epoch in range(epochs):
    total_loss = 0

    for sequence_indices, target_index in zip(training_sequences, training_targets):

        # Forward pass: read each word and update hidden state
        hidden_state = np.zeros((1, hidden_size))

        for word_index in sequence_indices:
            word_vector  = one_hot_encode(word_index, vocabulary_size)
            hidden_input = np.dot(word_vector,  weights_input_to_hidden)  + \
                           np.dot(hidden_state, weights_hidden_to_hidden) + \
                           bias_hidden
            hidden_state = np.tanh(hidden_input)

        # Output layer: score every word in vocabulary
        output_scores        = np.dot(hidden_state, weights_hidden_to_output) + bias_output
        output_probabilities = softmax(output_scores[0])

        # Cross-entropy loss: how wrong is the prediction for the correct word
        correct_word_probability = output_probabilities[target_index]
        loss                     = -np.log(correct_word_probability + 1e-8)
        total_loss              += loss

        # Backward pass: gradients for output layer
        output_gradient                  = output_probabilities.copy()
        output_gradient[target_index]   -= 1
        output_gradient                  = output_gradient.reshape(1, -1)

        weights_hidden_to_output -= learning_rate * np.dot(hidden_state.T, output_gradient)
        bias_output              -= learning_rate * output_gradient

        # Backward pass: gradients for hidden layer
        hidden_gradient  = np.dot(output_gradient, weights_hidden_to_output.T)
        hidden_gradient *= (1 - hidden_state ** 2)   # tanh derivative

        weights_hidden_to_hidden -= learning_rate * np.dot(hidden_state.T, hidden_gradient)
        weights_input_to_hidden  -= learning_rate * np.dot(
            one_hot_encode(sequence_indices[-1], vocabulary_size).T, hidden_gradient
        )
        bias_hidden -= learning_rate * hidden_gradient

    average_loss = total_loss / len(training_sequences)
    loss_history.append(average_loss)

    # ---- Validation pass (no weight updates) ----
    val_total_loss = 0
    for sequence_indices, target_index in zip(val_sequences, val_targets):
        hidden_state = np.zeros((1, hidden_size))
        for word_index in sequence_indices:
            word_vector  = one_hot_encode(word_index, vocabulary_size)
            hidden_input = (np.dot(word_vector, weights_input_to_hidden) +
                            np.dot(hidden_state, weights_hidden_to_hidden) + bias_hidden)
            hidden_state = np.tanh(hidden_input)
        output_scores        = np.dot(hidden_state, weights_hidden_to_output) + bias_output
        output_probabilities = softmax(output_scores[0])
        val_total_loss      += -np.log(output_probabilities[target_index] + 1e-8)
    val_loss_history.append(val_total_loss / len(val_sequences))

    if epoch % 200 == 0:
        print(f"Epoch {epoch:5d}  train loss: {average_loss:.4f}  val loss: {val_loss_history[-1]:.4f}")


# ---- Text generation ----

def generate_text(seed_words, number_of_words_to_generate=6):
    """
    Generate new words given a seed sequence.
    Repeatedly predicts the next word and adds it to the sequence.
    """
    generated_words = seed_words.copy()

    for _ in range(number_of_words_to_generate):
        context_words = generated_words[-sequence_length:]
        hidden_state  = np.zeros((1, hidden_size))

        for word in context_words:
            if word not in word_to_index:
                break
            word_vector  = one_hot_encode(word_to_index[word], vocabulary_size)
            hidden_input = np.dot(word_vector,  weights_input_to_hidden)  + \
                           np.dot(hidden_state, weights_hidden_to_hidden) + \
                           bias_hidden
            hidden_state = np.tanh(hidden_input)

        output_scores        = np.dot(hidden_state, weights_hidden_to_output) + bias_output
        output_probabilities = softmax(output_scores[0])
        predicted_word_index = np.argmax(output_probabilities)
        predicted_word       = index_to_word[predicted_word_index]
        generated_words.append(predicted_word)

    return ' '.join(generated_words)


print()
print("Generated text samples:")
print(" ", generate_text(['the', 'sky', 'is']))
print(" ", generate_text(['bring', 'your', 'umbrella']))
print(" ", generate_text(['dark', 'clouds', 'mean']))
print(" ", generate_text(['the', 'rain', 'will']))
print(" ", generate_text(['a', 'clear', 'sky']))


# ---- Plot the loss curve ----

plt.figure(figsize=(10, 5))
plt.plot(loss_history,     color='steelblue', linewidth=1.5, label='Training loss')
plt.plot(val_loss_history, color='tomato',    linewidth=1.5, label='Validation loss', linestyle='--')
plt.axhline(
    y=np.log(vocabulary_size),
    color='gray', linestyle=':', linewidth=1,
    label=f'Random baseline (log {vocabulary_size} = {np.log(vocabulary_size):.2f})'
)
plt.title('RNN Training vs Validation Loss on Weather Corpus', fontsize=13)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Cross-Entropy Loss', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.show()

print("\nLoss curve saved to loss_curve.png")