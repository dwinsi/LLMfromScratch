"""
Project 6: A Transformer block from scratch.

Builds one complete Transformer decoder block using numpy.
The full architecture that powers every modern LLM, one block at a time.

Components:
  - Positional encoding (sine/cosine)
  - Multi-head self-attention with causal masking
  - Residual connections
  - Layer normalisation
  - Feed forward network with ReLU
  - Dropout

Project 7 will stack multiple blocks of this kind into a mini LLM.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(42)


# ---- Helper functions ----

def softmax_along_last_axis(scores):
    """
    Convert raw scores into probabilities along the last axis.
    Subtracting the max prevents numerical overflow.
    """
    scores_shifted    = scores - np.max(scores, axis=-1, keepdims=True)
    scores_exp        = np.exp(scores_shifted)
    return scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)


def apply_layer_normalisation(token_representations, learned_scale, learned_shift, epsilon=1e-6):
    """
    Normalise each token's representation across the embedding dimension.
    Keeps values in a stable range regardless of how deep the network is.
    learned_scale (gamma) and learned_shift (beta) are trained parameters.
    """
    token_mean          = np.mean(token_representations, axis=-1, keepdims=True)
    token_variance      = np.var(token_representations,  axis=-1, keepdims=True)
    normalised_tokens   = (token_representations - token_mean) / np.sqrt(token_variance + epsilon)
    return learned_scale * normalised_tokens + learned_shift


def relu_activation(x):
    """Rectified linear unit. Sets all negative values to zero."""
    return np.maximum(0, x)


def apply_dropout(activations, dropout_rate, is_training=True):
    """
    Randomly zero a fraction of activations during training.
    Remaining values are scaled up to preserve expected sum.
    Disabled during generation (is_training=False).
    """
    if not is_training or dropout_rate == 0:
        return activations
    keep_probability  = 1 - dropout_rate
    dropout_mask      = np.random.binomial(1, keep_probability, activations.shape) / keep_probability
    return activations * dropout_mask


def compute_sinusoidal_positional_encoding(sequence_length, embedding_dim):
    """
    Compute sine/cosine positional encodings.
    Each position in the sequence gets a unique vector.
    Even dimensions use sine, odd dimensions use cosine.
    The varying frequencies create a unique fingerprint for every position.
    """
    position_indices  = np.arange(sequence_length)[:, np.newaxis]    # (seq_len, 1)
    dimension_indices = np.arange(embedding_dim)[np.newaxis, :]      # (1, embedding_dim)

    frequency_scaling = position_indices / np.power(
        10000, (2 * (dimension_indices // 2)) / embedding_dim
    )

    positional_encoding                    = frequency_scaling.copy()
    positional_encoding[:, 0::2]           = np.sin(frequency_scaling[:, 0::2])
    positional_encoding[:, 1::2]           = np.cos(frequency_scaling[:, 1::2])

    return positional_encoding                                        # (seq_len, embedding_dim)


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

print(f"Training sequences: {len(training_sequences)}")


# ---- Hyperparameters ----

embedding_dim                   = 64
number_of_attention_heads       = 4
dimensions_per_attention_head   = embedding_dim // number_of_attention_heads  # 16
feedforward_hidden_dim          = 128
dropout_rate                    = 0.1
learning_rate                   = 0.001
number_of_epochs                = 500

assert embedding_dim % number_of_attention_heads == 0, \
    "embedding_dim must be divisible by number_of_attention_heads"

print(f"Embedding dim: {embedding_dim}")
print(f"Attention heads: {number_of_attention_heads}")
print(f"Dimensions per head: {dimensions_per_attention_head}")


# ---- Initialise learnable weights ----

# Embedding: converts word index into a dense vector
word_embedding_matrix = np.random.randn(vocabulary_size, embedding_dim) * 0.01

# Multi-head attention: four weight matrices
# All heads share these matrices and are split after projection
attention_query_weights  = np.random.randn(embedding_dim, embedding_dim) * 0.01
attention_key_weights    = np.random.randn(embedding_dim, embedding_dim) * 0.01
attention_value_weights  = np.random.randn(embedding_dim, embedding_dim) * 0.01
attention_output_weights = np.random.randn(embedding_dim, embedding_dim) * 0.01

# Layer normalisation after attention sublayer
layer_norm1_scale = np.ones(embedding_dim)
layer_norm1_shift = np.zeros(embedding_dim)

# Feed forward network: expand then compress
feedforward_expand_weights   = np.random.randn(embedding_dim, feedforward_hidden_dim) * 0.01
feedforward_expand_bias      = np.zeros(feedforward_hidden_dim)
feedforward_compress_weights = np.random.randn(feedforward_hidden_dim, embedding_dim) * 0.01
feedforward_compress_bias    = np.zeros(embedding_dim)

# Layer normalisation after feed forward sublayer
layer_norm2_scale = np.ones(embedding_dim)
layer_norm2_shift = np.zeros(embedding_dim)

# Output projection: from embedding space to vocabulary probabilities
output_projection_weights = np.random.randn(embedding_dim, vocabulary_size) * 0.01
output_projection_bias    = np.zeros(vocabulary_size)

# Precompute positional encodings once (same for every forward pass)
precomputed_positional_encoding = compute_sinusoidal_positional_encoding(
    sequence_length, embedding_dim
)


# ---- Parameter count ----

total_parameters = (
    vocabulary_size * embedding_dim +                           # word embedding matrix
    4 * embedding_dim * embedding_dim +                         # Q, K, V, output attention
    2 * (embedding_dim * feedforward_hidden_dim) +              # FFN expand and compress weights
    feedforward_hidden_dim + embedding_dim +                    # FFN biases
    4 * embedding_dim +                                         # layer norm scales and shifts
    embedding_dim * vocabulary_size + vocabulary_size           # output projection
)
print(f"Total parameters: {total_parameters:,}")


# ---- Forward pass through one Transformer block ----

def transformer_block_forward(word_indices_in_sequence, is_training=True):
    """
    Run one complete Transformer block forward pass.

    Steps:
      1. Embed words and add positional encoding
      2. Multi-head self-attention with causal masking
      3. Residual connection + layer normalisation
      4. Feed forward network
      5. Residual connection + layer normalisation
      6. Project last token to vocabulary probabilities

    Returns output probabilities and attention weights from all heads.
    """
    number_of_tokens = len(word_indices_in_sequence)

    # ---- Step 1: Embed words and add positional encoding ----
    word_embeddings            = word_embedding_matrix[word_indices_in_sequence]  # (tokens, 64)
    token_representations      = word_embeddings + precomputed_positional_encoding[:number_of_tokens]

    # ---- Step 2: Multi-head self-attention ----

    # Project all token representations to query, key, value spaces
    all_queries = np.dot(token_representations, attention_query_weights)  # (tokens, 64)
    all_keys    = np.dot(token_representations, attention_key_weights)
    all_values  = np.dot(token_representations, attention_value_weights)

    # Split into heads: each head gets its own slice of the embedding dimensions
    # Shape becomes (tokens, number_of_heads, dimensions_per_head)
    queries_per_head = all_queries.reshape(number_of_tokens, number_of_attention_heads, dimensions_per_attention_head)
    keys_per_head    = all_keys.reshape(   number_of_tokens, number_of_attention_heads, dimensions_per_attention_head)
    values_per_head  = all_values.reshape( number_of_tokens, number_of_attention_heads, dimensions_per_attention_head)

    attention_head_outputs       = []
    attention_weights_per_head   = []

    for head_index in range(number_of_attention_heads):

        # Extract this head's query, key, value matrices
        head_queries = queries_per_head[:, head_index, :]   # (tokens, 16)
        head_keys    = keys_per_head[   :, head_index, :]
        head_values  = values_per_head[ :, head_index, :]

        # Compute attention scores: how much should each token attend to each other token
        raw_attention_scores = np.dot(head_queries, head_keys.T)             # (tokens, tokens)
        scaled_scores        = raw_attention_scores / math.sqrt(dimensions_per_attention_head)

        # Apply causal mask: prevent attending to future positions
        # np.triu with k=1 fills the upper triangle (future positions) with -infinity
        # After softmax these become zero: the model cannot see future words
        causal_mask          = np.triu(np.full_like(scaled_scores, -1e9), k=1)
        masked_scores        = scaled_scores + causal_mask

        # Convert scores to attention weights (probabilities)
        attention_weights    = softmax_along_last_axis(masked_scores)        # (tokens, tokens)
        attention_weights    = apply_dropout(attention_weights, dropout_rate, is_training)

        # Weighted sum of values: each token collects information from others
        head_output          = np.dot(attention_weights, head_values)        # (tokens, 16)

        attention_head_outputs.append(head_output)
        attention_weights_per_head.append(attention_weights)

    # Concatenate all head outputs back to full embedding dimension
    concatenated_head_outputs = np.concatenate(attention_head_outputs, axis=-1)  # (tokens, 64)

    # Final output projection
    attention_sublayer_output = np.dot(concatenated_head_outputs, attention_output_weights)
    attention_sublayer_output = apply_dropout(attention_sublayer_output, dropout_rate, is_training)

    # ---- Step 3: Residual connection + layer normalisation ----
    # Add the original input back (residual) then normalise
    token_representations = apply_layer_normalisation(
        token_representations + attention_sublayer_output,
        layer_norm1_scale, layer_norm1_shift
    )

    # ---- Step 4: Feed forward network ----
    # Each token processed independently: expand to 128 then compress back to 64
    expanded_representations  = relu_activation(
        np.dot(token_representations, feedforward_expand_weights) + feedforward_expand_bias
    )                                                                        # (tokens, 128)
    expanded_representations  = apply_dropout(expanded_representations, dropout_rate, is_training)

    compressed_representations = (
        np.dot(expanded_representations, feedforward_compress_weights) + feedforward_compress_bias
    )                                                                        # (tokens, 64)
    compressed_representations = apply_dropout(compressed_representations, dropout_rate, is_training)

    # ---- Step 5: Residual connection + layer normalisation ----
    token_representations = apply_layer_normalisation(
        token_representations + compressed_representations,
        layer_norm2_scale, layer_norm2_shift
    )

    # ---- Step 6: Project last token to vocabulary probabilities ----
    # Only the last token's representation is used to predict the next word
    last_token_representation  = token_representations[-1:, :]              # (1, 64)
    raw_vocabulary_scores      = np.dot(last_token_representation, output_projection_weights) + output_projection_bias
    next_word_probabilities    = softmax_along_last_axis(raw_vocabulary_scores[0])  # (vocabulary_size,)

    return next_word_probabilities, attention_weights_per_head


# ---- Training loop ----

training_loss_history = []

for epoch in range(number_of_epochs):
    total_epoch_loss = 0

    for sequence_word_indices, target_word_index in zip(training_sequences, training_targets):

        predicted_probabilities, _ = transformer_block_forward(
            sequence_word_indices, is_training=True
        )

        # Cross-entropy loss: how wrong is the prediction for the correct word
        cross_entropy_loss  = -np.log(predicted_probabilities[target_word_index] + 1e-8)
        total_epoch_loss   += cross_entropy_loss

        # Simplified backward pass: only update output projection
        output_gradient                        = predicted_probabilities.copy()
        output_gradient[target_word_index]    -= 1

        last_token_embedding = word_embedding_matrix[sequence_word_indices[-1]]

        output_projection_weights -= learning_rate * np.outer(last_token_embedding, output_gradient)
        output_projection_bias    -= learning_rate * output_gradient

    average_epoch_loss = total_epoch_loss / len(training_sequences)
    training_loss_history.append(average_epoch_loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}  loss: {average_epoch_loss:.4f}")


# ---- Text generation ----

def generate_text(seed_words, number_of_words_to_generate=6):
    """
    Generate new words given a seed sequence.
    Repeatedly predicts the next word and appends it.
    """
    generated_words = seed_words.copy()

    for _ in range(number_of_words_to_generate):
        context_words   = generated_words[-sequence_length:]
        context_indices = [word_to_index.get(word, 0) for word in context_words]

        predicted_probabilities, _ = transformer_block_forward(
            context_indices, is_training=False
        )
        predicted_word_index = np.argmax(predicted_probabilities)
        generated_words.append(index_to_word[predicted_word_index])

    return ' '.join(generated_words)


print()
print("Generated text:")
print(" ", generate_text(['the', 'sky', 'is']))
print(" ", generate_text(['bring', 'your', 'umbrella']))
print(" ", generate_text(['dark', 'clouds', 'mean']))
print(" ", generate_text(['the', 'rain', 'will']))
print(" ", generate_text(['a', 'clear', 'sky']))

print(f"\nFinal loss: {training_loss_history[-1]:.4f}")


# ---- Plot loss curve ----

plt.figure(figsize=(10, 5))
plt.plot(training_loss_history, color='steelblue', linewidth=1.5, label='Transformer block')
plt.axhline(
    y=np.log(vocabulary_size),
    color='tomato', linestyle='--', linewidth=1,
    label=f'Random baseline (log {vocabulary_size} = {np.log(vocabulary_size):.2f})'
)
plt.title('Transformer Block Training Loss', fontsize=13)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Cross-Entropy Loss', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.show()

print("\nLoss curve saved to loss_curve.png")