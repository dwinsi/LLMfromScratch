# 05: RNN with Attention in PyTorch

Project 4 built a word-level RNN from scratch using numpy. Every matrix multiplication was explicit. Every gradient was computed by hand. The math was visible but the code was long and the architecture had a fundamental weakness: the hidden state could only carry context forward for a few words before earlier information faded away.

This project addresses both of those things at once.

It rebuilds the RNN in PyTorch (replacing the manual hidden state loop with `nn.RNN`), and it adds an **attention mechanism** that allows the network to look back at any earlier word directly rather than relying on the hidden state to carry that memory forward.

The attention math stays written by hand. That is intentional. The Q, K, V calculation is the thing worth seeing clearly at this stage. The RNN loop is safe to abstract.

---

## The problem attention solves

Recall from Project 4 that the RNN processes words one at a time and compresses everything it has read into a single fixed-size hidden state vector. By the time it reaches the last word, the hidden state must represent the entire sequence in just 64 numbers.

This works for short sequences. For longer ones, earlier information gets overwritten as the hidden state is updated with each new word. The network forgets.

Attention solves this by giving the network a direct line back to every position in the sequence. Instead of asking "what is stored in the hidden state?", the network can ask "which words in my input are most relevant to what I am trying to predict right now?" and retrieve them directly.

---

## How attention works: queries, keys, and values

Attention is built on three concepts: **queries**, **keys**, and **values**. The names come from information retrieval (think of a database search), but the math is straightforward.

Every hidden state at every position in the sequence gets projected into three different vectors:

- **Query (Q):** what this position is looking for
- **Key (K):** what this position contains (used to match against queries)
- **Value (V):** the actual information this position will contribute if selected

To produce an attended output, the network takes the query of the position it cares about most (in this case, the last position in the sequence, because that is the one making the prediction) and compares it against the keys of every other position using a dot product.

A high dot product between a query and a key means those two positions are relevant to each other. These scores are scaled and passed through softmax to produce attention weights: a probability distribution over the sequence positions.

The final output is a weighted sum of all the value vectors, weighted by those attention weights. Positions with high attention weights contribute more to the output.

In code, the full calculation is:

```python
# Project each hidden state into Q, K, V
query_vectors = self.weights_query(rnn_output)       # (batch, seq_len, attention_size)
key_vectors   = self.weights_key(rnn_output)         # (batch, seq_len, attention_size)
value_vectors = self.weights_value(rnn_output)       # (batch, seq_len, hidden_size)

# Use only the last position's query (it is making the prediction)
last_query = query_vectors[:, -1:, :]                # (batch, 1, attention_size)

# Compare last query against all keys: dot product scores
attention_scores  = torch.bmm(last_query, key_vectors.transpose(1, 2))
attention_scores  = attention_scores / (self.attention_size ** 0.5)   # scale
attention_weights = torch.softmax(attention_scores, dim=-1)           # normalise

# Weighted sum of values: the context vector
context_vector = torch.bmm(attention_weights, value_vectors)          # (batch, 1, hidden_size)
context_vector = context_vector.squeeze(1)                            # (batch, hidden_size)
```

The scaling step (`/ attention_size ** 0.5`) prevents the dot products from becoming very large when the attention dimension is large, which would push softmax into its saturated region where gradients are near zero.

---

## What each new PyTorch building block does

### nn.Embedding

In Project 4 we converted each word to a one-hot vector and multiplied it by a weight matrix. As noted there, multiplying a one-hot vector by a matrix just selects one row of that matrix. `nn.Embedding` does exactly this, but as a direct index lookup rather than a sparse matrix multiply. Faster, cleaner, and the weights in the embedding table are learned during training just like any other weight matrix.

```python
self.embedding = nn.Embedding(vocabulary_size, hidden_size)
# Usage: self.embedding(word_indices)  -> (batch, seq_len, hidden_size)
```

`nn.Embedding(77, 64)` creates a learnable table of 77 rows and 64 columns. When you pass it an integer index, it returns the corresponding row.

### nn.RNN

In Project 4, the RNN forward pass was an explicit Python loop that updated the hidden state one word at a time and collected each hidden state into a list. `nn.RNN` replaces that entire loop with a single call:

```python
self.rnn = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

rnn_output, final_hidden = self.rnn(embedded_input)
```

`rnn_output` has shape `(batch, sequence_length, hidden_size)`. It contains the hidden state at every position in the sequence, the same matrix we built manually in numpy. We no longer need to loop, append, and stack. PyTorch handles the recurrent loop internally and computes all gradients through all time steps automatically when `loss.backward()` is called.

`batch_first=True` means the batch dimension comes first in the input tensor, which is the more intuitive layout when you think of data as rows of examples.

### nn.CrossEntropyLoss

In Project 4 we computed softmax and then negative log likelihood separately. `nn.CrossEntropyLoss` combines both into one operation and is numerically more stable:

```python
loss_function = nn.CrossEntropyLoss()
loss = loss_function(output_scores, target_indices)
```

It expects raw scores (logits) as input, not probabilities. It applies log-softmax internally. The result is mathematically identical to what we computed manually in Project 4.

### Adam optimiser

Projects 1 through 4 used SGD (stochastic gradient descent): every weight is updated by the same fixed learning rate. Adam is an adaptive optimiser that tracks a running average of both the gradient and the squared gradient for each parameter, and uses that history to scale the learning rate individually for each weight.

The practical effect: Adam typically converges faster and is more forgiving of the learning rate choice. The script runs both Adam and SGD on the same network so you can compare the loss curves directly.

---

## The full network

```python
class RNNWithAttention(nn.Module):

    def __init__(self, vocabulary_size, hidden_size, attention_size):
        super(RNNWithAttention, self).__init__()

        self.hidden_size    = hidden_size
        self.attention_size = attention_size

        self.embedding     = nn.Embedding(vocabulary_size, hidden_size)
        self.rnn           = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.weights_query = nn.Linear(hidden_size, attention_size, bias=False)
        self.weights_key   = nn.Linear(hidden_size, attention_size, bias=False)
        self.weights_value = nn.Linear(hidden_size, hidden_size,    bias=False)
        self.output_layer  = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, input_sequence):

        # Step 1: convert word indices to dense vectors
        embedded_input = self.embedding(input_sequence)       # (batch, seq_len, hidden_size)

        # Step 2: run the RNN, get a hidden state for every position
        rnn_output, _ = self.rnn(embedded_input)              # (batch, seq_len, hidden_size)

        # Step 3: project hidden states into Q, K, V
        query_vectors = self.weights_query(rnn_output)        # (batch, seq_len, attention_size)
        key_vectors   = self.weights_key(rnn_output)          # (batch, seq_len, attention_size)
        value_vectors = self.weights_value(rnn_output)        # (batch, seq_len, hidden_size)

        # Step 4: attend from the last position to all positions
        last_query        = query_vectors[:, -1:, :]          # (batch, 1, attention_size)
        attention_scores  = torch.bmm(last_query, key_vectors.transpose(1, 2))
        attention_scores  = attention_scores / (self.attention_size ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Step 5: weighted sum of values
        context_vector = torch.bmm(attention_weights, value_vectors).squeeze(1)

        # Step 6: project to vocabulary size
        output_scores = self.output_layer(context_vector)     # (batch, vocabulary_size)

        return output_scores, attention_weights.squeeze(1)
```

The `forward` method reads as a description of the architecture: embed, run through RNN, compute Q K V, attend, output. The attention weights are returned alongside the scores so we can inspect which words the network focused on during generation.

---

## Device setup: GPU support

Before creating the model or any tensors, the code detects which hardware to use:

```python
device = torch.device(
    'cuda' if torch.cuda.is_available()  else
    'mps'  if torch.backends.mps.is_available() else
    'cpu'
)
```

`cuda` is for NVIDIA GPUs. `mps` is for Apple Silicon (M1/M2/M3). `cpu` is the fallback.

The model and all tensors must live on the same device:

```python
model = RNNWithAttention(...).to(device)
sequence_tensor = torch.tensor(indices).to(device)
```

For this small corpus the GPU will not feel noticeably faster than CPU. The overhead of moving data to the GPU is comparable to the computation at this scale. But writing device-agnostic code from the start is the right habit. By Project 7 (the mini LLM), the GPU makes a real difference.

---

## Parameter count

```text
Component                     Shape        Parameters
-----------------------------------------------------
embedding layer               77 x 64        4,928
rnn input weights             64 x 64        4,096
rnn hidden weights            64 x 64        4,096
rnn bias                           64            64
weights_query                 64 x 32        2,048
weights_key                   64 x 32        2,048
weights_value                 64 x 64        4,096
output layer weights          64 x 77        4,928
output layer bias                  77            77
-----------------------------------------------------
Total                                        26,381
```

Slightly more than the numpy version from Project 4 (14,093 parameters) because the embedding layer is now a proper learned matrix rather than a fixed one-hot lookup.

---

## The training loop

```python
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for sequence_tensor, target_tensor in zip(training_sequences_tensor, training_targets_tensor):
        sequence_input = sequence_tensor.unsqueeze(0)   # add batch dimension: (1, seq_len)
        target_input   = target_tensor                   # (1,)

        optimiser.zero_grad()
        output_scores, attention_weights = model(sequence_input)
        loss = loss_function(output_scores, target_input)
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
```

`loss.backward()` computes gradients for every parameter: the embedding table, all RNN weights, all three attention matrices, and the output layer. One line handles the entire network.

The target tensor does not need an extra dimension. `nn.CrossEntropyLoss` expects targets as a 1D tensor of class indices with shape `(batch,)`, not `(batch, 1)`.

---

## Adam vs SGD: a direct comparison

The script trains two identical networks from the same random seed: one with Adam, one with SGD. This makes the difference in convergence speed directly visible.

Adam typically reaches a much lower loss in the same number of epochs. On this corpus at 1000 epochs:

```text
Optimiser    Final train loss    Final val loss
---------------------------------------------
Adam              ~0.05              ~0.35
SGD               ~1.80              ~2.10
```

Adam's advantage comes from its per-parameter learning rate adaptation. Weights that receive large, consistent gradients get their learning rate reduced automatically. Weights that receive small or noisy gradients get their learning rate increased. This makes the overall optimisation smoother and faster.

SGD with a well-tuned learning rate can eventually reach similar results, but it takes more epochs and more careful tuning of the learning rate.

---

## Attention heatmaps

The script also produces a heatmap showing which words the network attended to when making each prediction. For a three-word input sequence, the attention weights are three numbers that sum to 1. A high weight on a position means the network relied heavily on the hidden state at that position when forming the context vector.

```text
Input: ['bring', 'your', 'umbrella']

Attention weights:
  bring:     0.12
  your:      0.21
  umbrella:  0.67   <- most attended
```

The network places the most weight on the last word ("umbrella") when predicting the next word ("when"). This makes sense for this corpus: "umbrella when" is a strong pattern in the training data.

The heatmaps are saved to `attention_heatmap.png` when you run the script.

---

## What PyTorch handles vs what stays visible

```text
Abstracted by PyTorch:              Kept explicit in our code:
  RNN hidden state loop               Q, K, V projection matrices
  All gradient computation            Scaled dot-product scores
  Weight initialisation               Softmax attention weights
  Embedding lookup mechanics          Context vector construction
  GPU/CPU tensor placement            Attention weight inspection
```

The attention math stays hand-written because this project is about understanding attention. Every line of the attention forward pass can be traced back to the Q, K, V description above. Seeing it written out at this scale makes the same math in larger Transformer models much easier to follow in later projects.

---

## What is next

Project 6 moves to the full **Transformer block**: self-attention across all positions simultaneously (not just from the last position), feed-forward layers, residual connections, and layer normalisation. All of the concepts introduced in this project are used there in a more general form.

---

## Running

```bash
pip install torch matplotlib
python 05-rnn_attention_pytorch.py
```

The script trains two models (Adam and SGD), prints loss every 200 epochs for both, generates text samples from both models, saves an Adam vs SGD comparison plot to `sgd_vs_adam.png`, and saves an attention heatmap to `attention_heatmap.png`.

## Files

```text
05-rnn_attention_pytorch.py    full training script with attention
weather_corpus.txt             30-sentence training corpus
config.json                    hyperparameters (hidden size, attention size, epochs, learning rate)
images/
  sgd_vs_adam.png              training and validation loss for both optimisers
  attention_heatmap.png        per-word attention weights for sample inputs
```
