# RNN with Attention in PyTorch

![cover](/05-attention/images/cover.png)

*Project 5 in my build series at github.com/dwinsi/LLMfromScratch*

Project 4 built a word-level RNN from scratch using numpy. Every matrix multiplication was explicit. Every gradient was computed by hand. The math was visible but the code was long.

Project 3 showed what happens when you rebuild a numpy network in PyTorch. Thirty lines of manual backpropagation became three. The math did not change. The abstraction did.

Project 5 does the same thing for the RNN with attention. Same architecture as the numpy version. Same corpus, same vocabulary, same Q, K, V attention mechanism. But now PyTorch handles the RNN, the gradients, and the weight updates.

The attention forward pass stays written by hand. That is intentional. The Q, K, V math is the thing worth seeing clearly. The RNN loop is the thing worth abstracting.

---

## What PyTorch replaces

In the numpy version, the RNN forward pass was a manual loop collecting hidden states at every step. In PyTorch, nn.RNN replaces this entirely.

```python
self.rnn = nn.RNN(
    input_size=hidden_size,
    hidden_size=hidden_size,
    batch_first=True
)

rnn_output, final_hidden = self.rnn(embedded_input)
```

rnn_output is a tensor of shape (batch, sequence_length, hidden_size). It contains all the hidden states from every time step, the same matrix we built manually in numpy. We no longer need to loop, append and vstack. PyTorch does it in one call.

The gradients for the entire RNN, through all time steps, flow automatically when we call loss.backward().

![Manual numpy RNN loop versus single nn.RNN call in PyTorch](/05-attention/images/numpy_vs_pytorch_rnn.png)

---

## The embedding layer

In Project 4 we converted each word to a one-hot vector and multiplied it by the weight matrix. As we noted in that article, one-hot times a weight matrix is just a row lookup.

PyTorch makes this explicit with nn.Embedding.

```python
self.embedding = nn.Embedding(vocabulary_size, hidden_size)
```

Instead of creating a sparse one-hot vector and multiplying, nn.Embedding directly indexes into a learned weight matrix. Same result, much faster, cleaner code. The embedding layer also adds learnable parameters: vocabulary_size times hidden_size = 77 times 64 = 4,928.

![One-hot multiply versus nn.Embedding direct row lookup comparison](/05-attention/images/embedding_lookup.png)

---

## The network definition

```python
class RNNWithAttention(nn.Module):

    def __init__(self, vocabulary_size, hidden_size, attention_size):
        super(RNNWithAttention, self).__init__()

        self.hidden_size    = hidden_size
        self.attention_size = attention_size

        # Embedding layer: word index -> dense vector
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)

        # RNN layer: processes the sequence, returns all hidden states
        # batch_first=True means input shape is (batch, sequence, features)
        # which is more intuitive than the default (sequence, batch, features)
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
```

The forward method reads like a description of the architecture. Embed, run through RNN, compute Q K V, attend, output. The attention weights are returned alongside the output scores so we can inspect what the network is attending to during generation.

---

## Device setup and GPU support

Before creating the model or any tensors, the code detects which device to use.

```python
device = torch.device(
    'cuda'  if torch.cuda.is_available()  else
    'mps'   if torch.backends.mps.is_available() else
    'cpu'
)
print(f"Using device: {device}")
```

`cuda` is for NVIDIA GPUs. `mps` is for Apple Silicon, M1, M2, M3 chips. `cpu` is the fallback. The code checks in that order and uses the first one available. When you run the script it will print which device it found.

For this small corpus with 163 training sequences the GPU will not feel noticeably faster than CPU. The overhead of moving data to the GPU is roughly equal to the computation saved at this scale. But writing device-agnostic code from the start is the right habit. When the series reaches the mini LLM in Project 7, the GPU will make a real difference.

The model and all tensors are moved to the chosen device with `.to(device)`.

```python
model                     = RNNWithAttention(...).to(device)
training_sequences_tensor = [torch.tensor(seq).to(device) for seq in training_sequences]
training_targets_tensor   = [torch.tensor([tgt]).to(device) for tgt in training_targets]
```

Every tensor that touches the model must live on the same device as the model. Mixing CPU and GPU tensors causes an error.

---

## Training setup

```python
model         = RNNWithAttention(vocabulary_size, hidden_size=64, attention_size=32).to(device)
loss_function = nn.CrossEntropyLoss()
optimiser     = optim.Adam(model.parameters(), lr=0.001)
```

Two changes from Project 3 worth noting.

nn.CrossEntropyLoss combines softmax and negative log likelihood into one operation. It is the standard loss for classification over multiple classes, identical to what we computed manually in Project 4.

Adam is an adaptive learning rate optimiser that adjusts the learning rate for each parameter individually based on how much it has been updated. For language models it typically converges faster and more reliably than plain SGD.

---

## Parameter count

```python
total_parameters = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_parameters:,}")
```

The breakdown for our network:

```
embedding layer:          77 × 64  =  4,928
rnn input weights:        64 × 64  =  4,096
rnn hidden weights:       64 × 64  =  4,096
rnn bias:                      64  =     64
weights_query:            64 × 32  =  2,048
weights_key:              64 × 32  =  2,048
weights_value:            64 × 64  =  4,096
output layer weights:     64 × 77  =  4,928
output layer bias:             77  =     77
                                    -------
Total:                             26,381
```

Slightly more than the numpy version because the embedding layer is now a proper learned parameter rather than a fixed one-hot lookup.

---

## Training loop

```python
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for sequence_tensor, target_tensor in zip(training_sequences_tensor,
                                               training_targets_tensor):

        sequence_input = sequence_tensor.unsqueeze(0)   # (1, seq_len)
        target_input   = target_tensor                   # (1,)

        optimiser.zero_grad()
        output_scores, attention_weights = model(sequence_input)
        loss = loss_function(output_scores, target_input)
        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    average_loss = total_loss / len(training_sequences_tensor)
    training_loss_history.append(average_loss)

    if epoch % 200 == 0:
        print(f"Epoch {epoch:5d}  loss: {average_loss:.4f}")
```

loss.backward() computes gradients for every parameter in the network including the embedding layer, the RNN weights, all three attention matrices and the output layer. All of it in one line.

Note that `target_input` does not need unsqueeze here. nn.CrossEntropyLoss expects the target as a 1D tensor of class indices, shape (batch,), not (batch, 1).

---

## Training results

Run the script on your machine and update this section with the actual output. The expected pattern based on the numpy version is a steep drop in the first 200 epochs followed by slower improvement.

```
Using device: [cuda / mps / cpu depending on your machine]
Vocabulary size: 77
Training sequences: 163
Total parameters: 26,381

Epoch     0  loss: [your value]
Epoch   200  loss: [your value]
Epoch   400  loss: [your value]
Epoch   600  loss: [your value]
Epoch   800  loss: [your value]

Generated text:
  the sky is [your output]
  bring your umbrella [your output]
  dark clouds mean [your output]
  the rain will [your output]
  a clear sky [your output]
```

Replace the placeholders above with your actual numbers before publishing.

![Training loss curve showing convergence over 1000 epochs](/05-attention/images/loss_curve.png)
![sgd_vs_adam](/05-attention/images/sgd_vs_adam.png)

---

## Text generation

```python
def generate_text(seed_words, number_of_words_to_generate=6):
    model.eval()
    generated_words = seed_words.copy()

    with torch.no_grad():
        for _ in range(number_of_words_to_generate):
            context_words   = generated_words[-sequence_length:]
            context_indices = [word_to_index[w] for w in context_words
                               if w in word_to_index]
            sequence_tensor = torch.tensor(context_indices).unsqueeze(0)

            output_scores, _ = model(sequence_tensor)
            predicted_index  = torch.argmax(output_scores, dim=-1).item()
            generated_words.append(index_to_word[predicted_index])

    return ' '.join(generated_words)
```

torch.no_grad() tells PyTorch not to track gradients during generation. No backward pass is needed and this saves memory and computation.

---

## What PyTorch replaced vs what stayed visible

It is worth being explicit about the trade-off made here.

```
PyTorch handles:                  We kept visible:
  RNN loop and hidden states        Q, K, V projections
  All gradient computation          Scaled dot product
  Weight updates                    Softmax attention weights
  Embedding lookup                  Context vector construction
```

The attention math stayed hand-rolled because that is what this project is about. Every line of the attention forward pass has a direct counterpart in the numpy version from the previous article. Seeing both versions side by side is what makes each one meaningful.

---

## What I took away from this

Rebuilding the numpy attention network in PyTorch felt like the same moment as Project 3. The code got shorter. The intent got clearer. The math did not change.

Reading torch.bmm(last_query, key_vectors.transpose(1, 2)) and knowing exactly what that batch matrix multiplication is doing, because we already wrote it manually in numpy, is a completely different experience from using it as a black box.

That is the whole point of the build-from-scratch-first approach. Every abstraction in PyTorch now has a concrete referent in something we already built.

Project 6 moves to the full Transformer block. Self-attention across the entire sequence, feed forward layers, residual connections, layer normalisation. Everything we have been building toward.

---

*Project 5 code lives at github.com/dwinsi/LLMfromScratch in the 05-attention folder. Run rnn_attention_pytorch.py and compare the loss curve and generated text against the numpy version from Project 4.*
