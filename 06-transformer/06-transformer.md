# 06: Building a Transformer Block From Scratch

![Transformer block diagram](/06-transformer/images/cover.png)

Every project in this series has been building toward this one.

Project 1 was a single neuron. Project 2 added layers and backpropagation. Project 3 moved to PyTorch. Project 4 introduced sequence prediction with an RNN. Project 5 added an attention mechanism on top of the RNN.

Project 6 builds the **Transformer block**: the repeating unit inside every modern language model, including GPT, LLaMA, Gemma, and Mistral. Not a full language model yet. One block, understood completely. Project 7 will stack multiple blocks into a working mini LLM.

---

## What a Transformer block contains

A single Transformer block has four components, always arranged in the same order:

1. **Multi-head self-attention.** Every token in the sequence attends to every other token simultaneously, across multiple parallel attention heads. This is how the model decides which earlier words are relevant to each position.

2. **Feed-forward network.** After attention, each token is processed independently through a small two-layer network. This is where each token integrates and transforms what it gathered from attention.

3. **Residual connections.** The input to each component is added back to its output before passing to the next step. This skip path allows gradients to flow cleanly through many stacked blocks during training.

4. **Layer normalisation.** Applied after each residual connection. Keeps the values flowing through the network in a stable range, making training faster and more reliable.

These four ideas repeat in every block. A large language model like LLaMA 2 is just 32 or 80 of these blocks stacked on top of each other, with shared embedding and output layers.

![Transformer block showing four components stacked vertically with residual connections](/06-transformer/images/transformer_block_overview.png)

---

## Two versions in this project

This project implements the Transformer block twice.

The **numpy version** (`transformer.py`) makes every operation explicit. Every matrix multiplication, every reshape, every attention score is written out. It does not train properly because only the output layer's gradients are computed manually. It exists for understanding, not performance.

The **PyTorch version** (`06-transformer_pytorch.py`) trains all 43,405 parameters through autograd and produces real results. This is the version that matters for training.

The recommendation: read the numpy version first to understand each operation, then read the PyTorch version to see how those same operations are expressed when properly abstracted.

---

## Hyperparameters

```python
embedding_dim                 = 64
number_of_attention_heads     = 4
dimensions_per_attention_head = 16    # embedding_dim / number_of_attention_heads
feedforward_hidden_dim        = 128
dropout_rate                  = 0.1
```

The embedding dimension (64) must be exactly divisible by the number of attention heads (4). This is because the embedding is split evenly across heads: 64 / 4 = 16 dimensions per head. If these do not divide evenly, the reshape that splits queries, keys, and values into heads cannot work. The code includes an assertion that catches this immediately:

```python
assert embedding_dim % number_of_attention_heads == 0, \
    "embedding_dim must be divisible by number_of_attention_heads"
```

---

## Positional encoding: telling the model about order

The Transformer has no recurrent loop. Unlike the RNN in Projects 4 and 5, it sees all words in the sequence at the same time. This means it has no built-in sense of word order. "the sky is cloudy" and "cloudy sky is the" would look identical without some way of encoding position.

**Positional encoding** solves this by adding a unique vector to each word embedding before attention happens. The vector encodes the position of that word in the sequence.

The encoding uses sine and cosine waves at different frequencies:

```python
def compute_sinusoidal_positional_encoding(sequence_length, embedding_dim):
    position_indices  = np.arange(sequence_length)[:, np.newaxis]
    dimension_indices = np.arange(embedding_dim)[np.newaxis, :]
    frequency_scaling = position_indices / np.power(
        10000, (2 * (dimension_indices // 2)) / embedding_dim
    )
    positional_encoding          = frequency_scaling.copy()
    positional_encoding[:, 0::2] = np.sin(frequency_scaling[:, 0::2])  # even dims: sine
    positional_encoding[:, 1::2] = np.cos(frequency_scaling[:, 1::2])  # odd dims: cosine
    return positional_encoding
```

Even-numbered embedding dimensions use sine, odd-numbered use cosine. The denominator `10000^(2i/d)` creates a different frequency for each dimension. Low dimensions oscillate quickly across positions. High dimensions change slowly. Together they produce a unique fingerprint for every position, regardless of sequence length.

The positional encoding is added directly to the word embeddings:

```python
token_representations = word_embeddings + positional_encoding[:sequence_length]
```

From this point forward, each token's representation carries both its meaning (from the embedding) and its position (from the encoding).

![Sine and cosine waves at different frequencies showing how each position gets a unique encoding](/06-transformer/images/positional_encoding.png)

---

## Query, Key, and Value: what they are and why there are three

Q, K, V is how attention works. Before going further it is worth understanding clearly what these three things are.

**They are not three different inputs.** They are three different projections of the same input.

```python
all_queries = np.dot(token_representations, attention_query_weights)   # (seq_len, 64)
all_keys    = np.dot(token_representations, attention_key_weights)     # (seq_len, 64)
all_values  = np.dot(token_representations, attention_value_weights)   # (seq_len, 64)
```

The same `token_representations` tensor goes into all three. Three separate weight matrices project it into three different spaces, each specialised for a different role.

**Query:** what this token is looking for. When the model processes the word "is", its query vector encodes the question "which other words are relevant to me right now?"

**Key:** what this token contains. When the model checks whether "sky" is relevant to "is", it compares "is"'s query against "sky"'s key.

**Value:** the actual information a token contributes when selected. Once the attention weights determine which tokens matter, the model reads their value vectors to collect that information.

The process: compute a dot product between each query and every key to get relevance scores, scale and softmax those scores into weights, then take a weighted sum of all value vectors. The result is a new representation of each token that has been informed by the tokens it attended to.

![Same token representations projected into three different spaces: Query, Key and Value](/06-transformer/images/qkv_projections.png)

---

## Multi-head self-attention: running attention in parallel

In Project 5 we used a single attention head and only the last token's query. This project upgrades to **multi-head self-attention**, where:

- Every token attends to every other token (not just the last one)
- Multiple heads run the attention calculation in parallel, each in its own subspace

### Why multiple heads?

Different attention heads can learn different kinds of relationships simultaneously. One head might learn that adjectives attend to the nouns they modify. Another might learn that verbs attend to their subjects. A third might track reference across longer distances. Running four heads in parallel gives the model a richer view of the sequence than any single head could provide.

### How the split works

The Q, K, V matrices are projected to the full embedding dimension (64), then split across heads:

```python
# Before split: shape is (seq_len, 64)
# After split:  shape is (seq_len, 4 heads, 16 dims per head)

queries_per_head = all_queries.reshape(num_tokens, num_heads, dims_per_head)
keys_per_head    = all_keys.reshape(   num_tokens, num_heads, dims_per_head)
values_per_head  = all_values.reshape( num_tokens, num_heads, dims_per_head)
```

The numbers do not change. The reshape just instructs numpy (or PyTorch) to treat the 64 dimensions as 4 groups of 16. Each head then works with its own 16-dimensional slice.

### Each head computes attention independently

```python
for head_index in range(number_of_attention_heads):
    head_queries = queries_per_head[:, head_index, :]     # (seq_len, 16)
    head_keys    = keys_per_head[:,    head_index, :]
    head_values  = values_per_head[:,  head_index, :]

    raw_scores        = np.dot(head_queries, head_keys.T)
    scaled_scores     = raw_scores / math.sqrt(dims_per_head)
    masked_scores     = scaled_scores + causal_mask
    attention_weights = softmax(masked_scores)
    head_output       = np.dot(attention_weights, head_values)    # (seq_len, 16)
```

### Concatenate and project

After all four heads compute their outputs, they are concatenated back to a (seq_len, 64) shape and passed through one more linear layer:

```python
concatenated = np.concatenate(head_outputs, axis=-1)        # (seq_len, 64)
attention_output = np.dot(concatenated, output_weights)      # (seq_len, 64)
```

This final projection mixes information across all four heads into a single unified representation.

![Multi-head attention showing split into four heads, parallel computation, and concatenation](/06-transformer/images/multi_head_attention.png)

---

## The causal mask: preventing the model from seeing the future

The Transformer sees all tokens simultaneously. Without any constraint, a token at position 2 could attend to tokens at position 3, 4, and beyond, meaning the model could see the answer before predicting it.

The **causal mask** prevents this. It blocks attention from flowing to future positions by adding a large negative number to the corresponding attention scores before softmax. After softmax, those positions get a weight of approximately zero.

```python
causal_mask   = np.triu(np.full_like(scaled_scores, -1e9), k=1)
masked_scores = scaled_scores + causal_mask
```

`np.triu(..., k=1)` fills the upper triangle above the diagonal with -1e9. The result for a three-token sequence looks like this:

```text
                 attends to:
               token 0   token 1   token 2
token 0  [ allowed    blocked    blocked  ]
token 1  [ allowed    allowed    blocked  ]
token 2  [ allowed    allowed    allowed  ]
```

Token 0 can only see itself. Token 1 can see tokens 0 and 1. Token 2 can see all three. No token ever attends to a position that comes after it in the sequence.

![3x3 causal mask matrix showing allowed attention positions and blocked future positions](/06-transformer/images/causal_mask.png)

---

## Residual connections: why deep networks can be trained at all

After attention, the original input is added back to the attention output before layer normalisation:

```python
token_representations = layer_norm(token_representations + attention_output)
```

That addition is the **residual connection** (also called a skip connection). The network learns to compute a correction to add to its input, rather than learning to transform the input from scratch.

Without residual connections, training a network with many stacked layers would fail. As gradients flow backward through layer after layer, they would shrink and vanish before reaching the early layers. The weights in early layers would barely update, and the network would effectively stop learning.

Residual connections create a direct path that gradients can take through the entire network. Even if the attention sublayer produces very small gradients, the gradient can bypass it via the skip path and reach earlier layers undiminished.

The same pattern is applied after the feed-forward network:

```python
token_representations = layer_norm(token_representations + feedforward_output)
```

![Residual connection diagram showing skip path bypassing the attention sublayer](/06-transformer/images/residual_connection.png)

---

## Layer normalisation: keeping values stable

After each residual connection, layer normalisation is applied:

```python
def apply_layer_normalisation(x, scale, shift, epsilon=1e-6):
    mean        = np.mean(x, axis=-1, keepdims=True)
    variance    = np.var(x,  axis=-1, keepdims=True)
    normalised  = (x - mean) / np.sqrt(variance + epsilon)
    return scale * normalised + shift
```

The mean and variance are computed across the embedding dimension for each token independently. After normalisation, the values are rescaled and shifted by learned parameters (`scale` and `shift`).

As data flows through many layers of a deep network, values can grow or shrink in unpredictable ways. Without normalisation, some neurons saturate (becoming stuck at their activation extremes) and gradients flow poorly. Layer normalisation keeps values in a healthy range at every layer, making training faster and more stable.

The small constant `epsilon` in the denominator prevents a division-by-zero when variance is exactly zero.

---

## The feed-forward network: where each token thinks for itself

After attention has allowed tokens to gather information from each other, each token is passed independently through a small two-layer network:

```python
# Expand from 64 to 128 dimensions
expanded = relu(np.dot(token_representations, expand_weights) + expand_bias)

# Compress back to 64 dimensions
compressed = np.dot(expanded, compress_weights) + compress_bias
```

This **expand-then-compress** structure serves two purposes.

First, it gives the model room to think. The intermediate expansion to 128 dimensions creates a larger space in which each token can transform its representation. More dimensions means more capacity to represent complex patterns.

Second, the ReLU activation between the two layers adds non-linearity. Without it, the two linear layers would collapse into a single linear layer and the feed-forward network would add no expressive power. ReLU allows each layer to learn something genuinely new from the previous one.

Attention is how tokens communicate with each other. The feed-forward network is where each token independently processes and transforms what it collected.

![Feed-forward network showing 64 to 128 expansion with ReLU then compression back to 64](/06-transformer/images/feedforward_network.png)

---

## Dropout: preventing over-reliance on any single pathway

Dropout appears in three places: after the attention weights, after the attention output projection, and after the first feed-forward layer.

```python
def apply_dropout(activations, dropout_rate, is_training=True):
    if not is_training or dropout_rate == 0:
        return activations
    keep_probability = 1 - dropout_rate
    mask = np.random.binomial(1, keep_probability, activations.shape) / keep_probability
    return activations * mask
```

During training, 10% of activations are randomly set to zero. The remaining values are scaled up by `1 / keep_probability` to preserve the expected sum. During generation (inference), dropout is turned off entirely.

The effect: the network cannot rely too heavily on any single neuron or pathway. If a pathway gets dropped at random during training, the network must learn alternative ways to represent the same information. This redundancy helps the model generalise better to unseen examples.

This is the first proper solution to the overfitting problem that has appeared in every project since Project 2. Overfitting in small models comes partly from over-reliance on a small number of strong pathways. Dropout disrupts those pathways during training and forces the model to spread its learning more broadly.

---

## Parameter count

```text
Component                          Shape          Parameters
------------------------------------------------------------
word_embedding_matrix              77 x 64            4,928
attention_query_weights            64 x 64            4,096
attention_key_weights              64 x 64            4,096
attention_value_weights            64 x 64            4,096
attention_output_weights           64 x 64            4,096
layer_norm_1 (scale + shift)        2 x 64              128
feedforward_expand_weights         64 x 128           8,192
feedforward_expand_bias                128              128
feedforward_compress_weights      128 x 64            8,192
feedforward_compress_bias               64               64
layer_norm_2 (scale + shift)        2 x 64              128
output_projection_weights          64 x 77            4,928
output_projection_bias                  77               77
------------------------------------------------------------
Total (numpy version):                                43,149
Total (PyTorch version):                              43,405
```

The PyTorch version has 256 more parameters. `nn.MultiheadAttention` adds bias terms to the Q, K, V projections by default (4 heads x 64 dimensions = 256 extra). The architecture is otherwise identical.

---

## The PyTorch version

The numpy version made every operation explicit. The PyTorch version trains all 43,405 parameters and produces real results.

```python
class TransformerBlock(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, number_of_attention_heads,
                 feedforward_hidden_dim, dropout_rate, sequence_length):
        super(TransformerBlock, self).__init__()

        self.word_embedding = nn.Embedding(vocabulary_size, embedding_dim)

        positional_encoding = self._compute_positional_encoding(sequence_length, embedding_dim)
        self.register_buffer('positional_encoding', positional_encoding)

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=number_of_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.layer_norm_after_attention   = nn.LayerNorm(embedding_dim)
        self.feedforward_expand           = nn.Linear(embedding_dim, feedforward_hidden_dim)
        self.feedforward_compress         = nn.Linear(feedforward_hidden_dim, embedding_dim)
        self.feedforward_relu             = nn.ReLU()
        self.feedforward_dropout          = nn.Dropout(dropout_rate)
        self.layer_norm_after_feedforward = nn.LayerNorm(embedding_dim)
        self.output_projection            = nn.Linear(embedding_dim, vocabulary_size)

    def forward(self, word_indices_in_sequence):
        seq_len = word_indices_in_sequence.shape[1]

        # Step 1: embed words and add positional encoding
        token_representations = (self.word_embedding(word_indices_in_sequence)
                                  + self.positional_encoding[:, :seq_len, :])

        # Step 2: masked multi-head self-attention
        causal_mask = self._build_causal_mask(seq_len)
        attention_output, _ = self.multihead_attention(
            query=token_representations,
            key=token_representations,
            value=token_representations,
            attn_mask=causal_mask
        )

        # Step 3: residual connection + layer normalisation
        token_representations = self.layer_norm_after_attention(
            token_representations + attention_output
        )

        # Step 4: feed-forward network
        ff = self.feedforward_expand(token_representations)
        ff = self.feedforward_relu(ff)
        ff = self.feedforward_dropout(ff)
        ff = self.feedforward_compress(ff)

        # Step 5: residual connection + layer normalisation
        token_representations = self.layer_norm_after_feedforward(
            token_representations + ff
        )

        # Step 6: take the last token and project to vocabulary
        last_token    = token_representations[:, -1, :]
        output_scores = self.output_projection(last_token)
        return output_scores
```

`nn.MultiheadAttention` replaces the entire head-splitting, scoring, masking, and concatenating loop from the numpy version. Because we already wrote that loop by hand, this line is transparent rather than mysterious.

`register_buffer` stores the positional encoding as part of the model but not as a learned parameter. It moves to the correct device automatically when you call `.to(device)`.

The residual connections (`token_representations + attention_output`, `token_representations + ff`) are still written explicitly. This is intentional: they are conceptually important and worth keeping visible even when everything else is abstracted.

---

## Training results

```text
Using device: cpu
Vocabulary size: 77
Training sequences: 130
Validation sequences: 33
Total parameters: 43,405

Epoch     0  loss: 4.4567
Epoch   200  loss: 0.0252
Epoch   400  loss: 0.0816
Epoch   600  loss: 0.0113
Epoch   800  loss: 0.0122
```

The loss drops from 4.4567 (random baseline is `log(77) = 4.34`) to the 0.01 range. The slight oscillation between epochs 200 and 400 is normal with Adam on a small dataset. The network is finding different near-optimal configurations and settling between them.

![Training loss curve showing drop from 4.4567 to 0.01 range over 1000 epochs](/06-transformer/images/loss_curve.png)

---

## Generated text

```text
the sky is        ->  cloudy today bring your umbrella when
bring your umbrella -> when it rains dark clouds mean
dark clouds mean  ->  heavy rain the weather looks wet
the rain will     ->  stop by evening sunny weather makes
a clear sky       ->  means no rain today carry an
```

Every seed continues correctly into the next training sentence. Compare this to an RNN trained for the same number of epochs, which tends to repeat high-frequency words or degenerate toward the same phrase regardless of the seed. The Transformer's ability to attend directly to all context words rather than relying on the hidden state to carry that information forward makes a real difference even on this small corpus.

---

## What PyTorch handles vs what stays visible

```text
Abstracted by PyTorch:              Kept explicit in our code:
  Multi-head attention internals      Residual connections: x = x + sublayer(x)
  All gradient computation            Block structure: embed, attend, FFN, project
  Layer normalisation math            Causal mask construction
  Dropout implementation              The six forward-pass steps
  Positional encoding device moves
```

The residual connections stay visible because they are conceptually load-bearing. `token_representations = layer_norm(token_representations + attention_output)` is worth reading carefully. The model is learning what correction to add to its representation, not learning to replace it from scratch.

---

## What is next

Project 7 stacks four of these blocks into a complete mini language model, adds a proper batched training loop with a learning rate schedule, and trains on the same weather corpus. The model will produce coherent multi-sentence text and will be architecturally equivalent to early GPT models.

---

## Running

```bash
pip install torch numpy matplotlib
python 06-transformer_pytorch.py
```

The script trains for 1000 epochs, prints loss every 200 epochs, generates five text samples, and saves a loss curve to `loss_curve.png`.

## Files

```text
transformer.py                numpy version (explicit math, no proper training)
06-transformer_pytorch.py     PyTorch version (full training, 43,405 parameters)
weather_corpus.txt            30-sentence training corpus
config.json                   hyperparameters (embedding dim, heads, FFN size, epochs)
images/
  cover.png
  transformer_block_overview.png
  positional_encoding.png
  qkv_projections.png
  multi_head_attention.png
  causal_mask.png
  residual_connection.png
  feedforward_network.png
  loss_curve.png
```
