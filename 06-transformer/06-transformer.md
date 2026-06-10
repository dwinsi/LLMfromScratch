# Building a Transformer Block From Scratch

![Transformer](/06-transformer/images/cover.png)

*Project 6 in my build series at github.com/dwinsi/LLMfromScratch*

Every project in this series has been building toward this one.

Project 1 was a single neuron. Project 2 added layers and backpropagation. Project 3 moved to PyTorch. Project 4 introduced sequence prediction with an RNN. Project 5 added attention on top of the RNN.

Project 6 builds the Transformer block. Not a full language model yet. One block, understood completely. Project 7 will stack multiple blocks into the mini LLM.

This is where the theory series and the build series fully converge.

---

## What a Transformer block contains

A single Transformer block has four components arranged in a specific order.

**Multi-head self-attention.** Every token attends to every other token simultaneously, across multiple attention heads in parallel.

**Feed forward network.** After attention, each token is passed independently through a small two-layer network. This adds non-linearity and depth.

**Residual connections.** The input to each sublayer is added back to its output. This skip connection makes stacking many blocks possible.

**Layer normalisation.** Applied after each sublayer. Stabilises values and makes training faster.

One block. Four ideas. This is the repeating unit inside every modern LLM.

![Transformer block showing four components stacked vertically with residual connections](/06-transformer/images/transformer_block_overview.png)

---

## Two versions, one purpose

This project builds the Transformer block twice.

The numpy version makes every operation explicit. The forward pass is written line by line so the math is visible. It does not train properly because the backward pass only updates the output layer. It is purely for understanding.

The PyTorch version trains all 43,405 parameters through autograd. It produces real results. This is the version that matters for training. The numpy version is the version that matters for learning.

---

## Hyperparameters

```python
embedding_dim                 = 64
number_of_attention_heads     = 4
dimensions_per_attention_head = 16    # embedding_dim // number_of_attention_heads
feedforward_hidden_dim        = 128
dropout_rate                  = 0.1

assert embedding_dim % number_of_attention_heads == 0, \
    "embedding_dim must be divisible by number_of_attention_heads"
```

The constraint between embedding_dim and number_of_attention_heads is hard. 64 divided by 4 gives 16 dimensions per head. If these do not divide evenly the reshape operation that splits queries, keys and values into heads will fail. The assertion catches this immediately.

---

## Positional encoding

The Transformer has no recurrent loop. Unlike the RNN in Projects 4 and 5, it sees all words simultaneously. This means it has no built-in sense of word order. Positional encoding injects that order information before attention happens.

Each position in the sequence gets a unique vector added to its embedding.

```python
def compute_sinusoidal_positional_encoding(sequence_length, embedding_dim):
    position_indices  = np.arange(sequence_length)[:, np.newaxis]
    dimension_indices = np.arange(embedding_dim)[np.newaxis, :]
    frequency_scaling = position_indices / np.power(
        10000, (2 * (dimension_indices // 2)) / embedding_dim
    )
    positional_encoding          = frequency_scaling.copy()
    positional_encoding[:, 0::2] = np.sin(frequency_scaling[:, 0::2])
    positional_encoding[:, 1::2] = np.cos(frequency_scaling[:, 1::2])
    return positional_encoding
```

Sine functions for even dimensions, cosine for odd. The denominator creates different frequencies across dimensions. Low dimensions oscillate quickly with position. High dimensions change slowly. Together they produce a unique fingerprint for every position that the model can learn to read.

The result is added directly to the word embeddings before attention.

```python
word_embeddings       = word_embedding_matrix[word_indices_in_sequence]
token_representations = word_embeddings + precomputed_positional_encoding[:number_of_tokens]
```

Without positional encoding, the Transformer would treat "the sky is cloudy" identically to "cloudy sky the is". The order would be invisible.

![Sine and cosine waves at different frequencies showing how each position gets a unique encoding](/06-transformer/images/positional_encoding.png)

---

## Query, Key and Value

Before explaining multi-head attention, Query, Key and Value need a clear explanation because they appear in every attention calculation.

Q, K and V are not three different inputs. They are three different projections of the same input.

```python
all_queries = np.dot(token_representations, attention_query_weights)
all_keys    = np.dot(token_representations, attention_key_weights)
all_values  = np.dot(token_representations, attention_value_weights)
```

The same `token_representations` goes into all three. Three different weight matrices project it into three different spaces.

**Query** represents what each token is looking for. When token "is" wants to figure out which other tokens are relevant to it, it uses its query vector to ask the question.

**Key** represents what each token contains. When token "sky" is being considered as a potential match for another token's query, it uses its key vector to answer.

**Value** represents the actual information each token carries. Once the network decides which tokens are relevant (using Q and K), it uses the value vectors to collect that information.

The dot product between a query and all keys produces attention scores. High score means high relevance. Softmax converts those scores into weights. The weighted sum of values is the output.

![Same token representations projected into three different spaces: Query, Key and Value](/06-transformer/images/qkv_projections.png)

---

## Multi-head self-attention

This is the upgrade from Project 5. In Project 5, one attention head used the last hidden state as the query. In Project 6, four heads run in parallel and every token attends to every other token.

**Why multiple heads?**

Different heads learn different types of relationships simultaneously. One might learn that adjectives attend to the nouns they describe. Another might learn that verbs attend to their subjects. Running them in parallel gives the network a richer view than any single head could provide.

**Splitting into heads.**

All queries, keys and values are projected together then split.

```python
queries_per_head = all_queries.reshape(number_of_tokens, number_of_attention_heads, dimensions_per_attention_head)
keys_per_head    = all_keys.reshape(   number_of_tokens, number_of_attention_heads, dimensions_per_attention_head)
values_per_head  = all_values.reshape( number_of_tokens, number_of_attention_heads, dimensions_per_attention_head)
```

Before the reshape: shape (3, 64). After: shape (3, 4, 16). The 64 embedding dimensions are split into 4 groups of 16. Nothing moves. The numbers are the same. Numpy just looks at them differently.

**Each head computes its own attention.**

```python
for head_index in range(number_of_attention_heads):
    head_queries = queries_per_head[:, head_index, :]   # (3, 16)
    head_keys    = keys_per_head[   :, head_index, :]
    head_values  = values_per_head[ :, head_index, :]

    raw_attention_scores = np.dot(head_queries, head_keys.T)
    scaled_scores        = raw_attention_scores / math.sqrt(dimensions_per_attention_head)
    attention_weights    = softmax_along_last_axis(masked_scores)
    head_output          = np.dot(attention_weights, head_values)
```

**Concatenate and project.**

```python
concatenated_head_outputs = np.concatenate(attention_head_outputs, axis=-1)   # (3, 64)
attention_sublayer_output = np.dot(concatenated_head_outputs, attention_output_weights)
```

Four (3, 16) outputs concatenated give (3, 64). The output projection mixes information across all heads into one representation.

![Multi-head attention showing split into four heads, parallel computation, and concatenation](/06-transformer/images/multi_head_attention.png)

---

## The causal mask

The Transformer sees all tokens at once. Without a mask, each token could attend to future tokens, which would mean the model sees the answer before predicting it. The causal mask prevents this.

```python
causal_mask   = np.triu(np.full_like(scaled_scores, -1e9), k=1)
masked_scores = scaled_scores + causal_mask
```

`np.triu` with k=1 fills the upper triangle with negative infinity. After softmax these become zero. The result for a sequence of three tokens:

```
         token 0   token 1   token 2
token 0 [  0.xx      0         0    ]
token 1 [  0.xx      0.xx      0    ]
token 2 [  0.xx      0.xx      0.xx ]
```

Token 0 can only see itself. Token 1 can see tokens 0 and 1. Token 2 can see all three. No token looks forward.

This is the masking described in the Three Problems with Self-Attention article in the theory series. Here it is a numpy array applied in an actual forward pass.

![3x3 causal mask matrix showing allowed attention in teal and blocked future positions in ochre](/06-transformer/images/causal_mask.png)

---

## Residual connections

After attention, the original input is added back before layer normalisation.

```python
token_representations = apply_layer_normalisation(
    token_representations + attention_sublayer_output,
    layer_norm1_scale, layer_norm1_shift
)
```

That addition is the residual connection. The network learns what to add to the input rather than learning to replace it entirely.

Without residual connections, training a deep Transformer would fail. Gradients would vanish before reaching the early layers. Adding the input back creates a direct path for gradients to flow through the entire network. It is one of the most important architectural ideas in modern deep learning.

The same pattern applies after the feed forward network.

```python
token_representations = apply_layer_normalisation(
    token_representations + compressed_representations,
    layer_norm2_scale, layer_norm2_shift
)
```

![Residual connection diagram showing skip path bypassing the attention sublayer](/06-transformer/images/residual_connection.png)

---

## Layer normalisation

Layer normalisation is applied after each residual connection.

```python
def apply_layer_normalisation(token_representations, learned_scale, learned_shift, epsilon=1e-6):
    token_mean        = np.mean(token_representations, axis=-1, keepdims=True)
    token_variance    = np.var(token_representations,  axis=-1, keepdims=True)
    normalised_tokens = (token_representations - token_mean) / np.sqrt(token_variance + epsilon)
    return learned_scale * normalised_tokens + learned_shift
```

The mean and variance are computed across the embedding dimension for each token independently. The result is scaled and shifted by learned parameters. The small epsilon prevents division by zero.

Without layer normalisation, values flowing through a deep network can grow or shrink unpredictably across layers. Layer norm keeps them in a stable range at every step, making training faster and more reliable.

---

## Feed forward network

After attention, every token is processed independently through a small two-layer network.

```python
expanded_representations  = relu_activation(
    np.dot(token_representations, feedforward_expand_weights) + feedforward_expand_bias
)
compressed_representations = (
    np.dot(expanded_representations, feedforward_compress_weights) + feedforward_compress_bias
)
```

First layer expands from 64 to 128 dimensions. ReLU adds non-linearity. Second layer compresses back to 64.

Attention is how tokens communicate with each other. Feed forward is where each token processes that gathered information on its own. The expansion to 128 then compression back to 64 gives each token more expressive capacity to transform what it collected from attention.

Stacking purely linear operations collapses into a single linear operation. The ReLU breaks that collapse and ensures each layer can learn something the previous one cannot.

![Feed forward network showing 64 to 128 expansion with ReLU then compression back to 64](/06-transformer/images/feedforward_network.png)

---

## Dropout

Dropout appears in three places. After attention weights, after the attention output projection, and after the first feed forward layer.

```python
def apply_dropout(activations, dropout_rate, is_training=True):
    if not is_training or dropout_rate == 0:
        return activations
    keep_probability = 1 - dropout_rate
    dropout_mask     = np.random.binomial(1, keep_probability, activations.shape) / keep_probability
    return activations * dropout_mask
```

During training, 10 percent of activations are set to zero at random. The remaining values are scaled up to preserve the expected sum. During generation, dropout is turned off.

This is the first real solution to the overfitting problem we have been building toward since Project 2. The network cannot rely on any single pathway too heavily. It is forced to learn redundant representations that generalise better.

---

## Parameter count

```
word_embedding_matrix:          77  × 64  =  4,928
attention_query_weights:        64  × 64  =  4,096
attention_key_weights:          64  × 64  =  4,096
attention_value_weights:        64  × 64  =  4,096
attention_output_weights:       64  × 64  =  4,096
layer_norm1_scale + shift:       2  × 64  =    128
feedforward_expand_weights:     64  × 128 =  8,192
feedforward_expand_bias:             128  =    128
feedforward_compress_weights:  128  × 64  =  8,192
feedforward_compress_bias:            64  =     64
layer_norm2_scale + shift:       2  × 64  =    128
output_projection_weights:      64  × 77  =  4,928
output_projection_bias:               77  =     77
                                          -------
Total (numpy version):                     43,149
Total (PyTorch version):                   43,405
```

The PyTorch version has 256 more parameters. `nn.MultiheadAttention` adds bias terms to the Q, K, V projections by default. 4 × 64 = 256 extra parameters. Same architecture, small implementation difference.

---

## The PyTorch version

The numpy version showed the math. The PyTorch version trains it.

```python
class TransformerBlock(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, number_of_attention_heads,
                 feedforward_hidden_dim, dropout_rate, sequence_length):
        super(TransformerBlock, self).__init__()

        self.word_embedding = nn.Embedding(vocabulary_size, embedding_dim)

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

        word_embeddings       = self.word_embedding(word_indices_in_sequence)
        token_representations = word_embeddings + self.positional_encoding[:, :seq_len, :]

        causal_mask = self._build_causal_mask(seq_len)

        attention_output, _ = self.multihead_attention(
            query=token_representations,
            key=token_representations,
            value=token_representations,
            attn_mask=causal_mask
        )

        token_representations = self.layer_norm_after_attention(
            token_representations + attention_output
        )

        feedforward_output = self.feedforward_expand(token_representations)
        feedforward_output = self.feedforward_relu(feedforward_output)
        feedforward_output = self.feedforward_dropout(feedforward_output)
        feedforward_output = self.feedforward_compress(feedforward_output)

        token_representations = self.layer_norm_after_feedforward(
            token_representations + feedforward_output
        )

        last_token_representation = token_representations[:, -1, :]
        output_scores             = self.output_projection(last_token_representation)

        return output_scores
```

The forward method reads like a description of the architecture. Embed, add position, attend, normalise, feed forward, normalise, project. The residual connections are still explicitly written as additions. Everything else is handled by PyTorch.

`nn.MultiheadAttention` replaces the entire head-splitting, scoring, masking, concatenating loop from the numpy version. Because we already wrote that loop by hand, this abstraction is now transparent rather than opaque.

---

## Training results

```
Using device: cpu
Vocabulary size: 77
Training sequences: 163
Total parameters: 43,405

Epoch     0  loss: 4.4567
Epoch   200  loss: 0.0252
Epoch   400  loss: 0.0816
Epoch   600  loss: 0.0113
Epoch   800  loss: 0.0122
```

Loss drops from 4.4567 to the 0.01 range. The slight oscillation between epoch 200 and 400 is normal with Adam on a small dataset. The network is settling between slightly different minima.

![Training loss curve showing drop from 4.4567 to 0.01 range over 1000 epochs](/06-transformer/images/loss_curve.png)

---

## Generated text

```
the sky is   -> cloudy today bring your umbrella when
bring your umbrella -> when it rains dark clouds mean
dark clouds mean    -> heavy rain the weather looks wet
the rain will       -> stop by evening sunny weather makes
a clear sky         -> means no rain today carry an
```

Every seed chains correctly into the next training sentence. Compare this to the numpy version which collapsed to "the the the the". That gap is the difference between training only the output layer versus training all 43,405 parameters.

---

## What I took away from this

Writing the causal mask as a numpy array and watching it zero out the upper triangle was the moment where the masking article from the theory series became concrete. Writing the residual connection as `token_representations + attention_sublayer_output` made the skip path tangible in a way that a diagram never quite did.

The Transformer block is not a collection of complicated operations. It is four simple ideas arranged carefully. Attention so tokens can communicate. Feed forward so each token can process what it gathered. Residuals so gradients can flow. Layer norm so values stay stable.

Each idea is simple. The combination is powerful.

Project 7 stacks multiple blocks into a complete language model, adds a proper training loop, and trains it end to end. That is the capstone of the build series.
