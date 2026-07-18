# 07: Building a Mini Language Model

![Mini LLM architecture overview](/07-mini-LLM/images/cover.png)

Six projects ago we built a single neuron. It multiplied three numbers by three weights and produced one output. No memory, no language, no sense of sequence.

This project is the capstone of the series. Four stacked Transformer blocks, 159,558 parameters, trained on an expanded corpus to generate coherent text. The architecture is identical to GPT at a much smaller scale. The same building blocks, the same structure, the same training loop. The only difference between this and the models powering large AI systems is scale.

---

## What this project adds on top of Project 6

Project 6 built and trained a single Transformer block. This project extends that in three specific directions:

**Stacking multiple blocks.** Instead of one Transformer block, the model has four stacked in sequence. The output of each block feeds directly into the input of the next. Each block does another full round of attention and feed-forward processing, letting the model build progressively more abstract representations of the input sequence.

**Proper batching.** Instead of processing one training sequence at a time (as in all previous projects), the model now processes 32 sequences per weight update. This produces a much more stable training signal and is the standard approach for training any real neural network.

**Cosine annealing.** The learning rate starts at 0.001 and gradually decays to near zero over the course of training following a cosine curve. This allows the model to take large steps early in training and settle precisely at the end, without manually choosing when to reduce the rate.

These three changes collectively make deep training work reliably. A four-block model trained one sequence at a time with a fixed learning rate would oscillate and converge poorly. With batching and cosine annealing, it converges smoothly to near-zero loss.

---

## The expanded corpus

The weather corpus used in Projects 4 through 6 had 30 sentences and 77 unique words. For a model with 159,558 parameters, that is far too small. A model this large will simply memorise 30 sentences during the first few epochs and stop learning anything useful.

The expanded corpus (`weather_corpus_v2.txt`) has 91 sentences, 503 total words, and 198 unique vocabulary items. It covers a wider range of weather phenomena: fog, frost, hail, lightning, snow, thunder, rainbows, seasons, and temperature changes.

```text
the forecast says rain all week
morning fog covers the valley
thunder follows lightning in storms
the temperature drops before rain
spring rain feeds the flowers
autumn clouds bring cool wind
...
```

The sequence length is also increased from 3 to 4. The model now sees four words before predicting the fifth, giving it slightly more context per prediction.

---

## The model architecture

The full model is two classes. `TransformerBlock` is the same block from Project 6, used here as a repeating unit. `MiniLanguageModel` stacks four of them and adds the surrounding structure.

```python
class MiniLanguageModel(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, number_of_attention_heads,
                 feedforward_hidden_dim, number_of_blocks, dropout_rate, max_sequence_length):
        super(MiniLanguageModel, self).__init__()

        self.word_embedding = nn.Embedding(vocabulary_size, embedding_dim)

        positional_encoding = self._compute_positional_encoding(
            max_sequence_length, embedding_dim
        )
        self.register_buffer('positional_encoding', positional_encoding)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                number_of_attention_heads=number_of_attention_heads,
                feedforward_hidden_dim=feedforward_hidden_dim,
                dropout_rate=dropout_rate
            )
            for _ in range(number_of_blocks)
        ])

        self.final_layer_norm  = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocabulary_size)
        self.embedding_dropout = nn.Dropout(dropout_rate)
```

### Why nn.ModuleList matters

`nn.ModuleList` is the key new pattern here. It is how you tell PyTorch about a collection of submodules.

When you call `model.parameters()` to get everything the optimiser should update, PyTorch walks the model's registered components. A plain Python list of blocks (`self.transformer_blocks = [block1, block2, block3, block4]`) is invisible to PyTorch's parameter tracker. Those blocks' weights would never get updated.

`nn.ModuleList` registers each block as a proper submodule. PyTorch includes all of their parameters when you call `model.parameters()`, and they all get updated during training.

### The forward pass

```python
def forward(self, word_indices_in_sequence):
    seq_len = word_indices_in_sequence.shape[1]

    # Step 1: embed words and add position information
    word_embeddings       = self.word_embedding(word_indices_in_sequence)
    token_representations = word_embeddings + self.positional_encoding[:, :seq_len, :]
    token_representations = self.embedding_dropout(token_representations)

    # Step 2: build causal mask once, reuse across all blocks
    causal_mask = self._build_causal_mask(seq_len)

    # Step 3: pass through all Transformer blocks in sequence
    for transformer_block in self.transformer_blocks:
        token_representations = transformer_block(token_representations, causal_mask)

    # Step 4: final normalisation
    token_representations = self.final_layer_norm(token_representations)

    # Step 5: take the last token and project to vocabulary scores
    last_token    = token_representations[:, -1, :]
    output_scores = self.output_projection(last_token)
    return output_scores
```

The first block receives the raw word embeddings plus positional encoding. The second block receives what the first block produced. The third receives what the second produced. And so on. By the time the representations reach the output projection, they have passed through four full rounds of multi-head attention and feed-forward processing.

Each block builds on the work of the previous one. The first block might learn basic word associations. The second might learn phrase-level patterns. The third and fourth can represent more abstract relationships that span the full sequence. This is the core reason why stacking blocks improves model quality.

![Four stacked Transformer blocks with token representations flowing through each sequentially](/07-mini-LLM/images/stacked_transformer_blocks.png)

---

## GELU: why the activation function changed

Project 6 used ReLU in the feed-forward network. This project uses GELU.

```python
# Project 6
self.feedforward_activation = nn.ReLU()

# Project 7
self.feedforward_activation = nn.GELU()
```

**ReLU** is a hard cutoff at zero. Any input below zero becomes exactly zero. The derivative at those points is also exactly zero, meaning those neurons produce no gradient signal and contribute nothing to learning for that step.

**GELU** (Gaussian Error Linear Unit) is a smooth approximation. Negative values are not cut to zero but tapered gently. A small negative input produces a small negative output rather than nothing. This preserves gradient information through more of the network.

In practice, GELU trains better than ReLU for deep language models because the smoother gradient flow helps weight updates reach all parts of a multi-block network. GPT-2, GPT-3, BERT, and most modern Transformers use GELU as their standard activation.

![GELU smooth curve versus ReLU hard cutoff at zero showing gradient preservation at negative values](/07-mini-LLM/images/gelu_vs_relu.png)

---

## Batching: why one sequence at a time is not enough

In Projects 1 through 6, we fed training data one sequence at a time. Each weight update was based on a single example.

For a small network solving a simple problem, this works. For a four-block network with 159,558 parameters learning from a corpus of 499 sequences, it breaks down. The gradient computed from a single 4-word sequence is too noisy to be informative. The model cannot tell signal from noise and training oscillates rather than converges.

**Batching** solves this by computing the gradient over many sequences simultaneously and averaging.

```python
batch_size  = 32
permutation = torch.randperm(sequences_tensor.shape[0])   # shuffle indices

for batch_start in range(0, sequences_tensor.shape[0], batch_size):
    batch_indices   = permutation[batch_start : batch_start + batch_size]
    batch_sequences = sequences_tensor[batch_indices]
    batch_targets   = targets_tensor[batch_indices]

    optimiser.zero_grad()
    output_scores = model(batch_sequences)                 # process 32 at once
    loss          = loss_function(output_scores, batch_targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimiser.step()
```

`torch.randperm` shuffles the training indices at the start of each epoch so the model sees sequences in a different order every time. This prevents it from learning the order of the training data rather than the patterns within the data.

**Gradient clipping** (`clip_grad_norm_`) caps the total gradient magnitude at 1.0. In deep networks, gradients can occasionally grow very large, causing a single update to overshoot the minimum by a large amount and destabilise training. Clipping prevents this. The direction of the gradient is preserved; only its magnitude is capped if it exceeds the threshold.

![One sequence at a time versus batch of 32 showing the averaged gradient](/07-mini-LLM/images/batching_comparison.png)

---

## Cosine annealing: automatically reducing the learning rate

With a fixed learning rate, the model risks two failure modes. If the rate is too high, it overshoots the minimum and never settles. If it is too low, training proceeds too slowly.

**Cosine annealing** solves this by automatically reducing the learning rate over time:

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=number_of_epochs)

# Called once per epoch, after the weight update
scheduler.step()
```

The learning rate follows a cosine curve from its starting value (0.001) down to near zero over the total number of epochs:

```text
Epoch     0   lr: 0.001000
Epoch   400   lr: 0.000904
Epoch   800   lr: 0.000654
Epoch  1200   lr: 0.000345
Epoch  1600   lr: 0.000095
Epoch  2000   lr: ~0.000000
```

Early in training, the model is far from a good solution and the large learning rate lets it move quickly. Late in training, the model is close to the minimum and the small learning rate lets it settle precisely without overshooting.

A useful mental model: finding a parking space. You drive fast when you are far away and slow down as you get close. Cosine annealing does this automatically without needing to manually decide when to reduce the rate.

![Cosine annealing learning rate curve showing smooth decay from 0.001 to near zero over 2000 epochs](/07-mini-LLM/images/cosine_annealing.png)

---

## Parameter count

```text
Component                               Shape          Parameters
-----------------------------------------------------------------
word_embedding_matrix                   198 x 64           12,672

Per Transformer block (x 4):
  attention Q, K, V, O weights          4 x 64 x 64        16,384
  attention biases                      4 x 64                256
  layer norm 1 (scale + shift)          2 x 64                128
  feedforward expand weights            64 x 128            8,192
  feedforward expand bias               128                   128
  feedforward compress weights          128 x 64            8,192
  feedforward compress bias             64                     64
  layer norm 2 (scale + shift)          2 x 64                128
  subtotal per block                                        33,472
  total across 4 blocks                                    133,888

final layer norm (scale + shift)        2 x 64                128
output projection weights               64 x 198           12,672
output projection bias                  198                   198
-----------------------------------------------------------------
Total                                                      159,558
```

For perspective across the series:

```text
Project 1 (single neuron):         3 weights + 1 bias =           4
Project 2 (two-layer network):                              ~  250
Project 4 (word-level RNN):                                  14,093
Project 6 (one Transformer block):                           43,405
Project 7 (four Transformer blocks):                        159,558
GPT-2 small:                                            117,000,000
GPT-3:                                              175,000,000,000
```

The ideas are the same across the entire range. The scale is what differs.

---

## Training results

```text
Using device: cpu
Vocabulary size:    198
Total words:        503
Training sequences: 399
Validation sequences: 100
Total parameters:   159,558

Epoch     0   train loss: 5.0619   val loss: 5.1204   lr: 0.001000
Epoch   400   train loss: 0.0027   val loss: 0.1832   lr: 0.000904
Epoch   800   train loss: 0.0011   val loss: 0.1541   lr: 0.000654
Epoch  1200   train loss: 0.0009   val loss: 0.1380   lr: 0.000345
Epoch  1600   train loss: 0.0001   val loss: 0.1201   lr: 0.000095

Final training loss: 0.0000
```

The loss drops from 5.06 to essentially zero on the training set. The validation loss drops from 5.12 to around 0.12, which is much higher than the training loss. This gap is overfitting: the model has memorised the training sequences far better than it can generalise to unseen ones.

![Mini LLM training loss curve showing steep drop in first 100 epochs then flat line near zero](/07-mini-LLM/images/loss_curve.png)

---

## Understanding the starting loss

The first epoch loss of 5.06 was not predicted in advance. It came from the first forward pass with random weights.

For a completely uninformed model predicting over a vocabulary of 198 words, the expected loss is:

```text
-log(1/198) = log(198) = 5.29
```

This is the cross-entropy loss you get when all 198 words are assigned equal probability and the correct word has probability 1/198. It is the theoretical floor for a model that knows nothing.

The actual first-epoch loss of 5.06 is slightly below this baseline for two reasons. Random weight initialisation introduces small non-uniformities in the output distribution even before any training happens, nudging some words slightly higher than others. The first epoch also averages loss across all training sequences, and some sequences happen to be easier than others, pulling the average slightly below the theoretical maximum.

The dotted reference line in the loss curve plot marks `log(198) = 5.29`. Everything below that line represents genuine learning beyond random guessing.

---

## The shape of the loss curve

The most striking feature of the loss curve is how front-loaded the learning is. Almost all of the improvement happens in the first 100 epochs. The curve drops steeply like a cliff, then hugs near zero for the remaining 1900 epochs.

This pattern (called diminishing returns in gradient descent) appeared in Project 2 with 800 versus 5000 epochs on a toy problem. It is the same phenomenon at larger scale. The model makes large improvements quickly when it is far from the minimum, then makes smaller and smaller improvements as it approaches the optimal weights.

The cosine annealing scheduler and gradient clipping kept the curve smooth throughout with no oscillations. Compare this to what a fixed learning rate without gradient clipping would produce on a four-block network: visible noise and occasional spikes as large gradients cause overshooting.

---

## Text generation with temperature

After training, the model generates text by reading a seed sequence and predicting one word at a time. The key new concept here is **temperature**, which controls how random the predictions are.

```python
def generate_text(seed_words, number_of_words_to_generate=8, temperature=1.0):
    model.eval()
    generated_words = seed_words.copy()

    with torch.no_grad():
        for _ in range(number_of_words_to_generate):
            context_words   = generated_words[-sequence_length:]
            context_indices = [word_to_index.get(w, 0) for w in context_words]
            sequence_tensor = torch.tensor(context_indices).unsqueeze(0).to(device)

            output_scores = model(sequence_tensor)

            if temperature == 0.0:
                # Greedy: always pick the most likely word
                predicted_index = torch.argmax(output_scores, dim=-1).item()
            else:
                # Sampling: divide scores by temperature before softmax
                scaled_scores   = output_scores / temperature
                probabilities   = torch.softmax(scaled_scores, dim=-1)
                predicted_index = torch.multinomial(probabilities, num_samples=1).item()

            generated_words.append(index_to_word[predicted_index])

    return ' '.join(generated_words)
```

**Temperature = 0.0 (greedy):** always picks the single most probable word. Deterministic. Produces the same output every time for the same seed. Can get stuck in repetitive loops if the model is very confident about a small set of words.

**Temperature = 1.0 (standard sampling):** samples from the probability distribution as computed. Introduces randomness proportional to the model's uncertainty.

**Temperature < 1.0 (conservative):** dividing scores by a number less than 1 amplifies the differences between high and low scores before softmax. The highest-probability words become even more dominant. The output is less random and more predictable.

**Temperature > 1.0 (creative):** dividing scores by a number greater than 1 flattens the differences. Low-probability words become more likely to be selected. The output is more varied but less coherent.

Sample output:

```text
Greedy (temperature=0.0):
  the sky is cloudy today bring your umbrella when it rains dark
  bring your umbrella when it rains dark clouds mean heavy rain the
  dark clouds mean heavy rain the weather looks wet outside a clear

Temperature=0.8:
  the sky is cloudy today bring your umbrella when it rains dark
  bring your umbrella when it rains dark clouds mean heavy rain the
  the storm is moving closer rain is falling on the street a
```

The greedy and temperature=0.8 outputs are nearly identical. When the model's loss is near zero it is extremely confident about each next word, and even sampling with moderate temperature reliably picks the top choice.

---

## The overfitting arc, completed

Every project in this series has touched on overfitting from a different angle. Here is the full picture:

**Project 2** planted the question. Two experiments, 800 epochs versus 5000. Both got every prediction correct, but the 5000-epoch model was more confident. More training on a tiny dataset leads to tighter memorisation.

**Project 3** introduced the validation split: separate training loss and validation loss, so we can see when one falls while the other rises.

**Project 4** showed overfitting in generated text. The RNN reproducing training phrases almost verbatim rather than generalising.

**Project 5** showed the gap quantitatively. A training loss of 0.01 on 163 examples is not learning, it is memorisation.

**Project 6** introduced dropout as the first structural countermeasure. Randomly zeroing 10% of activations during training prevents any single pathway from being relied on too heavily.

**Project 7** adds batching, gradient clipping, and cosine annealing. And the model still memorises the corpus. Training loss reaches 0.0000 while validation loss stays around 0.12.

The honest conclusion: 159,558 parameters trained on 399 sequences of 4 words each will always overfit. The techniques introduced across this series genuinely help (without them the training curve oscillates and the validation loss is worse) but they cannot solve the fundamental mismatch between model capacity and dataset size. The real solution is more training data.

GPT-3 was trained on 45 billion tokens. This model was trained on roughly 1,600 words. The architecture is the same. The scale is not.

---

## What each PyTorch abstraction refers to

After seven projects written from scratch, every line in the PyTorch version has a concrete referent:

```text
nn.Embedding            one-hot row lookup from Project 4
nn.MultiheadAttention   Q, K, V loop with masking from Projects 5 and 6
nn.LayerNorm            mean and variance normalisation from Project 6
nn.GELU                 smooth activation replacing the ReLU from Project 6
nn.ModuleList           the container that makes stacking blocks trainable
loss.backward()         chain rule backwards through all layers from Project 2
CosineAnnealingLR       automatic learning rate schedule, new in this project
clip_grad_norm_         gradient magnitude cap, introduced in Project 6
```

None of it is magic. All of it is math we built by hand at some point in the series, now automated and arranged.

---

## Running

```bash
pip install torch matplotlib
python mini_llm.py
```

The script trains for 2000 epochs, printing loss and learning rate every 200 epochs, then generates text at temperature 0.0 and 0.8, and saves the loss curve to `loss_curve.png`.

## Files

```text
mini_llm.py              main training script (4 blocks, GELU, batching, cosine annealing)
mini_llmv2.py            extended version with additional experiments
weather_corpus_v2.txt    expanded 91-sentence training corpus
config.json              hyperparameters (embedding dim, heads, blocks, epochs, batch size)
images/
  cover.png
  stacked_transformer_blocks.png
  gelu_vs_relu.png
  batching_comparison.png
  cosine_annealing.png
  loss_curve.png
```
