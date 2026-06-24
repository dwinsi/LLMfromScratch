# Building a Mini Language Model

![Cover](/07-mini-LLM/images/cover.png)

Project 7 in my build series at [Github repo](github.com/dwinsi/LLMfromScratch)

Six projects ago I built a single neuron. It took a few numbers, multiplied them by weights, and produced one output. That neuron had no memory, no sense of sequence, no understanding of language.

This project trains a language model. Four stacked Transformer blocks, 159,558 parameters, trained on an expanded weather corpus. The architecture is identical to GPT at a much smaller scale. The same ideas. The same structure. The same training loop.

The only difference between this and the models powering ChatGPT and Claude is scale.

---

## What changed from Project 6

Project 6 built one Transformer block and trained it in isolation. Project 7 stacks four of those blocks into a complete language model and adds three things that make deep training work properly.

**Batching.** Instead of feeding one training sequence at a time, the model now processes 32 sequences per weight update. Each gradient is averaged across 32 examples rather than computed from a single one. This produces a much more stable training signal and dramatically faster convergence.

**Cosine annealing.** The learning rate starts at 0.001 and gradually reduces to near zero following a cosine curve over 2000 epochs. Early training takes large steps toward the minimum. Late training takes tiny precise ones, settling into the best weights the model can find.

**GELU activation.** The feed forward network now uses GELU instead of ReLU. GELU tapers negative values smoothly rather than cutting them to exactly zero. It is the standard activation in GPT, BERT and every modern LLM because it produces better gradients in deep networks.

These three changes are the difference between a model that barely learns and one that converges to near-zero loss.

---

## The expanded corpus

The weather corpus from Projects 4 through 6 had 30 sentences and 77 unique words. For a model with 159,558 parameters, that is far too small. The model memorises the training data before it has a chance to learn anything useful.

The expanded corpus has 91 sentences, 503 words and 198 unique vocabulary. It covers the full range of weather phenomena: fog, frost, hail, lightning, snow, thunder, rainbows, seasons and temperature changes. Richer vocabulary gives the model more patterns to learn and more relationships to discover between words.

```text
the forecast says rain all week
morning fog covers the valley
thunder follows lightning in storms
the temperature drops before rain
spring rain feeds the flowers
autumn clouds bring cool wind
...
```

Sequence length is now 4. The model sees four words before predicting the fifth. More context per prediction compared to the three-word sequences used in earlier projects.

---

## The architecture

The full model is two classes. `TransformerBlock` is identical to Project 6. `MiniLanguageModel` stacks four of them and adds the surrounding structure.

```python
class MiniLanguageModel(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, number_of_attention_heads,
                 feedforward_hidden_dim, number_of_blocks, dropout_rate, max_sequence_length):
        super(MiniLanguageModel, self).__init__()

        self.word_embedding   = nn.Embedding(vocabulary_size, embedding_dim)

        # Positional encoding registered as buffer, moves to GPU automatically
        positional_encoding = self._compute_positional_encoding(
            max_sequence_length, embedding_dim
        )
        self.register_buffer('positional_encoding', positional_encoding)

        # Four Transformer blocks registered as a ModuleList
        # ModuleList tracks all parameters automatically
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

**`nn.ModuleList`** is the key new pattern here. Registering the blocks as a ModuleList tells PyTorch to track all their parameters as part of the model. When you call `model.parameters()`, every weight across all four blocks is included. Without ModuleList, PyTorch would not know these blocks exist and would not update their weights during training.

The forward pass loops through all blocks sequentially:

```python
def forward(self, word_indices_in_sequence):
    seq_len = word_indices_in_sequence.shape[1]

    word_embeddings       = self.word_embedding(word_indices_in_sequence)
    token_representations = word_embeddings + self.positional_encoding[:, :seq_len, :]
    token_representations = self.embedding_dropout(token_representations)

    causal_mask = self._build_causal_mask(seq_len, word_indices_in_sequence.device)

    for transformer_block in self.transformer_blocks:
        token_representations = transformer_block(token_representations, causal_mask)

    token_representations = self.final_layer_norm(token_representations)

    last_token_representation = token_representations[:, -1, :]
    return self.output_projection(last_token_representation)
```

Each block receives the output of the previous one. The first block sees the raw embeddings plus positional encoding. The second block sees what the first block produced. The third sees what the second produced. And so on. By the time the representations reach the output projection they have passed through four rounds of attention and feed forward processing.

![Four stacked Transformer blocks with token representations flowing through each sequentially](/07-mini-LLM/images/stacked_transformer_blocks.png)

---

## GELU vs ReLU

One change from Project 6 worth explaining properly.

```python
# Project 6
self.feedforward_activation = nn.ReLU()

# Project 7
self.feedforward_activation = nn.GELU()
```

ReLU is a hard cutoff. Any input below zero becomes exactly zero. The derivative at those points is exactly zero, which means those neurons contribute nothing to the gradient.

GELU is a smooth approximation. Negative values are not cut to zero but instead tapered gently. A small negative input produces a small negative output rather than nothing. This preserves gradient information through more of the network and is why GELU tends to train better than ReLU for deep language models.

GPT-2, GPT-3, BERT, and most modern Transformers use GELU. We used ReLU in Project 6 to keep it simple. Now that we are building the full model, the standard choice is the right one.

![GELU smooth curve versus ReLU hard cutoff showing gradient preservation at negative values](/07-mini-LLM/images/gelu_vs_relu.png)

---

## Batching and the training loop

The previous projects fed one sequence at a time. This works for small networks but breaks down for deeper ones. With four stacked blocks and 159,558 parameters, a gradient computed from a single 4-word sequence is too noisy to be useful. The model cannot tell signal from noise.

Batching fixes this by computing the gradient over 32 sequences simultaneously and averaging.

```python
batch_size  = 32
permutation = torch.randperm(sequences_tensor.shape[0])

for batch_start in range(0, sequences_tensor.shape[0], batch_size):
    batch_indices   = permutation[batch_start : batch_start + batch_size]
    batch_sequences = sequences_tensor[batch_indices]
    batch_targets   = targets_tensor[batch_indices]

    optimiser.zero_grad()
    output_scores = model(batch_sequences)
    loss          = loss_function(output_scores, batch_targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimiser.step()
```

`torch.randperm` shuffles the training indices each epoch so the model sees sequences in a different order every time. This prevents it from learning the order of the training data rather than the patterns within it.

Gradient clipping remains. In a four-block network, gradients can still become large. Capping them at 1.0 keeps training stable throughout.

![One sequence at a time versus batch of 32 showing stable averaged gradient](/07-mini-LLM/images/batching_comparison.png)

---

## The cosine annealing scheduler

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=number_of_epochs)
```

After each epoch, `scheduler.step()` reduces the learning rate following a cosine curve. The learning rate starts at 0.001 and gradually decreases to near zero by epoch 2000.

The actual values from training:

```text
Epoch     0  lr: 0.001000
Epoch   400  lr: 0.000904
Epoch   800  lr: 0.000654
Epoch  1200  lr: 0.000345
Epoch  1600  lr: 0.000095
```

Early epochs take large steps because the model is far from a good solution and needs to move quickly. Late epochs take small steps because the model is close to the minimum and needs to settle precisely rather than overshoot.

This is the same principle as finding a parking space. You drive quickly when you are far away and slow down as you get close.

![Cosine annealing learning rate curve showing smooth decay from 0.001 to near zero over 2000 epochs](/07-mini-LLM/images/cosine_annealing.png)

---

## Parameter count

```text
word_embedding_matrix:         198 × 64  =  12,672

Per Transformer block (× 4):
  attention weights (Q K V O):  4 × 64 × 64  = 16,384
  attention biases:              4 × 64       =    256
  layer norm 1:                  2 × 64       =    128
  feedforward expand:            64 × 128     =  8,192
  feedforward expand bias:       128          =    128
  feedforward compress:          128 × 64     =  8,192
  feedforward compress bias:     64           =     64
  layer norm 2:                  2 × 64       =    128
  subtotal per block:                         = 33,472
  total across 4 blocks:                      = 133,888

final layer norm:                2 × 64       =    128
output projection:               64 × 198     = 12,672
output bias:                     198          =    198
                                              --------
Total:                                         159,558
```

Project 1 had one neuron. Project 6 had 43,405 parameters. Project 7 has 159,558. The jump from one block to four accounts for most of the increase. Each additional block adds another 33,472 parameters and another round of attention and processing.

---

## Training results

```text
Using device: cpu
Vocabulary size:    198
Total words:        503
Training sequences: 499
Total parameters:   159,558

Epoch     0  loss: 5.0619  lr: 0.001000
Epoch   400  loss: 0.0027  lr: 0.000904
Epoch   800  loss: 0.0011  lr: 0.000654
Epoch  1200  loss: 0.0009  lr: 0.000345
Epoch  1600  loss: 0.0001  lr: 0.000095

Final loss: 0.0000
```

Loss drops from 5.06 to essentially zero. The random baseline for 198 vocabulary is log(198) = 5.29, so the model starts just below random guessing and converges all the way to near-perfect prediction on the training data.

The most striking thing about the loss curve is the shape. Almost all of the learning happens in the first 100 epochs. The curve drops steeply like a cliff then hugs near zero for the remaining 1900 epochs. The cosine scheduler and gradient clipping kept the curve smooth throughout with no oscillations.

---

## Where the initial loss of 5.06 comes from

The starting loss was not calculated in advance. It came from the first forward pass of the model with random weights.

For a randomly initialised model predicting over a vocabulary of 198 words, the expected loss is:

```text
log(vocabulary_size) = log(198) = 5.29
```

This is the cross-entropy loss of a model that assigns equal probability to every word. If all 198 words are equally likely, the probability of the correct one is 1/198, and:

```text
-log(1/198) = log(198) = 5.29
```

That is the theoretical random baseline. A model that knows nothing should produce exactly this.

The actual first epoch loss was 5.06, slightly below 5.29. Two reasons for the small gap.

Weight initialisation is not perfectly uniform. Random weights from a normal distribution create slight biases toward some outputs over others from the very first forward pass.

The first epoch averages loss across all 499 sequences in random batches. Some sequences are easier than others, pulling the average slightly below the theoretical maximum.

The dashed red line in the loss curve plot marks exactly this baseline at log(198) = 5.29. When the curve sits below that line from the very first epoch, it means the model is already very slightly better than random before any real learning has happened. Everything below that line is genuine learning.

---

## The shape of the loss curve

The most striking thing about the loss curve is the shape. Almost all of the learning happens in the first 100 epochs. The curve drops steeply like a cliff then hugs near zero for the remaining 1900 epochs. The cosine scheduler and gradient clipping kept the curve smooth throughout with no oscillations.

![Mini LLM training loss curve showing steep drop in first 100 epochs then flat line near zero](/07-mini-LLM/images/loss_curve.png)

---

## Generated text

```text
Greedy (temperature=0.0):
  the sky is cloudy today bring your umbrella when it rains dark
  bring your umbrella when it rains dark clouds mean heavy rain the
  dark clouds mean heavy rain the weather looks wet outside a clear
  the rain will stop by evening sunny weather makes people happy dark
  a clear sky means no rain wind and clouds bring storms heavy

Temperature=0.8:
  the sky is cloudy today bring your umbrella when it rains dark
  bring your umbrella when it rains dark clouds mean heavy rain the
  the storm is moving closer rain is falling on the street a
```

The model chains correctly from one training sentence into the next. "The sky is cloudy today" flows into "bring your umbrella when it rains" flows into "dark clouds mean heavy rain". The model is using its context window to navigate the corpus rather than just predicting one word at a time independently.

The greedy and temperature=0.8 outputs are nearly identical. This is a consequence of the loss reaching near zero. The model is so confident about each next word that even with 0.8 temperature the most likely word always wins. You would need temperature above 1.5 to see real variation, at which point coherence starts to break down.

---

## The overfitting arc closes

Across seven projects we have watched overfitting develop and then address it.

Project 2 planted the question. Two experiments, 800 epochs versus 5000. Both correct but different confidence.

Project 3 introduced the validation split. Training loss and validation loss tracked separately.

Project 4 showed overfitting in generated text. The RNN reproducing training phrases exactly.

Project 5 made it explicit. Loss of 0.01 on a 163-example training set is memorisation.

Project 6 introduced dropout as the first structural solution.

Project 7 adds batching, gradient clipping and cosine annealing. And still the model memorises the corpus because 159,558 parameters on 499 training sequences is a large model for a small dataset.

The honest conclusion: overfitting on a dataset this small is inevitable. The solutions introduced across this series (dropout, gradient clipping, learning rate scheduling) help but do not eliminate it. The real solution is more training data. GPT-3 trained on 45 billion tokens. Our model trained on 499 sequences of 4 words each.

The architecture is the same. The scale is not.

---

## What I took away from this

Starting from a single neuron and arriving here took seven projects and a lot of code written by hand. Every abstraction in PyTorch now has a concrete referent in something I built from scratch.

`nn.MultiheadAttention` is the Q, K, V loop from Project 5. `nn.LayerNorm` is the mean and variance calculation from Project 6. `nn.Embedding` is the one-hot row lookup from Project 4. `loss.backward()` is the chain rule applied backwards through every layer, which I wrote manually in Project 2.

None of it is magic. All of it is math I have already written, automated and arranged.

The series is not finished. There is more to explore. Mixture of Experts. Different tokenisation strategies. Fine-tuning. The gap between this 159,558 parameter model and the models powering real LLMs is enormous in scale but surprisingly small in ideas.

The ideas are here. The code is here. The foundation is built.
