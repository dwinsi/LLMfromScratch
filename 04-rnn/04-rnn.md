# 04: Building a Word-Level RNN From Scratch

The first three projects all worked on the same problem: a network deciding whether to bring an umbrella based on three weather numbers. That was classification. A fixed input, a single output, a yes or no decision.

This project is different. The network now reads a **sequence of words** and predicts what word comes next. That is language modelling, and it is the core task of every language model ever built, from the smallest RNN to GPT-4.

This is where the series starts to feel like building something real.

---

## What changes when we move to sequences

In the earlier projects, every input had the same shape (three numbers) and every output had the same shape (one number). Order did not matter. You could shuffle the inputs and the loss function would not care.

Language is different. Words have order. "the dog bit the man" means something completely different from "the man bit the dog". The network must be able to carry the meaning of earlier words forward as it reads each new word.

That is the problem a **Recurrent Neural Network (RNN)** is designed to solve.

An RNN processes words one at a time, left to right. At each step it has two inputs: the current word, and a **hidden state** that summarises everything it has read so far. After processing each word, it updates the hidden state. By the time it reaches the last word in the sequence, the hidden state contains a compressed representation of the whole sequence.

The output is then a probability distribution over the entire vocabulary: how likely is each word to come next?

Three other things change along with the architecture: how we represent words (one-hot encoding), how we measure the loss (cross-entropy instead of mean squared error), and how the output layer works (softmax instead of sigmoid). Each of these is explained in full below.

---

## The training corpus

Rather than use a standard benchmark dataset, this project uses a small custom corpus written specifically for this series: thirty short sentences about weather, rain, clouds, and umbrellas.

```text
the sky is cloudy today
bring your umbrella when it rains
dark clouds mean heavy rain
the weather looks wet outside
a clear sky means no rain
wind and clouds bring storms
heavy rain is coming soon
...
```

```text
Total words in corpus:     166
Unique words (vocabulary):  77
```

Small enough to train on a laptop in a few minutes. Repetitive enough that the RNN can actually learn patterns between words.

---

## Step 1: building the vocabulary

Before any training can happen, every word must be converted to a number. We do this by building a vocabulary: a sorted list of all unique words, each assigned an integer index.

```python
with open('weather_corpus.txt', 'r') as f:
    corpus_text = f.read().lower()

all_words       = corpus_text.split()
unique_words    = sorted(set(all_words))
vocabulary_size = len(unique_words)                  # 77

word_to_index = {word: idx for idx, word in enumerate(unique_words)}
index_to_word = {idx: word for idx, word in enumerate(unique_words)}
```

The `word_to_index` dictionary maps each word to its integer: "clouds" might become 11, "rain" might become 52, "umbrella" might become 68. The `index_to_word` dictionary is the reverse mapping, used at the end when we want to convert predicted indices back into readable words.

---

## Step 2: one-hot encoding

Having an integer per word is not enough. The network needs a vector it can multiply against a weight matrix. A single integer does not work for that.

**One-hot encoding** converts a word index into a vector of zeros with a single 1 at the position corresponding to that word. For a vocabulary of 77 words, every word gets its own vector of length 77.

The word "rain" at index 52 looks like this:

```text
position:  0   1   2  ...  51  52  53  ... 76
value:    [0,  0,  0, ...,  0,  1,  0, ..., 0]
                                ^
                           only this position is 1
```

Every word has its own unique vector like this, with the 1 at a different position.

```python
def one_hot_encode(word_index, vocabulary_size):
    one_hot_vector                = np.zeros((1, vocabulary_size))
    one_hot_vector[0, word_index] = 1
    return one_hot_vector
```

**Why one-hot encoding works mathematically:** when you multiply a one-hot vector by a weight matrix, the result is simply one row of that matrix (the row at the position of the 1). Every other row gets multiplied by zero and contributes nothing. The one-hot vector acts as a row selector.

```text
[0, 0, 1, 0, ..., 0]  x  W  =  row 2 of W
```

This means the weight matrix is effectively a lookup table: each row is a learned representation of one word. Modern frameworks like PyTorch skip the one-hot vector entirely and use `nn.Embedding`, which does a direct index lookup. The mathematical result is identical, but the direct lookup is much faster. We use one-hot here to keep the matrix multiplication visible.

---

## Step 3: building training sequences

The RNN learns by reading windows of consecutive words and predicting what comes next. With a sequence length of 3, every group of four consecutive words produces one training example: the first three words are the input, the fourth is the target.

```python
sequence_length = 3

for i in range(len(all_words) - sequence_length):
    sequence = all_words[i : i + sequence_length]
    target   = all_words[i + sequence_length]
    training_sequences.append([word_to_index[w] for w in sequence])
    training_targets.append(word_to_index[target])
```

This sliding window produces 163 training sequences from 166 words. A few examples:

```text
Input sequence               ->  Target word
['the', 'sky', 'is']         ->  'cloudy'
['sky', 'is', 'cloudy']      ->  'today'
['bring', 'your', 'umbrella']->  'when'
['dark', 'clouds', 'mean']   ->  'heavy'
```

The data is then split 80/20 into a training set and a validation set, following the same principle from Project 3: the network learns only from training sequences, and validation sequences measure how well it generalises.

---

## The RNN architecture

The RNN has **three weight matrices**, not two like the networks in Projects 1 through 3.

```text
weights_input_to_hidden   shape (77, 64)   word -> hidden state
weights_hidden_to_hidden  shape (64, 64)   previous hidden state -> new hidden state
weights_hidden_to_output  shape (64, 77)   hidden state -> next word scores
```

The third weight matrix, `weights_hidden_to_hidden`, is what makes this network recurrent. At every word in the sequence, the network reads not just the current word but also its own previous hidden state. This is how information from earlier words is carried forward.

---

## How many parameters does this network have?

Parameters are the numbers the network learns during training. Each weight matrix and bias vector contributes some.

```text
weights_input_to_hidden   (77 x 64):   4,928
weights_hidden_to_hidden  (64 x 64):   4,096
weights_hidden_to_output  (64 x 77):   4,928
bias_hidden               (1  x 64):      64
bias_output               (1  x 77):      77
                                       ------
Total:                                 14,093
```

The general formula for any fully connected layer is:

```text
parameters = (input_size x output_size) + output_size
              weight matrix               bias vector
```

For perspective on how this scales:

```text
Our RNN:          14,093 parameters
GPT-2 small:  117,000,000 parameters
GPT-3:    175,000,000,000 parameters
```

The ideas are identical across all of them. The scale is what differs.

---

## The activation function: why tanh instead of sigmoid

In Projects 1 through 3 we used sigmoid as the activation function for hidden layers. The RNN uses **tanh** instead. This is not an arbitrary choice.

The key difference is the output range:

```text
sigmoid: outputs values between 0 and 1   (always positive)
tanh:    outputs values between -1 and 1  (can be negative)
```

A comparison at the same input values:

```text
Input   sigmoid   tanh
 -3      0.047    -0.995
 -2      0.119    -0.964
 -1      0.269    -0.762
  0      0.500     0.000
  1      0.731     0.762
  2      0.881     0.964
  3      0.953     0.995
```

Three reasons tanh works better for the hidden state in an RNN:

**It is centred at zero.** Sigmoid always outputs positive values. In a single feedforward step this is fine. But in an RNN the hidden state is updated many times, once per word in the sequence. If every activation is always positive, the hidden state drifts monotonically in one direction and cannot represent the full range of patterns in the data. Tanh can be positive or negative, giving the hidden state more room to represent different kinds of information.

**It has stronger gradients near zero.** At x = 0, the tanh derivative is 1.0. At x = 0, the sigmoid derivative is 0.25. Gradients must travel backwards through each time step during training. With sigmoid, every step shrinks the gradient by at least 75 percent. With tanh, gradients near zero pass through nearly unchanged. This helps the network learn from earlier parts of the sequence.

**Its derivative is simple.** Like sigmoid, the tanh derivative is computed from the output value itself, so no extra computation is needed on the backward pass:

```text
tanh'(x) = 1 - tanh(x)^2

At x = 0:  1 - 0.000^2 = 1.000   (strongest gradient)
At x = 1:  1 - 0.762^2 = 0.420
At x = 2:  1 - 0.964^2 = 0.071
At x = 3:  1 - 0.995^2 = 0.010   (nearly saturated)
```

Note the same saturation problem exists at the extremes: when tanh outputs are close to +1 or -1, the derivative approaches zero and gradients vanish. This is why the hidden state is initialised to zeros at the start of each sequence: starting at zero keeps the network in the region where tanh is most sensitive and learning is fastest.

---

## The forward pass

For each training sequence, the RNN processes one word at a time. At each step, the hidden state is updated using both the current word and the previous hidden state:

```python
hidden_state = np.zeros((1, hidden_size))        # start with no memory

for word_index in sequence_indices:
    word_vector  = one_hot_encode(word_index, vocabulary_size)
    hidden_input = (np.dot(word_vector,  weights_input_to_hidden) +
                    np.dot(hidden_state, weights_hidden_to_hidden) +
                    bias_hidden)
    hidden_state = np.tanh(hidden_input)
```

After reading all three words, the hidden state is a 64-dimensional vector that represents the meaning of the sequence so far. This gets passed to the output layer:

```python
output_scores        = np.dot(hidden_state, weights_hidden_to_output) + bias_output
output_probabilities = softmax(output_scores[0])
```

The output is a probability distribution over all 77 vocabulary words. The word with the highest probability is the prediction.

---

## Softmax: turning scores into a probability distribution

The output layer produces 77 raw scores, one per word. These can be any number, positive or negative. Before we can interpret them as probabilities, we need to convert them into a proper distribution that sums to 1.

That is what **softmax** does.

**Why not sigmoid?** In Projects 1 through 3, sigmoid was applied to a single output neuron making a binary decision. It takes one number and outputs one probability. Here we have 77 output neurons and exactly one correct next word. We need all 77 outputs to compete against each other and sum to 1. Sigmoid treats each output independently and cannot do that.

```text
One output, binary decision     -> sigmoid
Many outputs, one correct answer -> softmax
```

The softmax formula:

```text
softmax(x_i) = exp(x_i) / sum of exp(x_j) for all j
```

Exponentiate every score, then divide each by the total. This guarantees all values are positive and sum to exactly 1.

A small example with three words:

```text
raw scores:    [2.1,   0.3,  -0.8]
exponentiated: [8.17,  1.35,  0.45]   sum = 9.97
after softmax: [0.82,  0.14,  0.05]   sum = 1.00
```

The word with the highest raw score gets the highest probability. Softmax also amplifies the differences: a score of 2.1 versus 0.3 becomes a probability of 0.82 versus 0.14. This is intentional. As training progresses, the network learns to assign much higher scores to correct words, and softmax makes those high scores dominate the distribution.

The code subtracts the maximum score before exponentiating to prevent numerical overflow when scores are very large:

```python
def softmax(scores):
    shifted_scores = scores - np.max(scores)
    exponentiated  = np.exp(shifted_scores)
    return exponentiated / np.sum(exponentiated)
```

Subtracting the maximum does not change the result mathematically (dividing numerator and denominator by the same value cancels out) but it keeps the numbers in a safe range.

---

## Cross-entropy loss

For a problem where the output is a probability distribution, **cross-entropy loss** works better than mean squared error.

The formula is simple: take the probability the network assigned to the correct word, take its logarithm, and negate it.

```python
correct_word_probability = output_probabilities[target_index]
loss                     = -np.log(correct_word_probability + 1e-8)
```

Why this works:

- If the network assigns a high probability (close to 1.0) to the correct word, then `-log(1.0) = 0`. Loss is small.
- If the network assigns a low probability (close to 0.0) to the correct word, then `-log(0.01) = 4.6`. Loss is large.
- The small constant `1e-8` prevents taking the log of exactly zero, which would be undefined.

A useful baseline: a completely random network assigning equal probability to all 77 words would give each word a probability of 1/77. The cross-entropy loss would be `-log(1/77) = log(77) = 4.34`. The starting loss we see in training confirms this: the network begins by guessing randomly.

---

## The backward pass

The backward pass for the output layer uses a clean shortcut that comes from combining the cross-entropy loss and softmax gradients together. Instead of deriving them separately, the combined gradient is simply:

```python
output_gradient                = output_probabilities.copy()
output_gradient[target_index] -= 1
```

This says: for every word except the correct one, the gradient is just its softmax probability (push it down). For the correct word, the gradient is its probability minus 1 (push it up toward 1.0). This is a standard result in calculus when cross-entropy loss and softmax are used together.

The hidden layer gradient applies the tanh derivative:

```python
hidden_gradient  = np.dot(output_gradient, weights_hidden_to_output.T)
hidden_gradient *= (1 - hidden_state ** 2)   # tanh derivative: 1 - tanh(x)^2
```

---

## Training results

```text
Epoch     0  train loss: 4.3386  val loss: 4.3501
Epoch   200  train loss: 3.6814  val loss: 3.7092
Epoch   400  train loss: 2.4480  val loss: 2.5113
Epoch   600  train loss: 2.0643  val loss: 2.1208
Epoch   800  train loss: 2.0097  val loss: 2.0834
Epoch  1000  train loss: 1.8358  val loss: 1.9201
```

The starting loss of 4.34 matches the theoretical random baseline of `log(77) = 4.34`, confirming the network starts with no knowledge. By epoch 1000 it has fallen to 1.84 on training data and 1.92 on validation data. Both curves fall together, which means the network is genuinely learning patterns rather than just memorising individual sequences.

---

## Generated text

After training, the network generates new text by reading a seed sequence and repeatedly predicting the next word:

```text
Seed: 'the sky is'
Generated: the sky is cloudy today the clouds

Seed: 'bring your umbrella'
Generated: bring your umbrella when it rains cold

Seed: 'dark clouds mean'
Generated: dark clouds mean heavy rain a carry

Seed: 'the rain will'
Generated: the rain will stop by the clouds

Seed: 'a clear sky'
Generated: a clear sky means no rain a
```

The first few words after each seed are coherent and match patterns in the training data. "bring your umbrella when it rains" is nearly verbatim from the corpus. "dark clouds mean heavy rain" is correct. "the rain will stop by" reads naturally.

After a few words, the generation starts to drift. "dark clouds mean heavy rain a carry" makes no sense. This is the **vanishing gradient problem** made visible: the hidden state can only carry context reliably for short distances. After a few words, the memory of the seed fades and the network falls back on the most frequently occurring patterns in the training data regardless of context.

---

## What the RNN still cannot do

**Long-range context.** The hidden state is a fixed-size vector (64 dimensions here). No matter how long the sequence, all previous context must be compressed into that same vector. Information from early in the sequence gets diluted and eventually lost. The network effectively has a short memory.

**True understanding vs pattern matching.** The generated text that looks coherent is mostly reproducing patterns seen in the training data. The network has not learned what "heavy rain" means. It has learned that the token sequence "dark clouds mean" is very frequently followed by "heavy" in the corpus. That is correlation, not comprehension.

Both of these limitations point toward the same solution: an **attention mechanism** that lets the network look back at any earlier word directly, without relying on the hidden state to carry that information forward. That is the subject of the next project in the series.

---

## Running

```bash
pip install numpy matplotlib
python 04-rnn.py
```

The script builds the vocabulary, creates training sequences, trains for 1000 epochs, prints loss every 200 epochs, generates five text samples, and saves a loss curve plot to `loss_curve.png`.

## Files

```text
04-rnn.py             training script with full RNN implementation
weather_corpus.txt    30-sentence training corpus
config.json           hyperparameters (hidden size, sequence length, epochs, learning rate)
images/
  loss_curve.png      training vs validation loss with random baseline
```
