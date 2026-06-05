# Building a Word-Level RNN From Scratch

*Project 4 in my build series at github.com/dwinsi/LLMfromScratch*

The first three projects all worked on the same problem. A neural network deciding whether to bring an umbrella based on three weather inputs. It was a good starting point but it was not language.

Project 4 changes that. The network is no longer making a single prediction from a fixed set of inputs. It is reading a sequence of words and predicting what comes next. That is the core task of every language model ever built.

This is where the series starts to feel like building something real.

---

## The custom dataset

Rather than use an existing benchmark dataset, I wrote a small weather corpus specifically for this series. Thirty short sentences about weather, rain, clouds and umbrellas. Simple vocabulary, clear patterns, directly connected to everything we have built so far.

```
the sky is cloudy today
bring your umbrella when it rains
dark clouds mean heavy rain
the weather looks wet outside
a clear sky means no rain
wind and clouds bring storms
heavy rain is coming soon
...
```

The full corpus is 166 words with a vocabulary of 77 unique words. Small enough to train on a laptop in a few minutes. Repetitive enough that the RNN can actually learn the patterns.

---

## From classification to sequence prediction

In Projects 1 through 3, the network took a fixed input (three numbers) and produced a fixed output (one number). That is classification.

In Project 4, the network takes a sequence of words and predicts the next word. The output is a probability distribution over the entire vocabulary. The word with the highest probability is the prediction.

This changes the structure of the problem in three important ways.

The input is no longer a fixed-size vector of numbers. It is a sequence of words that need to be converted into numbers first. This is done by building a vocabulary and assigning each word an integer index.

The output is no longer a single number between 0 and 1. It is a probability distribution over all 77 words in the vocabulary. The network must decide which word is most likely to come next given the sequence it just read.

The loss function changes too. Instead of mean squared error, we use cross-entropy loss. It measures how wrong the probability distribution is compared to the actual next word.

![Fixed input classification versus word sequence prediction comparison](/04-rnn/images/classification_vs_sequence.png)

---

## Building the vocabulary

Before any training can happen, the text needs to be converted into numbers.

```python
with open('weather_corpus.txt', 'r') as corpus_file:
    corpus_text = corpus_file.read().lower()

all_words    = corpus_text.split()
unique_words = sorted(set(all_words))
vocabulary_size = len(unique_words)

word_to_index = {word: idx for idx, word in enumerate(unique_words)}
index_to_word = {idx: word for idx, word in enumerate(unique_words)}
```

This gives every unique word an integer ID. "clouds" might become 11, "rain" might become 52, "umbrella" might become 68. The mapping goes both ways so we can convert words to numbers for training and numbers back to words for generation.

```
Total words in corpus:    166
Unique words (vocabulary): 77
```

---

## One-hot encoding

Having an integer index for each word is not quite enough. The network cannot do meaningful matrix multiplication with a single integer. It needs a vector.

One-hot encoding converts a word index into a vector of zeros with a single 1 at the position of that word. Every word gets its own unique vector with the same length as the vocabulary.

For a vocabulary of 77 words, the word "rain" at index 52 becomes:

```
[0, 0, 0, 0, ... 0, 1, 0, ... 0, 0, 0]
 <--- 52 zeros --->  ^ <-- 24 zeros -->
               position 53 (index 52)
```

52 zeros, then a 1 at position 53 (because indexing starts at 0), then 24 more zeros. Total length is 77, matching the vocabulary size. Every word gets its own unique vector like this, with the 1 at a different position.

This might seem wasteful but it has a useful property. When you multiply a one-hot vector by a weight matrix, you are simply selecting one row of that matrix. The row corresponding to the word's index. This is exactly how the network looks up the representation for each word efficiently.

The code handles this with a small helper function:

```python
def one_hot_encode(word_index, vocabulary_size):
    one_hot_vector                = np.zeros((1, vocabulary_size))
    one_hot_vector[0, word_index] = 1
    return one_hot_vector
```

One line sets everything to zero. One line sets the right position to one. That is the whole encoding.

![One-hot vector showing 77 squares with one teal square at position 53 representing the word rain](/04-rnn/images/one_hot_vector.png)

---

## One-hot encoding and the identity matrix

If you stacked the one-hot vectors for all 77 words in the vocabulary into a single matrix, you would get a 77 by 77 identity matrix. Every row is a different word. Every row has exactly one 1 and the rest zeros. The diagonal is entirely ones.

```
word 0:  [1, 0, 0, 0, ... 0]
word 1:  [0, 1, 0, 0, ... 0]
word 2:  [0, 0, 1, 0, ... 0]
...
word 76: [0, 0, 0, 0, ... 1]
```

This reveals something useful about what the forward pass is actually doing. When you multiply a one-hot vector by the weight matrix, you are not really doing a matrix multiplication. You are selecting one row.

```
[0, 0, 1, 0, ... 0] × W = row 2 of W
```

The one-hot vector acts as a row selector. Everything else in the multiplication contributes zero. Only the position of the 1 matters.

This is why modern frameworks skip the one-hot vector entirely. Instead of creating a sparse vector and multiplying it away, PyTorch's `nn.Embedding` layer just does a direct index lookup into the weight matrix.

```python
# What we are doing (one-hot then multiply)
word_vector  = one_hot_encode(word_index, vocabulary_size)
hidden_input = np.dot(word_vector, weights_input_to_hidden)

# What PyTorch does instead (direct lookup, same result)
hidden_input = weights_input_to_hidden[word_index]
```

Same mathematical result. No sparse vector created. Much faster at scale.

We are using the one-hot approach here because it makes the matrix multiplication visible and explicit. In a later project when we switch to an embedding layer, you will already know exactly what it is doing underneath.

---

## Building training sequences

The RNN learns by reading sequences of words and predicting the next one. With a sequence length of 3, every group of three consecutive words becomes a training example with the fourth word as the target.

```python
sequence_length    = 3
training_sequences = []
training_targets   = []

for i in range(len(all_words) - sequence_length):
    sequence = all_words[i : i + sequence_length]
    target   = all_words[i + sequence_length]
    training_sequences.append([word_to_index[w] for w in sequence])
    training_targets.append(word_to_index[target])
```

This gives us 163 training sequences from 166 words. A few examples:

```
['the', 'sky', 'is']          -> 'cloudy'
['sky', 'is', 'cloudy']       -> 'today'
['bring', 'your', 'umbrella'] -> 'when'
['dark', 'clouds', 'mean']    -> 'heavy'
```

![Sliding window diagram showing how three consecutive words predict the next word](/04-rnn/images/sequence_window.png)

---

## The RNN architecture

The RNN has three weight matrices, not two like the networks in Projects 1 through 3.

The first connects input words to the hidden state.
The second connects the previous hidden state to the new hidden state.
The third connects the hidden state to the output predictions.

```python
weights_input_to_hidden  = np.random.randn(vocabulary_size, hidden_size) * 0.01
weights_hidden_to_hidden = np.random.randn(hidden_size, hidden_size) * 0.01
weights_hidden_to_output = np.random.randn(hidden_size, vocabulary_size) * 0.01
bias_hidden              = np.zeros((1, hidden_size))
bias_output              = np.zeros((1, vocabulary_size))
```

The hidden-to-hidden weight matrix is what makes it recurrent. At every step, the network reads the current word and also reads its own previous hidden state. This is how it carries memory of earlier words forward through the sequence.

![RNN architecture showing three weight matrices and the hidden state loop](/04-rnn/images/rnn_architecture.png)

---

## The activation function: tanh

In Projects 1 and 2 we used sigmoid as the activation function. In the RNN hidden state we use tanh instead.

Both sigmoid and tanh are S-shaped curves that squash their input into a bounded range. The difference is the range itself.

Sigmoid outputs values between 0 and 1. Tanh outputs values between -1 and 1.

```
Input   ->  sigmoid  ->  tanh
-3      ->  0.0474   ->  -0.9951
-2      ->  0.1192   ->  -0.9640
-1      ->  0.2689   ->  -0.7616
 0      ->  0.5000   ->   0.0000
 1      ->  0.7311   ->   0.7616
 2      ->  0.8808   ->   0.9640
 3      ->  0.9526   ->   0.9951
```

A few things to notice.

Tanh is centred at zero. An input of 0 produces an output of 0. This means the hidden state starts at zero and can move in either direction as it reads each word. Positive values push the hidden state one way. Negative values push it the other way.

Sigmoid is centred at 0.5. Its outputs are always positive. This makes it well suited for producing probabilities (which should be between 0 and 1) but less suited for a hidden state that needs to represent a richer range of information.

Tanh also has a stronger gradient near zero. At x = 0, the tanh derivative is 1.0. At x = 0, the sigmoid derivative is 0.25. This means tanh produces larger gradient signals during backpropagation, which helps learning in the early training stages.

The tanh derivative, like sigmoid, follows from the function itself:

```
tanh'(x) = 1 - tanh(x)²

At x = 0:  1 - 0.0000² = 1.0000   <- strongest signal
At x = 1:  1 - 0.7616² = 0.4200
At x = 2:  1 - 0.9640² = 0.0707
At x = 3:  1 - 0.9951² = 0.0099   <- nearly zero, saturated
```

The same saturation problem from sigmoid applies to tanh. At the extremes, the derivative approaches zero and gradients vanish. This is the same vanishing gradient problem the RNN article in the theory series described mathematically. Here it is visible in the actual derivative values.

This is why the hidden state is initialised to zeros at the start of each sequence. Starting at zero keeps the network in the region where tanh has its strongest gradient and learning is fastest.

![Tanh and sigmoid curves on the same axes showing tanh centred at zero and sigmoid centred at 0.5](/04-rnn/images/tanh_vs_sigmoid.png)

---

## The forward pass

For each training sequence, the RNN processes one word at a time. At each step it updates the hidden state using both the current word and the previous hidden state.

```python
hidden_state = np.zeros((1, hidden_size))

for word_index in sequence_indices:
    word_vector  = one_hot(word_index, vocabulary_size)
    hidden_input = np.dot(word_vector,  weights_input_to_hidden)  + \
                   np.dot(hidden_state, weights_hidden_to_hidden) + \
                   bias_hidden
    hidden_state = np.tanh(hidden_input)
```

After reading all three words, the hidden state is passed to the output layer which produces a score for every word in the vocabulary.

```python
output_scores        = np.dot(hidden_state, weights_hidden_to_output) + bias_output
output_probabilities = softmax(output_scores[0])
```

---

## Softmax

The output layer produces 77 raw scores, one for each word in the vocabulary. These scores can be any number, positive or negative, large or small. To turn them into a probability distribution that sums to 1, we apply softmax.

The formula is:

```
softmax(x_i) = exp(x_i) / sum of exp(x_j) for all j
```

In plain English: exponentiate every score, then divide each one by the total. This guarantees all values are positive and sum to 1.

A small example with three words to make it concrete:

```
raw scores:     [2.1,   0.3,  -0.8]

exponentiated:  [8.17,  1.35,  0.45]   <- exp of each score
sum:             9.97

after softmax:  [0.82,  0.14,  0.05]   <- each divided by sum
                 ----
                 this word wins
```

The word with the highest raw score gets the highest probability. But softmax amplifies the differences. A score of 2.1 versus 0.3 becomes a probability of 0.82 versus 0.14. The winner gets most of the probability mass.

This amplification is intentional. At the start of training when all scores are near zero, softmax produces a nearly flat distribution, close to random guessing. As training progresses and the network learns to assign higher scores to correct words, the distribution becomes more peaked. The network becomes more confident.

The code for softmax subtracts the maximum score before exponentiating. This prevents numerical overflow when scores are very large.

```python
def softmax(scores):
    shifted_scores = scores - np.max(scores)
    exponentiated  = np.exp(shifted_scores)
    return exponentiated / np.sum(exponentiated)
```

The highest probability is the predicted next word.

![Probability distribution over 77 vocabulary words showing one dominant bar as the predicted next word](/04-rnn/images/probability_distribution.png)

---

## Cross-entropy loss

For classification problems where the output is a probability distribution, cross-entropy loss works better than mean squared error.

The formula is simple. Take the probability the network assigned to the correct word, take its logarithm, and negate it.

```python
correct_probability = output_probabilities[target_index]
loss                = -np.log(correct_probability + 1e-8)
```

If the network is very confident and correct, the probability is close to 1 and the log is close to 0. Loss is small.

If the network is very wrong, the probability is close to 0 and the negative log is large. Loss is large.

The small constant 1e-8 prevents taking the log of exactly zero which would cause a numerical error.

![Cross-entropy loss curve showing loss is high when prediction is wrong and approaches zero when correct](/04-rnn/images/cross_entropy_loss.png)

---

## Training results

```
Epoch    0  loss: 4.3386
Epoch  200  loss: 3.6814
Epoch  400  loss: 2.4480
Epoch  600  loss: 2.0643
Epoch  800  loss: 2.0097
Epoch 1000  loss: 1.8358
```

The loss drops from 4.33 to 1.83 over 1000 epochs. For comparison, a network guessing randomly on 77 words would have a loss of approximately log(77) which is 4.34. Starting loss of 4.33 confirms the network starts by guessing randomly. By epoch 1000 it has learned real patterns.

![Loss curve showing training loss dropping from random baseline of 4.34 toward 1.83 over 1000 epochs](/04-rnn/images/loss_curve.png)

---

## Generated text

After training, the network can generate new text by reading a seed sequence and repeatedly predicting the next word.

```
Seed: 'the sky is'
Generated: the sky is falling needs bring rain the clouds

Seed: 'bring your umbrella'
Generated: bring your umbrella when it rains cold the a

Seed: 'dark clouds mean'
Generated: dark clouds mean heavy rain a carry an day

Seed: 'the rain will'
Generated: the rain will stop by the clouds day dark

Seed: 'a clear sky'
Generated: a clear sky skies sky needs rain needs by
```

The output is imperfect. After the first few words the generation loses coherence. This is expected for a small RNN trained on 166 words for 1000 epochs.

But look at what it did get right. "bring your umbrella when it rains" is a complete and coherent phrase, copied almost exactly from the training data. "the rain will stop by" is also coherent. "dark clouds mean heavy rain" is correct too.

The network has learned real associations between words. It knows that "umbrella" tends to follow "your" and "when it rains" tends to follow "umbrella". It knows that "heavy rain" tends to follow "dark clouds mean". It learned these patterns entirely from the data.

---

## What the RNN still cannot do well

The generation falls apart after a few words. This is the vanishing gradient problem from the RNN article in my theory series, now visible in the actual output rather than just in the math.

The hidden state can only carry so much information forward. For short sequences it works. For longer sequences the earlier context fades and the generation loses its thread.

This is exactly the problem that the Attention Mechanism was designed to solve, and exactly why the next project will add attention on top of this RNN. The same corpus, the same vocabulary, but a fundamentally different way of connecting words across distance.

The overfitting question from Project 3 is also becoming more visible here. The network has learned to reproduce phrases from the training data almost exactly. That is not generalisation. That is memorisation. A validation split on the generated text would reveal this clearly, and it is something the next few projects will address properly.

---

## What I took away from this

Moving from fixed-input classification to sequence prediction felt like a significant jump. The architecture is more complex, the data preparation is new, the loss function changed, and the output is now words rather than a single number.

But the core loop is identical to what we built in Project 2. Forward pass, compute loss, backward pass, update weights. The same three steps, applied to a different kind of problem.

That consistency is the most useful thing to hold onto as the series gets more complex. The details change. The loop does not.
