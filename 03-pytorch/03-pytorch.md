# 03: Rebuilding in PyTorch and Introducing the Validation Split

In Project 2 we built a neural network from scratch using only numpy. Every gradient was computed by hand. Every weight update was written explicitly. That was the right approach for understanding what backpropagation actually does.

But nobody builds real networks this way. Real networks are built with frameworks that handle the gradients automatically. The most widely used one in research and industry is **PyTorch**.

This project rebuilds the exact same neural network in PyTorch: same architecture (3 inputs, 4 hidden neurons, 1 output), same data, same problem. The goal is to show that PyTorch is not doing anything we do not already understand. It is a clean abstraction over math we already wrote by hand.

There is also one genuinely new concept here: the **validation split**. It introduces a fundamental idea in machine learning: the difference between a network that has memorised its training data and a network that has actually learned.

---

## What is PyTorch and why use it?

PyTorch is a Python library for building and training neural networks. It provides two things that matter most:

**Tensors**, which are multi-dimensional arrays similar to numpy arrays, but with the ability to run on a GPU and to track the operations performed on them.

**Autograd**, an automatic differentiation engine. Every time you perform an operation on a tensor, PyTorch records it. When you call `loss.backward()` at the end of a forward pass, PyTorch walks that record in reverse and computes the derivative of the loss with respect to every weight in the network. This is the chain rule applied automatically.

In Project 2, computing all those gradients by hand took around thirty lines of careful code. In PyTorch, one line does it:

```python
training_loss.backward()
```

That single call replaces everything we wrote manually. The math is identical. The implementation is automated.

The reason to learn it by hand first (as we did in Project 2) is so that this line is not mysterious. You know exactly what it is doing. That understanding becomes important when things go wrong in training and you need to reason about why.

---

## Three PyTorch building blocks

Three PyTorch classes appear in this project. It helps to understand each one before reading the code.

### nn.Module

The base class for every neural network in PyTorch. You inherit from it and define a `forward` method that describes how data moves through your layers. PyTorch takes care of tracking parameters, moving the network to a GPU, saving and loading weights, and many other housekeeping tasks.

```python
class WeatherPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(3, 4)
        self.output_layer = nn.Linear(4, 1)
        self.activation   = nn.Sigmoid()

    def forward(self, x):
        hidden = self.activation(self.hidden_layer(x))
        return self.activation(self.output_layer(hidden))
```

### nn.Linear

A fully connected layer. `nn.Linear(3, 4)` creates a layer with a (3, 4) weight matrix and a (4,) bias vector. When you call `self.hidden_layer(x)`, it computes `x @ weights.T + bias` automatically. This replaces the manual `np.dot(inputs, weights) + bias` from Project 2.

### Optimiser (SGD)

The optimiser takes the gradients computed by `backward()` and applies them to the weights. `optim.SGD(network.parameters(), lr=0.5)` creates a stochastic gradient descent optimiser with learning rate 0.5: the same update rule we wrote by hand in Project 2, now handled automatically.

---

## The training loop structure

Every PyTorch training loop follows the same four-step pattern per epoch:

```python
optimiser.zero_grad()                        # 1. clear old gradients
predicted = network(inputs)                  # 2. forward pass
loss = loss_function(predicted, labels)      # 3. compute loss
loss.backward()                              # 4. compute gradients (backprop)
optimiser.step()                             # 5. update weights
```

Step 1 is easy to forget and important: PyTorch accumulates gradients by default, so you must clear them at the start of each epoch or they will build up incorrectly.

Steps 2 and 3 are the forward pass and loss calculation, the same as Project 2.

Step 4 is backpropagation: PyTorch walks the computational graph in reverse and fills in the `.grad` attribute of every weight tensor.

Step 5 applies the updates: for each weight, `weight = weight - learning_rate * weight.grad`.

---

## The validation split

In Projects 1 and 2, the network was trained and evaluated on the same six examples. That is not a meaningful test of learning. A network can score perfectly on training data simply by memorising the exact examples it saw, without understanding the underlying pattern.

A **validation split** fixes this by separating the data into two groups before training begins:

- The **training set** is what the network learns from. Weights are updated based on training loss only.
- The **validation set** is used to measure how the network performs on examples it has never seen during training. No weight updates happen on validation data.

If training loss keeps falling but validation loss flattens or rises, the network is **overfitting**: it is getting better at the training examples while getting worse at generalising to new ones.

For this project the split is:

```text
Training set (6 examples, same as before):
  [0.9, 0.7, 0.3]  -> 1 (bring umbrella)
  [0.1, 0.2, 0.1]  -> 0 (leave it)
  [0.8, 0.6, 0.5]  -> 1
  [0.2, 0.3, 0.2]  -> 0
  [0.7, 0.8, 0.4]  -> 1
  [0.1, 0.1, 0.9]  -> 0

Validation set (3 new examples, never seen during training):
  [0.85, 0.75, 0.20]  -> 1 (cloudy and humid)
  [0.15, 0.25, 0.15]  -> 0 (clear and dry)
  [0.60, 0.50, 0.60]  -> 1 (moderately cloudy and windy)
```

---

## The code

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Training data: same six examples as Projects 1 and 2
training_inputs = torch.tensor([
    [0.9, 0.7, 0.3],
    [0.1, 0.2, 0.1],
    [0.8, 0.6, 0.5],
    [0.2, 0.3, 0.2],
    [0.7, 0.8, 0.4],
    [0.1, 0.1, 0.9],
], dtype=torch.float32)

training_labels = torch.tensor([[1], [0], [1], [0], [1], [0]], dtype=torch.float32)

# Validation data: new examples the network never sees during training
validation_inputs = torch.tensor([
    [0.85, 0.75, 0.20],
    [0.15, 0.25, 0.15],
    [0.60, 0.50, 0.60],
], dtype=torch.float32)

validation_labels = torch.tensor([[1], [0], [1]], dtype=torch.float32)


class WeatherPredictor(nn.Module):

    def __init__(self):
        super(WeatherPredictor, self).__init__()
        self.hidden_layer = nn.Linear(3, 4)
        self.output_layer = nn.Linear(4, 1)
        self.activation   = nn.Sigmoid()

    def forward(self, network_input):
        hidden_output  = self.activation(self.hidden_layer(network_input))
        network_output = self.activation(self.output_layer(hidden_output))
        return network_output


network       = WeatherPredictor()
loss_function = nn.MSELoss()
optimiser     = optim.SGD(network.parameters(), lr=0.5)

training_loss_history   = []
validation_loss_history = []

for epoch in range(5000):

    # Training step: forward pass, loss, backprop, weight update
    network.train()
    optimiser.zero_grad()
    predicted_training_output = network(training_inputs)
    training_loss             = loss_function(predicted_training_output, training_labels)
    training_loss.backward()
    optimiser.step()
    training_loss_history.append(training_loss.item())

    # Validation step: forward pass only, no gradient tracking, no weight update
    network.eval()
    with torch.no_grad():
        predicted_validation_output = network(validation_inputs)
        validation_loss             = loss_function(predicted_validation_output, validation_labels)
        validation_loss_history.append(validation_loss.item())

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d}  training loss: {training_loss.item():.4f}  "
              f"validation loss: {validation_loss.item():.4f}")
```

Two things in the validation step are worth noting:

`network.eval()` switches the network to evaluation mode. For this network it makes no practical difference (there is no dropout or batch normalisation here), but it is the correct habit to establish early. Some layers behave differently during training and evaluation.

`torch.no_grad()` tells PyTorch not to build the computational graph for this forward pass. Since we are not going to call `backward()` on validation data, there is no reason to track operations. This saves memory and runs faster.

---

## What the training output looks like

```text
Epoch     0  training loss: 0.3125  validation loss: 0.3401
Epoch  1000  training loss: 0.0024  validation loss: 0.0041
Epoch  2000  training loss: 0.0009  validation loss: 0.0017
Epoch  3000  training loss: 0.0005  validation loss: 0.0010
Epoch  4000  training loss: 0.0003  validation loss: 0.0007

Training data predictions:
  Sample 1: predicted 0.9901 -> bring umbrella (correct)
  Sample 2: predicted 0.0090 -> leave it       (correct)
  Sample 3: predicted 0.9789 -> bring umbrella (correct)
  Sample 4: predicted 0.0279 -> leave it       (correct)
  Sample 5: predicted 0.9812 -> bring umbrella (correct)
  Sample 6: predicted 0.0063 -> leave it       (correct)

Validation data predictions (unseen examples):
  Sample 1: predicted 0.9742 -> bring umbrella (correct)
  Sample 2: predicted 0.0221 -> leave it       (correct)
  Sample 3: predicted 0.8934 -> bring umbrella (correct)
```

Both training and validation loss fall together, and the network predicts the unseen validation examples correctly. On this simple problem that is expected: the patterns in the data are clean enough that the network generalises without difficulty.

---

## What to watch for: the two loss curves

The most important thing to observe when you run this script is the relationship between the two loss curves printed (and plotted) during training.

**Both curves fall together and stay close.** This is the healthy case. The network is learning a general rule that works on new data, not just memorising the training examples.

**Training loss keeps falling but validation loss flattens or rises.** This is overfitting. The network is improving on the training examples at the expense of generalisation. It has started to memorise rather than learn.

On this small, clean dataset the two curves will track closely because the problem is too simple for the network to overfit significantly. As later projects work with larger datasets and more complex patterns, keeping these two curves in view becomes one of the most important diagnostic habits in training.

---

## What PyTorch replaced line by line

Here is a direct comparison of what changed between Project 2 and this project.

**Defining the layers:**

```text
Project 2 (numpy):
  weights_input_to_hidden  = np.random.randn(3, 4) * 0.5
  bias_hidden              = np.zeros((1, 4))
  weights_hidden_to_output = np.random.randn(4, 1) * 0.5
  bias_output              = np.zeros((1, 1))

Project 3 (PyTorch):
  self.hidden_layer = nn.Linear(3, 4)
  self.output_layer = nn.Linear(4, 1)
```

**Forward pass:**

```text
Project 2:
  hidden_layer_input  = np.dot(training_inputs, weights_input_to_hidden) + bias_hidden
  hidden_layer_output = sigmoid(hidden_layer_input)
  output_layer_input  = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
  predicted_output    = sigmoid(output_layer_input)

Project 3:
  hidden_output  = self.activation(self.hidden_layer(x))
  return self.activation(self.output_layer(hidden_output))
```

**Backpropagation and weight update:**

```text
Project 2:
  ~30 lines of manual gradient computation and weight update

Project 3:
  loss.backward()
  optimiser.step()
```

The math is the same. PyTorch has replaced the manual bookkeeping, not the underlying computation.

---

## What is next

This project completes the foundation. We now have a network that learns from data, trains with automatic differentiation, and is evaluated correctly against unseen examples.

Project 4 moves beyond the umbrella problem entirely. The network will learn from real text, one character at a time, using a Recurrent Neural Network (RNN). That is where the series starts to look like building a language model rather than building a toy classifier.

---

## Running

```bash
pip install torch matplotlib
python weather_predictor.py
```

The script trains for 5000 epochs, prints loss at every 1000th epoch, prints final predictions on both training and validation data, and saves a loss curve plot to `loss_curve.png`.

## Files

```text
weather_predictor.py    full PyTorch training script
config.json             hyperparameters (epochs, learning rate, seed)
images/
  loss_curve.png        training vs validation loss across 5000 epochs
```
