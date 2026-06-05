# Rebuilding in PyTorch and Introducing the Validation Split

*Project 3 in my build series at github.com/dwinsi/LLMfromScratch*

In Project 2 I built a neural network from scratch using only numpy. Every gradient was computed by hand. Every weight update was written explicitly. It was the right way to learn what backpropagation actually does.

But nobody builds real networks this way. Real networks are built with frameworks that handle the gradients automatically. The most widely used one in research and industry is PyTorch.

Project 3 rebuilds the exact same neural network in PyTorch. Same architecture, same data, same problem. The goal is to show that what PyTorch is doing underneath is exactly what we already built by hand. The framework is not magic. It is a clean abstraction over math we already understand.

There is also something new in this project. A validation split. And that addition will start to surface a problem called overfitting that I first mentioned in Project 2.

---

## Why PyTorch

In Project 2, computing the gradients by hand required about thirty lines of careful code. In PyTorch, one line replaces all of it.

```python
training_loss.backward()
```

That single call computes every gradient in the network automatically using a system called autograd. PyTorch tracks every operation performed on a tensor and builds a computational graph as it goes. When you call backward, it walks that graph in reverse and computes the derivative of the loss with respect to every weight.

This is the same chain rule we applied by hand in Project 2. PyTorch just does it automatically, for networks of any size and depth.

The reason to learn it by hand first is exactly this. When you see `training_loss.backward()` now, you know what it is doing. It is not a black box. It is the backpropagation we already wrote, generalised and automated.

---

## Three new concepts in PyTorch

Before the code, three PyTorch concepts are worth naming clearly.

**nn.Module** is the base class for every neural network in PyTorch. You define your network by inheriting from it and writing a forward method that describes how data flows through the layers.

**nn.Linear** is a fully connected layer. It holds a weight matrix and a bias vector and applies them to the input. This replaces the manual `np.dot(input, weights) + bias` from Project 2.

**autograd** is PyTorch's automatic differentiation engine. Every tensor operation is tracked. When you call `loss.backward()`, autograd computes all the gradients. The optimiser then uses those gradients to update the weights.

---

## The train/validation split

In Projects 1 and 2 the network was trained and tested on the same six examples. It saw every example during training and was then asked to predict those same examples. That is not a real test of learning. A network can perform perfectly on training data simply by memorising it.

A validation split fixes this by dividing the data into two groups before training begins.

The training set is what the network learns from. Weights are updated based on training loss only.

The validation set is what the network is tested on during training. The network never learns from validation data. It only makes predictions on it, and those predictions measure how well the network is generalising to examples it has not seen.

If training loss keeps going down but validation loss starts going up, the network is overfitting. It is memorising the training examples rather than learning a general rule.

For our umbrella problem, the full dataset is small. So the split is:

Training set: six original examples from Projects 1 and 2.
Validation set: three new examples with weather conditions the network has never seen during training.

```
Validation examples:
  [0.85, 0.75, 0.20]  -> bring umbrella  (cloudy and humid)
  [0.15, 0.25, 0.15]  -> leave it        (clear and dry)
  [0.60, 0.50, 0.60]  -> bring umbrella  (moderately cloudy and windy)
```

---

## The code

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ---- Training data (same as Projects 1 and 2) ----
training_inputs = torch.tensor([
    [0.9, 0.7, 0.3],
    [0.1, 0.2, 0.1],
    [0.8, 0.6, 0.5],
    [0.2, 0.3, 0.2],
    [0.7, 0.8, 0.4],
    [0.1, 0.1, 0.9],
], dtype=torch.float32)

training_labels = torch.tensor(
    [[1], [0], [1], [0], [1], [0]],
    dtype=torch.float32
)

# ---- Validation data (new examples, never seen during training) ----
validation_inputs = torch.tensor([
    [0.85, 0.75, 0.20],
    [0.15, 0.25, 0.15],
    [0.60, 0.50, 0.60],
], dtype=torch.float32)

validation_labels = torch.tensor(
    [[1], [0], [1]],
    dtype=torch.float32
)


# ---- Network definition ----
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
epochs = 5000

for epoch in range(epochs):

    # ---- Training step ----
    network.train()
    optimiser.zero_grad()
    predicted_training_output = network(training_inputs)
    training_loss             = loss_function(predicted_training_output, training_labels)
    training_loss.backward()
    optimiser.step()
    training_loss_history.append(training_loss.item())

    # ---- Validation step ----
    network.eval()
    with torch.no_grad():
        predicted_validation_output = network(validation_inputs)
        validation_loss             = loss_function(predicted_validation_output, validation_labels)
        validation_loss_history.append(validation_loss.item())

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d}  "
              f"training loss: {training_loss.item():.4f}  "
              f"validation loss: {validation_loss.item():.4f}")


# ---- Final predictions ----
print()
print("Training data predictions:")
network.eval()
with torch.no_grad():
    final_training_predictions = network(training_inputs)
    for i, (predicted, actual) in enumerate(zip(final_training_predictions, training_labels)):
        decision = "bring umbrella" if predicted.item() > 0.5 else "leave it"
        correct  = "correct" if round(predicted.item()) == actual.item() else "wrong"
        print(f"  Sample {i+1}: predicted {predicted.item():.4f} -> {decision} ({correct})")

print()
print("Validation data predictions (unseen examples):")
with torch.no_grad():
    final_validation_predictions = network(validation_inputs)
    for i, (predicted, actual) in enumerate(zip(final_validation_predictions, validation_labels)):
        decision = "bring umbrella" if predicted.item() > 0.5 else "leave it"
        correct  = "correct" if round(predicted.item()) == actual.item() else "wrong"
        print(f"  Sample {i+1}: predicted {predicted.item():.4f} -> {decision} ({correct})")


# ---- Plot training vs validation loss ----
plt.figure(figsize=(10, 5))
plt.plot(training_loss_history,   color='steelblue', linewidth=1.5, label='Training loss')
plt.plot(validation_loss_history, color='tomato',    linewidth=1.5, label='Validation loss', linestyle='--')
plt.title('Training vs Validation Loss', fontsize=13)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Loss (Mean Squared Error)', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.show()

print("\nLoss curve saved to loss_curve.png")
```

---

## What to look for in the output

When you run this, watch two numbers printed at each epoch: training loss and validation loss.

At the start, both will be roughly similar. As training progresses, training loss will continue dropping. What validation loss does is the interesting part.

If both curves drop together and stay close, the network is generalising well. It is learning a rule that applies to new examples as well as training examples.

If training loss keeps dropping but validation loss flattens or starts to rise, the network has begun to overfit. It is getting better at memorising the training examples but no longer improving on unseen ones.

On this small dataset, the curves will mostly track together because the problem is simple and the data is clean. But the gap between the two curves is the thing to watch as the dataset grows larger and the problems get more complex in later projects.

![Training vs validation loss curve showing both curves across 5000 epochs](/03-pytorch/images/loss_curve.png)

---

## What PyTorch replaced from Project 2

It is worth being explicit about what PyTorch is doing for us now.

In Project 2, this block took around thirty lines:

```
# Manually compute every gradient
# Manually apply every weight update
```

In Project 3, it becomes four lines:

```python
optimiser.zero_grad()
training_loss.backward()
optimiser.step()
```

`zero_grad` clears any gradients left over from the previous epoch. `backward` runs backpropagation through the entire network automatically. `step` applies the weight updates using the computed gradients.

The math is identical. The implementation is abstracted. Knowing what is underneath makes these four lines readable instead of mysterious.

---

## What I took away from this

Rebuilding the same network in PyTorch after writing it from scratch made the framework feel completely transparent. Every method call has a counterpart in the numpy version. Nothing is hidden, just abbreviated.

The validation split is the more important addition. Training loss alone tells you how well the network is memorising the training data. Validation loss tells you how well it is actually learning. The difference between those two things is one of the most important concepts in all of machine learning.

The next project will move beyond the umbrella problem entirely. An RNN trained on real text, generating characters one at a time. That is where the series starts to feel like building a language model rather than building a toy.

---

*Project 3 code lives at github.com/dwinsi/LLMfromScratch in the 03-pytorch folder. Run weather_predictor.py, watch both loss curves, and notice whether they stay together or start to diverge.*
