# 01: Build a Single Neuron From Scratch

This is the first project in a series that builds from a single neuron all the way up to a small language model. Every project adds one concept on top of the last. This one starts at the very bottom: one neuron, doing one calculation, producing one number.

If neural networks have always felt like a black box to you, this is the right place to start. By the end of this document you will understand exactly what a neuron does, why it does it, and how to write one yourself in about ten lines of Python.

---

## What is a neuron?

A neuron is the basic unit of a neural network. The name comes from biology (brain cells are also called neurons), but the math is much simpler than biology.

A neuron takes in some numbers, combines them in a specific way, and produces one number as output.

That is all it does. The interesting part is how it combines them, and what happens when you connect thousands of neurons together and let them learn from data. But before we get there, we need to understand the single neuron first.

---

## The three ingredients

Every neuron has three ingredients: **inputs**, **weights**, and a **bias**.

### Inputs

Inputs are the numbers you feed into the neuron. They represent some measurable features of the thing you are trying to reason about.

For example, if you are building a neuron to predict whether it will rain, your inputs might be:

- Cloud cover (a number between 0 and 1, where 1 means completely overcast)
- Humidity (a number between 0 and 1)
- Wind speed (a number between 0 and 1)

The neuron does not care what those numbers represent. It just sees three numbers and works with them.

### Weights

Each input has a corresponding weight. The weight controls how much influence that input has on the final output.

A larger weight means "pay more attention to this input". A smaller weight means "this input matters less". A negative weight means "this input pushes the output down".

For the rain prediction example, cloud cover is probably the most important factor, so it gets a larger weight. Wind speed matters less, so it gets a smaller weight.

```text
weight for cloud cover = 0.6   (most important)
weight for humidity    = 0.3   (moderately important)
weight for wind speed  = 0.1   (least important)
```

These weights are numbers you choose. In a real trained neural network, the weights are numbers the network learns automatically from data. That is the subject of Project 2. For now, we pick them by hand.

### Bias

The bias is a single number added to the result after the inputs and weights have been combined. It shifts the output up or down regardless of what the inputs are.

Think of it as a default tendency. A negative bias makes the neuron skeptical by default; even with moderate inputs, it needs stronger evidence to produce a high output. A positive bias makes it optimistic by default.

```text
bias = -0.2
```

---

## How the neuron calculates its output

With inputs, weights, and a bias in hand, the neuron does two things:

### Step 1: Weighted sum

Multiply each input by its weight, add all of those products together, then add the bias. This is called the weighted sum.

```text
weighted_sum = (input_1 x weight_1) + (input_2 x weight_2) + (input_3 x weight_3) + bias
```

Using our umbrella example with inputs of 0.9, 0.7, 0.3:

```text
weighted_sum = (0.9 x 0.6) + (0.7 x 0.3) + (0.3 x 0.1) + (-0.2)
             = 0.54 + 0.21 + 0.03 - 0.2
             = 0.58
```

### Step 2: Activation function

The weighted sum (0.58) is just a raw number. It could theoretically be any value, positive or negative, large or small. Before using it as an output, we pass it through an **activation function** that squashes it into a useful range.

The most common activation function for outputs that represent probabilities is the **sigmoid function**:

```text
sigmoid(x) = 1 / (1 + e^(-x))
```

The sigmoid function takes any number and maps it to a value between 0 and 1. Large positive numbers map to values close to 1. Large negative numbers map to values close to 0. Numbers near zero map to values near 0.5.

```text
sigmoid(0.58) = 1 / (1 + e^(-0.58))
              = 1 / (1 + 0.5599)
              = 1 / 1.5599
              = 0.6411
```

The neuron's output is 0.6411. Since this is greater than 0.5, the decision is: bring the umbrella.

Here is the full picture in one diagram:

```text
inputs          weights         weighted sum      activation      output

cloud = 0.9  x  0.6  = 0.54
                              0.54
humidity = 0.7 x 0.3 = 0.21  + 0.21  = 0.58  --> sigmoid --> 0.6411
                              + 0.03
wind = 0.3   x  0.1  = 0.03  - 0.20
bias = -0.2
```

---

## Why do we need an activation function?

Without an activation function, a neuron just computes a weighted sum. That is a linear function: it draws a straight line (or a flat plane in higher dimensions) to separate two categories.

Most real problems are not linearly separable. You cannot always draw a straight line to separate "bring umbrella" from "leave it" across all possible weather combinations.

Activation functions introduce non-linearity. A network of neurons with non-linear activations can learn curved, complex boundaries between categories rather than just straight lines. That is what gives neural networks their power.

The sigmoid function is used here because it produces a value between 0 and 1, which maps naturally to a probability. Project 9 onwards uses a different activation called SwiGLU, which is better suited to large language models. But the idea is the same: apply a non-linear function after the weighted sum.

---

## The code

Here is the neuron implemented in Python using numpy:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def neuron(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    return sigmoid(weighted_sum)
```

The `np.dot` function computes the dot product of two arrays, which is exactly the weighted sum: multiply each pair of corresponding elements and add all the results together. It is more concise than writing out each multiplication individually and works for inputs of any length.

Running it with the umbrella example:

```python
inputs  = np.array([0.9, 0.7, 0.3])
weights = np.array([0.6, 0.3, 0.1])
bias    = -0.2

output = neuron(inputs, weights, bias)

print(f"Inputs: cloud cover={inputs[0]}, humidity={inputs[1]}, wind speed={inputs[2]}")
print(f"Output (probability): {output:.4f}")
print(f"Decision: {'bring umbrella' if output > 0.5 else 'leave it'}")
```

Output:

```text
Inputs: cloud cover=0.9, humidity=0.7, wind speed=0.3
Output (probability): 0.6411
Decision: bring umbrella
```

---

## What this neuron cannot do

This neuron has two important limitations.

**The weights are hand-picked.** I chose 0.6, 0.3, and 0.1 because they made intuitive sense. But the right weights for a real problem are not obvious. If you change the weights, you get a different answer for the same inputs. A real neural network learns the right weights automatically from labeled examples. That is the subject of Project 2.

**One neuron can only represent a simple decision boundary.** A single neuron divides its input space with a single hyperplane. If the true boundary between "bring umbrella" and "leave it" is curved or irregular, one neuron cannot represent it no matter what weights you use. To learn complex patterns you need many neurons organized in layers, each layer building on the output of the previous one. That is what the later projects build toward.

---

## Connecting this to the rest of the series

Every neural network, no matter how large, is built from this same calculation repeated many times:

```text
output = activation(dot(inputs, weights) + bias)
```

In Project 2, we connect neurons into a network and write the training loop that adjusts weights automatically. By Project 10, the architecture is equivalent to LLaMA 2. But every single unit inside that architecture is doing the same thing as the neuron on this page.

The complication in large models is not in the individual pieces. It is in how many of them there are, how they are connected, and how the training signal travels back through all of them to update the weights. Understanding this one neuron first makes all of that much easier to follow.

---

## Running

```bash
pip install numpy
python neuron.py
```

## Files

```text
neuron.py      the neuron implementation
config.json    stores the bias value
```
