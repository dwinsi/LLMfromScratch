# Build a Single Neuron From Scratch

*Project 1 in my build series, where I rebuild the path from neural networks to a tiny LLM, one piece at a time*

I have spent months writing about how LLMs work. The Transformer. Attention. Memory. The whole architecture.

Now I want to actually build it.

Not by using a framework that hides what is happening. From scratch. Starting with the smallest possible piece and building up until I have a working language model trained on my own machine. These are my notes from the first project. The simplest possible neural network. One neuron.

---

## What a neuron actually is

Strip away the hype and a neuron is a small piece of math. Three things go in. Some numbers get multiplied together. One number comes out.

The three things are inputs, weights and a bias.

Inputs are whatever you are giving the neuron to look at. Numbers representing features of something.

Weights are how much the neuron cares about each input. A larger weight means that input has more influence on the answer.

The bias is a constant added to nudge the result up or down regardless of the inputs.

The neuron multiplies each input by its corresponding weight, adds everything together, adds the bias, and then runs the result through an activation function that squashes it into a useful range.

That is the whole thing. One small piece of math.

---

## A concrete example

Let me make this real with an example.

Suppose I want to build a neuron that decides whether to bring an umbrella. The neuron looks at three inputs.

Cloud cover, a number between 0 and 1 representing how cloudy the sky is.
Humidity, a number between 0 and 1.
Wind speed, also between 0 and 1.

The neuron also has three weights, one for each input. Intuitively, cloud cover should matter the most, humidity should matter somewhat, and wind speed should matter the least.

So my weights might be something like:

```
weight for cloud cover = 0.6
weight for humidity = 0.3
weight for wind speed = 0.1
```

And a small negative bias to make the neuron slightly skeptical by default.

```
bias = -0.2
```

Now I feed in some inputs. A cloudy, slightly humid, light wind day:

```
cloud cover = 0.9
humidity = 0.7
wind speed = 0.3
```

The neuron does its math.

```
weighted sum = (0.9 × 0.6) + (0.7 × 0.3) + (0.3 × 0.1) + (-0.2)
             = 0.54 + 0.21 + 0.03 - 0.2
             = 0.58
```

This number on its own is not very useful. So the neuron passes it through an activation function. For this example I will use the sigmoid function, which squashes any number into a value between 0 and 1.

```
sigmoid(0.58) = 0.6411
```

That output, 0.6411, is the neuron's prediction. Higher than 0.5, so the answer is bring the umbrella.

That is one neuron. Doing one forward pass. With three real numbers and a sensible answer.

---

## The actual code

Here is the same example written in Python with numpy. About ten lines.

```python
import numpy as np

# A single neuron predicting whether you should bring an umbrella
# Inputs: cloud_cover (0-1), humidity (0-1), wind_speed (0-1)
inputs = np.array([0.9, 0.7, 0.3])
weights = np.array([0.6, 0.3, 0.1])
bias = -0.2

# Forward pass
weighted_sum = np.dot(inputs, weights) + bias

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

output = sigmoid(weighted_sum)

print(f"Weighted sum: {weighted_sum:.4f}")
print(f"Output (probability): {output:.4f}")
print(f"Decision: {'bring umbrella' if output > 0.5 else 'leave it'}")
```

Running this gives:

```
Weighted sum: 0.5800
Output (probability): 0.6411
Decision: bring umbrella
```

That is it. A working neuron. No PyTorch. No TensorFlow. Just numpy and a few lines of math.

---

## What is missing

What I just built is not really a neural network. It is one neuron with hand picked weights that I chose because they made sense for the example. There is no learning. No training. No data.

If the weights had been different, the neuron would have given a different answer for the same inputs. And here is the important part. A real neural network is a network of these neurons, with weights that are learned automatically from data, not picked by me.

That is where the next project goes. Building a small network of neurons, and then teaching that network to learn its own weights from examples instead of me writing them by hand.

The math gets more interesting. The code stays small for a while longer. By the end of the next project, the network will be teaching itself.

---

## What I took away from this

For a long time I thought of a neural network as a complicated black box. After writing one neuron from scratch, I realised the complication is not in the individual pieces. Each piece is small. Each piece is a few lines of math.

The complication comes from putting many of these small pieces together and letting them adjust their own weights. That is where the magic actually lives, and that is where the next project picks up.

For now, this small script sits on my laptop. It runs in a tenth of a second. It makes a sensible decision about whether to bring an umbrella. It is the smallest possible neural network. And every architecture I have written about, from RNNs to Transformers to LLMs, is built from this one idea repeated and arranged in clever ways.

I will say it again because I want to remember it. One neuron. Inputs, weights, bias, activation. That is the unit. Everything else is composition.

---

*This is the first project in my build series. The code for every project lives in a small GitHub repo that grows with the series. If you want to follow along, clone it, run it, modify it.*

*What was the moment something in machine learning finally clicked for you? Drop it in the comments, I read every one.*