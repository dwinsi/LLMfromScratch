# LLM from Scratch

A learning project where I build the path from a single neuron to a tiny LLM, one project at a time. Each project is small, runnable on a laptop, and accompanied by a written walkthrough.

This repo grows alongside my build series on LinkedIn. Each folder corresponds to one project. Each project has its own short README pointing back to the article it accompanies.

## Goal

Most people learning about LLMs either stop at theory or jump to using frameworks that hide what is happening. I wanted to do something different. Build every piece from scratch, in plain Python, with the math visible. Start with the smallest possible neural network and add one layer of complexity at a time until I have trained my own small language model end to end.

This is not a course. It is a working notebook of what I have built and what I have learned along the way.

## Structure

Each project lives in its own folder, numbered in the order they were built.

```
01-neuron/        A single neuron forward pass, no learning yet
02-network/       A small network with backpropagation (coming soon)
03-pytorch/       Same network rebuilt with PyTorch (coming soon)
04-rnn/           A character level RNN (coming soon)
05-attention/     Adding attention to the RNN (coming soon)
06-transformer/   A tiny Transformer block from scratch (coming soon)
07-mini-llm/      Training a small language model end to end (coming soon)
```

## Running the code

All projects use Python 3 with numpy at minimum. Later projects also use PyTorch.

```
pip install numpy
```

Then run any project directly.

```
cd 01-neuron
python neuron.py
```

## Reading along

The build series is published on LinkedIn. Each project article walks through the code, the math, and what I learned writing it. The articles are written in plain English with no assumed background beyond basic programming.

The first article: Build a Single Neuron From Scratch.

## Why from scratch

Frameworks like PyTorch and TensorFlow are excellent. They are also opaque if you have never built the pieces underneath them yourself. Writing each piece from scratch first, even when it is slower and less elegant than the framework version, makes the framework versions easier to understand later.

This repo is the long way around, on purpose.

## License

MIT. Use this for learning, teaching, or building on. Attribution appreciated but not required.
