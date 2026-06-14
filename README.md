# LLM from Scratch

A learning project where I build the path from a single neuron to a mini language model, one project at a time. Each project is small, runnable on a laptop, and accompanied by a written walkthrough published on LinkedIn.

This is not a course. It is a working notebook of what I built and what I learned along the way.

## Goal

I wanted to understand how LLMs actually work, not just use them. The only way I found to do that was to build every piece from scratch, with the math visible, starting from the smallest possible neural network and adding one layer of complexity at a time until I had trained a language model end to end on my own machine.

Frameworks like PyTorch are excellent. They are also opaque if you have never built the pieces underneath them yourself. Every abstraction in PyTorch now has a concrete referent in something I built by hand in an earlier project.

## The series

```
01-neuron/        A single neuron. Forward pass only. Weights picked by hand.
02-network/       A small network that learns. Backpropagation from scratch.
03-pytorch/       Same network rebuilt in PyTorch. Train/validation split.
04-rnn/           A word-level RNN on a custom weather corpus. numpy only.
05-attention/     RNN with attention rebuilt in PyTorch. Q, K, V hand-rolled.
06-transformer/   A complete Transformer block. numpy for math, PyTorch for training.
07-mini-llm/      Four stacked Transformer blocks. A mini language model end to end.
```

## Progress across the series

| Project | Parameters | Final loss | Key concept |
|---|---|---|---|
| 01 neuron | 1 neuron | n/a (no training) | Forward pass, sigmoid |
| 02 network | 61 | 0.0004 | Backpropagation, MSE loss |
| 03 pytorch | 61 | varies | PyTorch, validation split |
| 04 rnn | 14,093 | 1.83 | Sequence prediction, cross-entropy |
| 05 attention | 26,381 | run to verify | Q, K, V attention |
| 06 transformer | 43,405 | 0.01 | Multi-head attention, residuals, layer norm |
| 07 mini-llm | 159,558 | 0.00 | Stacked blocks, batching, cosine annealing |

## Requirements

Projects 1 through 4 use numpy only.

```
pip install numpy matplotlib
```

Projects 5 through 7 use PyTorch.

```
pip install torch matplotlib
```

Python 3.8 or higher. All projects run on CPU. Projects 5 through 7 will use a GPU automatically if one is available (CUDA on NVIDIA, MPS on Apple Silicon).

## Running any project

```
cd 01-neuron
python neuron.py
```

Each folder has its own README with expected output and what to look for when running.

## The overfitting arc

One thread runs through the entire series. Overfitting appears early and gets addressed gradually.

Project 2 plants the question. Project 3 introduces the validation split. Project 4 shows overfitting in generated text. Project 5 makes it explicit. Project 6 adds dropout. Project 7 adds batching, gradient clipping and cosine annealing.

The honest conclusion at the end of Project 7: overfitting on a dataset this small is inevitable. The real solution is more training data. GPT-3 trained on 45 billion tokens. This model trained on 499 sequences of 4 words each. The architecture is the same. The scale is not.

## The weather corpus

Projects 4 through 6 use a small custom weather corpus written specifically for the series (30 sentences, 77 vocabulary). Project 7 uses an expanded version (91 sentences, 198 vocabulary). Both are included in their respective project folders.

## Why from scratch

The build-from-scratch approach is not about avoiding frameworks. It is about making frameworks readable. Once you have written backpropagation by hand, `loss.backward()` is not a black box. Once you have written a Q, K, V attention loop in numpy, `nn.MultiheadAttention` is not mysterious.

This repo is the long way around, on purpose.

## References:

[CAPE: Encoding Relative Positions with Continuous Augmented Positional Embeddings](https://arxiv.org/abs/2106.03143)
[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
[Learning Positional Embeddings for Coordinate-MLPs](https://arxiv.org/abs/2112.11577)
