# HiPPO Explained Simply — A Beginner's Guide

*Understanding how AI models remember long sequences, starting from zero. Math background needed: up to 12th standard (basic calculus, vectors, and matrices).*

---

## What this guide is about

Modern AI models like Mamba can read very long texts efficiently. The secret behind them is a clever idea called **HiPPO**. This guide explains HiPPO from the ground up, in plain language, with small examples you can follow by hand.

We will avoid heavy notation. Whenever a formula appears, we will explain every symbol in words first.

Here is the one question HiPPO answers:

> **If a model can only remember a small, fixed amount of information, what is the smartest way to summarize everything it has read so far?**

That's it. Everything below builds toward answering this well.

---

## Table of Contents

1. [The core problem — a memory that never grows](#1-the-core-problem)
2. [The big idea — store a summary, not the raw data](#2-the-big-idea)
3. [A warm-up: summarizing numbers](#3-a-warm-up-summarizing-numbers)
4. [Building blocks — the "shapes" we summarize with](#4-building-blocks)
5. [Legendre polynomials — the shapes HiPPO uses](#5-legendre-polynomials)
6. [Putting it together — the HiPPO recipe](#6-the-hippo-recipe)
7. [A complete worked example](#7-a-complete-worked-example)
8. [How the model updates its memory each step](#8-updating-memory)
9. [Why the memory never blows up](#9-why-memory-never-blows-up)
10. [Where HiPPO fits — the road to Mamba](#10-where-hippo-fits)
11. [Quick reference and further reading](#11-quick-reference)

---

## 1. The Core Problem

Imagine you are reading a very long book, one word at a time. Someone will stop you at any random point and ask: *"What has happened in the story so far?"*

You have two options.

**Option A — remember every word.** This works perfectly, but it is impossible in practice. If the book has a million words, you would need a million slots of memory. And every new word makes the memory bigger. This is roughly how "attention" (the mechanism inside GPT-style models) works, and it is why those models get slow and expensive on long texts.

**Option B — keep a short summary.** You keep, say, a small notepad with room for only 64 numbers. As you read each word, you update the notepad. The notepad never grows — it always holds exactly 64 numbers, whether you've read 10 words or 10 million.

Option B is what HiPPO does. The obvious worry is: *won't you lose information?* Yes, some. The whole genius of HiPPO is figuring out **how to summarize so that you lose as little as possible** — so that from those 64 numbers, you can reconstruct the story surprisingly well.

---

## 2. The Big Idea

Here is the shift in thinking that makes HiPPO work.

Instead of storing the actual words (or actual data values), HiPPO stores a **description of the overall shape** of the history.

Think about how you might describe a graph to a friend over the phone. You wouldn't read out every single point. You'd say things like:

- "On average it sits around 5." (the overall level)
- "It's generally rising." (the trend)
- "It curves upward in the middle." (the bend)
- "There's a little wiggle near the end." (finer detail)

Each of those statements is **one number** capturing **one aspect** of the shape. Put enough of them together, and your friend can draw a graph that looks almost exactly like the original — without ever hearing the individual points.

HiPPO does exactly this. It stores a handful of numbers, where:
- the first number captures the average level,
- the second captures the trend (rising or falling),
- the third captures the curve,
- and so on, each number adding finer detail.

These numbers are called **coefficients**, and together they are the model's memory (its "hidden state").

---

## 3. A Warm-Up: Summarizing Numbers

Before functions, let's summarize a simple list of numbers, because the idea is the same.

Suppose you see the numbers: **2, 4, 6, 8**.

You could describe them with just two facts:
- **Average = 5** (the level: (2+4+6+8)/4 = 5)
- **Trend = +2 each step** (they go up by 2 each time)

From these two facts — "start around 5, rising by 2" — you can rebuild the list very closely: 2, 4, 6, 8. You compressed four numbers into two, with no loss, because the data was simple (a straight line).

If the data were more complicated (curving up and down), you'd need more facts: a curve number, a wiggle number, and so on. Each extra fact captures a finer detail.

**This is the entire philosophy of HiPPO.** Store a few well-chosen summary numbers. Simple histories need few; complex histories need more. The model typically keeps 16 to 64 such numbers.

The rest of this guide answers two questions:
1. What exactly are the "aspects" we measure? (Answer: Legendre polynomials — Section 5.)
2. How do we update the summary numbers as new data arrives, without redoing everything? (Answer: a simple matrix multiplication — Section 8.)

---

## 4. Building Blocks

To describe the shape of a history, we need a set of standard "reference shapes." We measure how much of each reference shape is present in our data.

### An analogy you already know: music

Any sound can be broken into pure tones — a low hum, a middle note, a high note, and so on. A sound engineer stores the *volume of each tone* rather than the raw sound wave. A few tone-volumes can reconstruct the sound. This is the idea behind audio compression (like MP3).

The "pure tones" are the reference shapes. The "volume of each tone" is a coefficient.

### HiPPO's reference shapes

HiPPO uses reference shapes that are good for describing *trends over time*. They are:

- **Shape 0:** a flat line → measures the **average**
- **Shape 1:** a straight slope → measures the **trend** (up or down)
- **Shape 2:** a gentle U or ∩ curve → measures the **curvature**
- **Shape 3:** an S-shaped wiggle → measures **finer oscillation**
- ... and so on, each one wigglier than the last.

These specific shapes are called **Legendre polynomials**. Let's meet them properly.

---

## 5. Legendre Polynomials

Don't be scared by the name. A "polynomial" is just an expression like $x$, or $3x^2 - 1$, or $x^3 - x$. The Legendre polynomials are a specific, famous list of them.

### The first four

Here they are (we write the variable as $x$, ranging from $-1$ to $1$):

| Name | Formula | What shape it is | What it measures |
|------|---------|------------------|------------------|
| $P_0$ | $1$ | a flat horizontal line | the average level |
| $P_1$ | $x$ | a straight diagonal line | the trend (slope) |
| $P_2$ | $\tfrac{1}{2}(3x^2 - 1)$ | a U-shaped curve | the curvature |
| $P_3$ | $\tfrac{1}{2}(5x^3 - 3x)$ | an S-shaped wiggle | finer wiggle |

If you plotted these from $x = -1$ to $x = 1$, you'd see: a flat line, a slope, a U, and an S. Each is "wigglier" than the one before.

### Where do these formulas come from? (Building them yourself)

You don't have to take these on faith. There's a simple procedure to build them, one at a time. The rule is:

> **Start with the simple powers $1, x, x^2, x^3, \dots$ and adjust each one so it is "independent" from all the previous ones.**

"Independent" has a precise meaning we'll explain in a moment. Let's build the first three.

**Building $P_0$:** Just take the simplest thing, a constant. Set $P_0 = 1$. Done.

**Building $P_1$:** Start with $x$. We need to check whether $x$ is already independent from $P_0 = 1$. The test for independence is: *multiply the two shapes together, then find the area under the resulting curve from $-1$ to $1$. If the area is zero, they are independent.*

For $x$ and $1$: multiply to get $x$, and the area under $x$ from $-1$ to $1$ is zero (the positive half and negative half cancel). So $x$ is already independent from $1$. Set $P_1 = x$.

**Building $P_2$:** Start with $x^2$. Now check it against both $P_0$ and $P_1$.

- Against $P_0 = 1$: multiply to get $x^2$; the area under $x^2$ from $-1$ to $1$ is $\tfrac{2}{3}$ (not zero!). So $x^2$ is **not** independent from the constant — it has some "average level" mixed in. We must subtract that out.
- The amount to subtract is $\tfrac{2/3}{2} = \tfrac{1}{3}$ (the mixed-in amount divided by the "size" of $P_0$, which is 2).

So we form $x^2 - \tfrac{1}{3}$.

- Against $P_1 = x$: multiply $x^2$ by $x$ to get $x^3$; the area under $x^3$ from $-1$ to $1$ is zero. Good, nothing to subtract there.

So the raw shape is $x^2 - \tfrac{1}{3}$. By tradition we rescale it so its value at $x = 1$ equals $1$: at $x=1$, $x^2 - \tfrac13 = \tfrac23$, so multiply by $\tfrac32$ to get $\tfrac{3}{2}x^2 - \tfrac{1}{2} = \tfrac{1}{2}(3x^2 - 1)$. That's $P_2$. ✓

This procedure (called **Gram–Schmidt**, if you want the technical name) generates every Legendre polynomial. Each new one is the next power, cleaned of any overlap with the earlier ones.

### Why "independence" matters

When two reference shapes are independent (their product has zero area), it means each captures **completely separate information**. Knowing the average tells you nothing about the trend; knowing the trend tells you nothing about the curve. No wasted storage — every summary number pulls its own weight.

This is exactly like the three directions in space (left-right, forward-back, up-down): moving in one direction doesn't change your position in the others. They're independent, so three numbers fully describe your position. Legendre polynomials are the "independent directions" for describing shapes.

---

## 6. The HiPPO Recipe

Now we combine the pieces. At any moment, HiPPO does three things.

**Step 1 — Look at the history.** All the data seen so far, from the start up to now.

**Step 2 — Measure how much of each reference shape is present.** For each Legendre shape (average, trend, curve, wiggle...), compute one number: how strongly does the history match that shape? This "matching" is done by multiplying the history by the shape and finding the area — the same independence test from Section 5, now used to measure.

**Step 3 — Store those numbers.** The collection of numbers is the memory. If we use 4 shapes, the memory is 4 numbers. If we use 64 shapes, it's 64 numbers.

### The "weighting" — which moments count most

There's one more choice: **when we look at the history, do all past moments count equally, or do recent moments count more?**

HiPPO's main version (called **LegS**, for "Legendre Scaled") counts **all of history equally**. Whether something happened at the very beginning or just now, it gets equal say in the summary. As time goes on, the window simply stretches to always cover everything from the start until now.

This equal-weighting is what gives HiPPO its special power: it can remember things from very far back just as reliably as recent things. It doesn't have a fixed "forgetting horizon."

(There's an alternative version, **LegT**, that only looks at a fixed recent window and forgets everything older — like a memory buffer of fixed size. But LegS, with its equal weighting over all history, is the one used in modern models.)

---

## 7. A Complete Worked Example

Let's make everything concrete with a tiny example using **3 numbers** of memory (3 Legendre shapes: average, trend, curve).

Suppose the history so far is a simple **ramp** — a signal that rises steadily from 0 up to 1. (Picture a straight line going up.)

We compute the three summary numbers by measuring how much of each shape is present.

**Number 0 — the average (using the flat shape $P_0$):**
The average value of a ramp that goes evenly from 0 to 1 is simply the midpoint:
$$\text{average} = \frac{1}{2} = 0.5.$$
So the first memory number is **0.5**. ✓ (Makes sense — a ramp from 0 to 1 sits at 0.5 on average.)

**Number 1 — the trend (using the sloped shape $P_1$):**
A ramp is steadily rising, so it should have a clear positive trend. Working out the measurement gives approximately **0.29**. A positive number confirms: the signal is rising. ✓

**Number 2 — the curve (using the U-shaped $P_2$):**
A ramp is a perfectly straight line — it has no bend at all. So the curve measurement comes out to **0**. ✓ (No curvature, exactly as expected.)

So the memory is:
$$\text{memory} = (\,0.5,\ \ 0.29,\ \ 0\,).$$

Read it in plain English: *"On average 0.5, steadily rising, with no bend."* That is a perfect description of a ramp — captured in just three numbers. And from these three numbers, you could redraw the ramp almost exactly.

This is the whole point: **three numbers faithfully summarized the entire history**, because we chose smart reference shapes.

---

## 8. Updating Memory

Here's the practical magic. When a new data point arrives, HiPPO does **not** re-scan the whole history. It updates the memory with one small calculation.

### The update rule in words

$$\text{new memory} = A \times (\text{old memory}) + B \times (\text{new input})$$

- **Old memory** = the summary numbers you already had.
- **New input** = the new data point that just arrived.
- **A** = a fixed grid of numbers (a matrix) that says how the old summary should shift as time moves forward.
- **B** = a small list of numbers that says how the new input gets blended in.

That's it. Multiply the old memory by A, add B times the new input, and you have the updated memory. It's just a bit of arithmetic — fast and constant-sized, no matter how long the history.

### What A and B look like (for our 3-number example)

$$
A = \begin{pmatrix} 1 & 0 & 0 \\ 1.73 & 2 & 0 \\ 2.24 & 3.87 & 3 \end{pmatrix},
\qquad
B = \begin{pmatrix} 1 \\ 1.73 \\ 2.24 \end{pmatrix}
$$

(Those decimals are square roots: $1.73 \approx \sqrt{3}$, $2.24 \approx \sqrt{5}$, $3.87 \approx \sqrt{15}$. They come from the Legendre shapes.)

### The remarkable fact

Here is what makes HiPPO beautiful, and it's worth stating plainly:

> **Nobody chose the numbers in A and B by hand.** They come out automatically from the math, once you decide (1) to store Legendre-shape summaries and (2) to weight all history equally.

In most AI models, the numbers inside matrices are learned by trial and error during training. But HiPPO's A and B are **derived** — they are the unique correct answer to the question "how does the summary change as time moves forward?" There is nothing to tune. This is why HiPPO works so well right from the start, before any training.

### Notice the triangle shape

Look at matrix A: everything **above** the diagonal is zero. This "lower-triangular" shape isn't an accident. It means information flows one way: the coarse summary (average) influences the finer summaries (trend, curve), but not the reverse. Coarse-to-fine, never backward. This falls directly out of the way the Legendre shapes relate to each other.

---

## 9. Why Memory Never Blows Up

There's a danger with any repeated calculation: if you multiply by something over and over, the result might explode to infinity (or shrink to nothing). Since HiPPO applies matrix A once per data point — thousands of times for a long sequence — we need to be sure the memory stays sensible.

### The key idea: eigenvalues

Every matrix has special numbers attached to it called **eigenvalues**. Here's what they mean in plain terms:

> An eigenvalue tells you the "stretch factor" of the matrix in a certain direction. If you apply the matrix over and over, the result grows or shrinks according to that stretch factor.

The rule for a repeated process like ours:
- If the stretch factor is **bigger than 1** in size → the memory **explodes** to infinity. Bad.
- If the stretch factor is **less than 1** in size → the memory **fades** toward zero in a controlled way. Good and stable.

(For the continuous-time math HiPPO actually uses, the exact condition is that the eigenvalues should be **negative** — but the spirit is the same: they must point toward "shrink," not "grow.")

### HiPPO is stable by design

For our example matrix A, the eigenvalues turn out to be exactly the numbers on the diagonal: **1, 2, 3** — and in the actual HiPPO equation these enter with a minus sign, making them **−1, −2, −3**. All negative. Every direction shrinks in a controlled way. The memory can never explode, no matter how long the sequence.

This isn't luck. The lower-triangular shape of A (Section 8) makes the eigenvalues easy to read (they're just the diagonal), and the way HiPPO is built guarantees they're all negative. **Stability is baked in.**

---

## 10. Where HiPPO Fits

HiPPO by itself is a theory of memory. Two famous models build on it.

**S4** (2021) took HiPPO and made it *fast*. The challenge was that computing with matrix A repeatedly can be slow. S4 found a clever way to reorganize A so the computation runs quickly even on very long sequences (16,000+ steps). S4 kept HiPPO's excellent memory, added speed.

**Mamba** (2023) added one crucial upgrade: **selectivity**.

In plain HiPPO, the matrices A, B (and a step-size setting) are the *same for every input*. The word "the" is processed exactly like the word "explosion." That's wasteful — some words matter far more than others.

Mamba makes B and the step-size **depend on the input word itself**. Now:
- an important, rare word → gets written into memory strongly, and lingers,
- a common filler word → barely touches the memory.

The model *learns to decide what to remember*. This is what makes Mamba competitive with (and sometimes better than) GPT-style models, while using far less memory on long texts.

The one-line summary of the whole journey:

> **HiPPO decides what the memory should be. S4 makes it fast. Mamba makes it smart about what to keep.**

---

## 11. Quick Reference

**The five ideas to remember:**

1. **The problem:** summarize an ever-growing history into a fixed, small set of numbers.
2. **The trick:** store the *shape* of the history (average, trend, curve, wiggle...) instead of raw data.
3. **The shapes:** Legendre polynomials — a standard set of independent reference shapes, built by cleaning up the simple powers $1, x, x^2, x^3, \dots$
4. **The update:** new memory $= A \times$ old memory $+ B \times$ new input. The matrices A and B are *derived from the math*, not tuned.
5. **The stability:** A's eigenvalues are all negative, so the memory fades gracefully and never explodes.

**What each memory number means (with 4 shapes):**

| Memory slot | Reference shape | Captures |
|-------------|-----------------|----------|
| 1st | flat line | average level |
| 2nd | slope | trend (rising/falling) |
| 3rd | U-curve | curvature |
| 4th | S-wiggle | finer oscillation |

**Further reading (from gentle to advanced):**

- **A Visual Guide to Mamba and State Space Models** by Maarten Grootendorst (blog) — lots of pictures, very beginner-friendly. Start here.
- **The Annotated S4** by Sasha Rush and Sidd Karamcheti (blog) — walks through real code line by line.
- **HiPPO: Recurrent Memory with Optimal Polynomial Projections** by Gu, Dao, Ermon, Rudra, and Ré (2020), the original research paper (arXiv:2008.07669) — for when you want the full mathematics.
- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** by Gu and Dao (2023), arXiv:2312.00752 — the model built on HiPPO.

**A note on the math:** This guide kept things intuitive. If you'd like the full rigorous derivation — with the calculus that produces A and B, the exact integrals, and formal proofs — that lives in the companion document written at a more advanced level. Everything here is a faithful, simplified picture of that same mathematics.

---

*You now understand HiPPO: what problem it solves, how it summarizes history using reference shapes, how it updates its memory with a simple rule, and why that memory stays stable. That's the foundation beneath some of the most capable AI models today.*
