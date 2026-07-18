# HiPPO and State Space Models: A Step-by-Step Mathematical Guide

This document derives HiPPO from first principles. Every formula is preceded by a plain-English explanation of what it means and why it appears. If you have read `HiPPO_for_beginners.md`, you already have the intuition. This document fills in the mathematics behind it.

**What you need:** comfort with basic algebra, derivatives, and integrals. Everything more advanced (inner products of functions, orthogonality, matrix exponentials) is built up from scratch here.

**What you will understand by the end:** where the HiPPO matrices A and B actually come from, why they are the unique correct answer, and how HiPPO connects to S4 and Mamba.

---

## Table of Contents

1. [The problem we are solving](#1-the-problem-we-are-solving)
2. [The object we are building: state space models](#2-the-object-we-are-building-state-space-models)
3. [Summarizing history as polynomial coefficients](#3-summarizing-history-as-polynomial-coefficients)
4. [What orthogonal means and why it matters](#4-what-orthogonal-means-and-why-it-matters)
5. [Building the Legendre polynomials from scratch](#5-building-the-legendre-polynomials-from-scratch)
6. [The HiPPO-LegS setup: scaling to the whole history](#6-the-hippo-legs-setup-scaling-to-the-whole-history)
7. [Deriving the HiPPO matrices A and B](#7-deriving-the-hippo-matrices-a-and-b)
8. [A fully worked example with three coefficients](#8-a-fully-worked-example-with-three-coefficients)
9. [Discretization: from continuous equations to token steps](#9-discretization-from-continuous-equations-to-token-steps)
10. [Stability: why the memory never explodes](#10-stability-why-the-memory-never-explodes)
11. [From HiPPO to S4 to Mamba](#11-from-hippo-to-s4-to-mamba)
12. [Symbol reference](#12-symbol-reference)
13. [References](#13-references)

---

## 1. The problem we are solving

A recurrent model reads a sequence of inputs one step at a time, for example one word per step. At every step it updates a hidden state, which is a fixed list of N numbers. The hidden state is all the model is allowed to keep from the past. When the next word arrives, the model uses the hidden state plus the new word to predict the next word and update the state.

The question HiPPO answers is:

> Given that we can only store N numbers, what is the mathematically optimal way to use those N numbers to summarize everything we have read so far?

The answer HiPPO gives is: store the coefficients of the best polynomial approximation of the history. This turns out to be optimal in a precise mathematical sense. And the rule for updating those coefficients when a new input arrives turns out to be a simple matrix-vector multiplication:

```text
new state = A * old state + B * new input
```

The matrices A and B are not guessed or learned. They are derived. This document shows the derivation.

---

## 2. The object we are building: state space models

### The equations

A state space model is a pair of equations that describes how a system evolves over time. In the continuous-time version:

```text
Equation 1 (state update):  d/dt x(t) = A * x(t) + B * u(t)

Equation 2 (output):        y(t) = C * x(t) + D * u(t)
```

Reading these in plain English:

- `u(t)` is the input at time t (a single number, for example the current word embedding).
- `x(t)` is the hidden state at time t (a list of N numbers).
- `d/dt x(t)` is the rate of change of the state (how fast each of the N numbers is changing right now).
- `y(t)` is the output (the prediction).
- `A` is an N-by-N matrix that describes how the state affects its own rate of change.
- `B` is an N-by-1 column of numbers that describes how the current input enters the state.
- `C` is a 1-by-N row that reads from the state to produce the output.
- `D` is a direct feedthrough term, often set to zero.

This notation comes from control theory, where engineers write down equations for physical systems like springs, circuits, and aircraft. The state x encodes "everything about the past that is relevant for predicting the future."

### The HiPPO question

In classical control theory, you know the physics and you derive A and B from it. HiPPO flips this around:

> We specify what the state x should mean (it should be the best polynomial summary of the input history), and then we derive the unique A and B that make the state evolve correctly.

The matrix A is not a free parameter. It is a theorem.

---

## 3. Summarizing history as polynomial coefficients

### Setting up the problem

At any time t, we have seen the input from the start up to now. Call this the history: all values `u(s)` for `s` between 0 and t.

We want to compress this entire history into N numbers. Here is how.

**Step 1: choose a weighting.** Decide how much each past moment counts. For HiPPO-LegS, all past moments count equally: a moment at the very beginning is just as important as a moment that happened recently. This is the "uniform" weighting, written mathematically as:

```text
weight(s) = 1/t   for s between 0 and t
            0     otherwise
```

The `1/t` ensures the total weight over the window sums to 1 (it is a probability distribution).

**Step 2: choose a basis.** Pick N reference shapes, called basis functions. Call them `g_0, g_1, ..., g_{N-1}`. These are the "pure shapes" that we will measure in the history, analogous to the pure tones in the music analogy from the beginner guide.

**Step 3: project.** For each basis function, compute one number: how strongly does the history match that shape? This number is:

```text
c_n(t) = integral from 0 to t of:  u(s) * g_n(s) * weight(s)  ds
```

In plain English: multiply the history by the reference shape at each moment, then add up the products over the whole window. A large positive number means the history looks a lot like that shape. A number near zero means the shape is not present. A negative number means the history is the mirror image of the shape.

The collection of N numbers, `[c_0(t), c_1(t), ..., c_{N-1}(t)]`, is the hidden state. This is the key insight of HiPPO: the hidden state is defined as the projection coefficients of the history onto the chosen basis.

**Why is this the best compression?** Because for an orthonormal basis (explained in the next section), these projection coefficients give the polynomial approximation that minimizes the total squared error between the actual history and its reconstruction. In mathematical terms, it is the optimal projection in a Hilbert space.

### The moving window problem

Here is what makes HiPPO nontrivial. As t increases, the window `[0, t]` grows. The weights change. The basis functions must rescale to cover the new window. Therefore the coefficients `c_n(t)` change continuously with time, even if no new input has arrived.

HiPPO's entire mathematical content is computing how fast the coefficients change, i.e., computing `d/dt c_n(t)`, and showing that this derivative depends only on the current coefficients and the current input:

```text
d/dt c_n(t) = (some linear combination of c_0, ..., c_{N-1}) + (some multiple of u(t))
```

This is exactly `d/dt x(t) = A * x(t) + B * u(t)`. The whole derivation is in Section 7.

---

## 4. What orthogonal means and why it matters

### Functions behave like vectors

You know how to add two vectors and scale them. You can do the same with functions: `f + g` means "at every point s, add the two values". This means functions form a vector space, just like arrows in 3D.

Vectors also have dot products. The dot product of two 3D vectors `a = (a1, a2, a3)` and `b = (b1, b2, b3)` is:

```text
a . b = a1*b1 + a2*b2 + a3*b3
```

For functions, we replace the sum with an integral. The inner product of two functions f and g over the interval `[a, b]` with a weight function `w(s)` is:

```text
<f, g> = integral from a to b of:  f(s) * g(s) * w(s)  ds
```

This behaves exactly like a dot product. Two functions are orthogonal if `<f, g> = 0`, meaning their inner product is zero.

### Why orthogonality means no wasted information

In 3D space, the three coordinate directions (left-right, forward-back, up-down) are orthogonal. This means knowing how far you are in the left-right direction tells you nothing about how far you are in the forward-back direction. The three measurements are completely independent, and together they fully describe your position.

When reference shapes are orthogonal, the same independence holds: knowing the average level of the history tells you nothing about the trend. Knowing the trend tells you nothing about the curvature. Each coefficient captures a completely separate aspect. No storage is wasted.

There is also a computational advantage. When the basis is orthonormal (orthogonal and each function has unit norm), projecting is as simple as computing one integral per coefficient. You do not need to solve a system of equations to find the best approximation: the projection coefficients are automatically the right answer.

### The norm and normalization

The norm of a function is the square root of its inner product with itself:

```text
||f|| = sqrt(<f, f>) = sqrt( integral of f(s)^2 * w(s) ds )
```

A function is normalized if its norm equals 1. An orthonormal family satisfies:

```text
<g_n, g_m> = 1   if n equals m
<g_n, g_m> = 0   if n does not equal m
```

---

## 5. Building the Legendre polynomials from scratch

### What they are

The Legendre polynomials are the orthogonal family you get when you apply Gram-Schmidt to the simple powers `1, s, s^2, s^3, ...` using the uniform weight `w(s) = 1` over the interval `[-1, 1]`.

### The Gram-Schmidt process

Gram-Schmidt takes a sequence of functions that are not orthogonal and produces a sequence that is. The idea is: take each new function, subtract off the parts that overlap with the previous ones, and what remains is orthogonal to all of them.

For polynomials, the procedure is:

1. Start with `p_0 = 1`.
2. For each new power `s^n`, subtract its projections onto all previously built polynomials.
3. Normalize.

Here is the full calculation for the first four:

**Building P0:**

Start with the constant function `1`. Nothing to subtract. Set `P_0 = 1`.

**Building P1:**

Start with `s`. Check independence from `P_0 = 1` by computing their inner product:

```text
<s, 1> = integral from -1 to 1 of:  s * 1  ds
       = [s^2/2] from -1 to 1
       = 1/2 - 1/2
       = 0
```

The inner product is zero, so `s` is already orthogonal to `1`. Set `P_1 = s`.

**Building P2:**

Start with `s^2`. Check against P0:

```text
<s^2, 1> = integral from -1 to 1 of:  s^2  ds
         = [s^3/3] from -1 to 1
         = 1/3 - (-1/3)
         = 2/3
```

This is not zero. `s^2` has 2/3 of `P_0` mixed in. Subtract it out. The overlap amount is the inner product divided by the norm squared of `P_0`:

```text
norm^2 of P_0 = integral from -1 to 1 of:  1^2  ds = 2

overlap = (2/3) / 2 = 1/3
```

So subtract `(1/3) * P_0` from `s^2`:

```text
s^2 - (1/3) * 1 = s^2 - 1/3
```

Check against P1 = s:

```text
<s^2, s> = integral from -1 to 1 of:  s^3  ds = 0   (odd function over symmetric interval)
```

Already orthogonal to P1. No further subtraction needed.

By convention, rescale so the value at s = 1 equals 1. At s = 1, `s^2 - 1/3 = 2/3`. Multiply by `3/2`:

```text
P_2 = (3/2) * (s^2 - 1/3) = (1/2)(3s^2 - 1)
```

Check: at s = 1, `(1/2)(3 - 1) = 1`. Correct.

**Building P3:**

Start with `s^3`. Check against P0:

```text
<s^3, 1> = integral from -1 to 1 of:  s^3  ds = 0   (odd function)
```

Check against P1 = s:

```text
<s^3, s> = integral from -1 to 1 of:  s^4  ds = [s^5/5] from -1 to 1 = 2/5
```

Not zero. Subtract:

```text
overlap = (2/5) / (2/3) = 3/5     (dividing by norm^2 of P_1 = integral of s^2 = 2/3)

s^3 - (3/5) * s
```

Check against P2. The integral of `s^3 * (s^2 - 1/3)` over `[-1, 1]` is zero by symmetry. No further subtraction.

Rescale so the value at s = 1 equals 1. At s = 1: `1 - 3/5 = 2/5`. Multiply by `5/2`:

```text
P_3 = (5/2)(s^3 - 3s/5) = (1/2)(5s^3 - 3s)
```

The four standard Legendre polynomials are:

```text
P_0 = 1
P_1 = s
P_2 = (1/2)(3s^2 - 1)
P_3 = (1/2)(5s^3 - 3s)
```

![The first four Legendre polynomials plotted from s = -1 to s = 1](figures/1_legendre_polynomials.png)

### Key properties we will need later

**Orthogonality and norms.** The inner product of two Legendre polynomials under uniform weight on `[-1, 1]` is:

```text
<P_n, P_m> = integral from -1 to 1 of:  P_n(s) * P_m(s)  ds

           = 2 / (2n + 1)    if n equals m
           = 0                if n does not equal m
```

This means the normalized versions are `sqrt((2n+1)/2) * P_n(s)`, and the factor `sqrt(2n+1)` will appear throughout the HiPPO matrices.

**Boundary values.** `P_n(1) = 1` for all n. This is true by the normalization convention. Also `P_n(-1) = (-1)^n`.

**Derivative identity.** A formula we will need in the derivation:

```text
P_n'(s) = (2n-1)*P_{n-1}(s) + (2n-5)*P_{n-3}(s) + ...
```

Written more compactly: the derivative of `P_n` equals a sum of `(2k+1) * P_k(s)` for all k less than n where n and k differ by an odd number.

**The three-term recurrence.** Each polynomial is linked to its neighbors:

```text
(n+1) * P_{n+1}(s) = (2n+1) * s * P_n(s) - n * P_{n-1}(s)
```

This "only neighbors interact" property is why the HiPPO matrix A turns out to be lower-triangular rather than dense.

---

## 6. The HiPPO-LegS setup: scaling to the whole history

### The measure

HiPPO-LegS uses the uniform measure over the whole history `[0, t]`:

```text
weight(s, t) = 1/t   for s in [0, t]
```

The `1/t` normalizes the total weight to 1. This is a probability distribution that weights every moment in history equally. As t grows, the window stretches to include more history, but every moment in it keeps equal weight.

This equal weighting gives HiPPO-LegS a key property: it does not privilege any particular time scale. A pattern that happened 1 second ago is treated with the same importance as a pattern that happened 1000 seconds ago. This is what the paper calls "timescale invariance."

### Rescaling the polynomials to the window

The Legendre polynomials live on `[-1, 1]`. We need them to live on `[0, t]`. The substitution that maps `s` in `[0, t]` to `xi` in `[-1, 1]` is:

```text
xi = (2s/t) - 1
```

When `s = 0`, `xi = -1`. When `s = t`, `xi = 1`. This is just a linear stretch.

The basis functions we use are the Legendre polynomials composed with this substitution, and then scaled by `sqrt(2n+1)` to make them orthonormal under the measure `weight(s, t) = 1/t`:

```text
g_n(s, t) = sqrt(2n+1) * P_n( (2s/t) - 1 )   for s in [0, t]
```

You can verify these are orthonormal:

```text
integral from 0 to t of:  g_n(s,t) * g_m(s,t) * (1/t)  ds

Substituting xi = (2s/t) - 1, so ds = (t/2) d_xi:

= sqrt((2n+1)(2m+1)) * (1/t) * (t/2) * integral from -1 to 1 of:  P_n(xi) * P_m(xi)  d_xi

= sqrt((2n+1)(2m+1)) * (1/2) * (2/(2n+1)) * delta_{nm}

= delta_{nm}
```

That last step uses the Legendre orthogonality formula from Section 5.

### The hidden state

The hidden state components are the projection coefficients of the history onto these basis functions:

```text
x_n(t) = (1/t) * integral from 0 to t of:  u(s) * sqrt(2n+1) * P_n( (2s/t) - 1 )  ds
```

This is just the inner product of the history with the nth basis function under the uniform measure. The state `x(t) = [x_0(t), x_1(t), ..., x_{N-1}(t)]` is the optimal N-coefficient polynomial approximation of the history at time t.

The entire task now reduces to: differentiate `x_n(t)` with respect to t, and express the result in terms of `x_0(t), x_1(t), ...` and `u(t)`.

---

## 7. Deriving the HiPPO matrices A and B

This section carries out the derivation. The result is the update rule `d/dt x(t) = -(1/t) A x(t) + (1/t) B u(t)` where A and B are explicit matrices. The derivation uses the Leibniz integral rule, which we state first.

### The Leibniz rule

When differentiating an integral whose upper limit depends on t and whose integrand also depends on t, the rule is:

```text
d/dt  integral from a to t of:  f(s, t)  ds

= f(t, t)   +   integral from a to t of:  (partial/partial t) f(s, t)  ds
```

The first term accounts for the upper limit moving (the new boundary contribution). The second term accounts for the integrand itself changing with t. This is the key computational tool for the whole derivation.

### Step 1: write down what we are differentiating

Define:

```text
I_n(t) = (1/t) * integral from 0 to t of:  u(s) * P_n( z(s,t) )  ds

where z(s, t) = (2s/t) - 1
```

Then `x_n(t) = sqrt(2n+1) * I_n(t)`. We will compute `d/dt I_n(t)` and then multiply by `sqrt(2n+1)` at the end.

### Step 2: apply the product rule and Leibniz rule

The function `I_n(t)` is a product of `1/t` and the integral. Differentiating:

```text
d/dt I_n(t) = d/dt [ (1/t) * integral from 0 to t of:  u(s) * P_n(z)  ds ]

By the product rule:

= -(1/t^2) * integral from 0 to t of:  u(s) * P_n(z)  ds

+ (1/t) * d/dt [ integral from 0 to t of:  u(s) * P_n(z)  ds ]
```

For the second line, apply the Leibniz rule. The "boundary term" (from the upper limit moving) is the integrand evaluated at s = t:

```text
At s = t:  z(t, t) = (2t/t) - 1 = 1

So the boundary term is:  u(t) * P_n(1) = u(t) * 1 = u(t)
```

The "interior term" (from the integrand changing with t) involves the partial derivative of `P_n(z)` with respect to t:

```text
partial/partial t of P_n(z) = P_n'(z) * (partial z / partial t)

partial z / partial t = partial/partial t of  (2s/t - 1) = -2s/t^2
```

So the interior derivative is `P_n'(z) * (-2s/t^2)`.

Collecting everything:

```text
d/dt I_n(t) = -(1/t^2) * integral(u(s) P_n(z) ds)

            + (1/t) * u(t)

            + (1/t) * integral from 0 to t of:  u(s) * P_n'(z) * (-2s/t^2)  ds
```

Simplify by noting that `-2s/t^2 = -(1/t)(2s/t) = -(1/t)(z + 1)`:

```text
d/dt I_n(t) = -(1/t) * I_n(t) + (1/t) * u(t) - (1/t^2) * integral from 0 to t of:  u(s) * (z+1) * P_n'(z)  ds
```

### Step 3: expand (z+1) * P_n'(z) in the Legendre basis

We need to write `(z + 1) * P_n'(z)` as a linear combination of `P_0, P_1, ..., P_n`. This is the algebraic step that produces the specific numbers in the A matrix.

Split into two parts:

```text
(z + 1) * P_n'(z) = z * P_n'(z) + P_n'(z)
```

For `z * P_n'(z)`, use the identity:

```text
z * P_n'(z) = n * P_n(z) + P_{n-1}'(z)
```

For `P_n'(z)`, use the derivative expansion:

```text
P_n'(z) = (2n-1) * P_{n-1}(z) + (2n-5) * P_{n-3}(z) + ...
         = sum over k < n where (n - k) is odd of:  (2k+1) * P_k(z)
```

Combining and grouping by `P_k`:

- The term `n * P_n(z)` comes from the first identity.
- The terms `(2k+1) * P_k(z)` for `k < n` with appropriate parity come from both identities.

After carrying through the algebra and applying the normalization factors `sqrt(2n+1)` (to convert from I_n to x_n), the final result is:

```text
d/dt x_n(t) = -(1/t) * sum over k from 0 to N-1 of:  A_{nk} * x_k(t)
            + (1/t) * B_n * u(t)
```

where:

```text
A_{nk} = sqrt(2n+1) * sqrt(2k+1)   if k < n
A_{nn} = n + 1
A_{nk} = 0                         if k > n

B_n = sqrt(2n+1)
```

### The result

Writing the full system in matrix form:

```text
d/dt x(t) = -(1/t) * A * x(t)  +  (1/t) * B * u(t)
```

The matrix A is lower-triangular with entries determined by the Legendre normalization constants. B is a column vector of those same normalization constants.

Three things to notice:

**The triangular shape.** Every entry above the diagonal of A is zero. This means information flows from coarse to fine (from low-index coefficients to high-index), never the other direction. This is a direct consequence of the derivative identity for Legendre polynomials.

**The diagonal values.** The diagonal entries of A are `1, 2, 3, ..., N`. With the leading minus sign in the ODE, the effective growth rates are `-1, -2, -3, ...`, all negative. This is what guarantees stability (Section 10).

**No free parameters.** We never chose A or B. Once we decided to use (1) projection coefficients, (2) Legendre polynomials, and (3) the uniform scaled measure, the calculus forced A and B to be exactly these matrices. They are theorems, not hyperparameters.

---

## 8. A fully worked example with three coefficients

Let us instantiate everything concretely with N = 3, meaning we keep three coefficients: the average, the trend, and the curvature.

### The matrices

Using the formulas `A_{nk} = sqrt(2n+1) * sqrt(2k+1)` for k less than n, `A_{nn} = n+1`, and `B_n = sqrt(2n+1)`:

```text
A =  [ 1      0      0   ]     B =  [ 1    ]
     [ 1.73   2      0   ]          [ 1.73 ]
     [ 2.24   3.87   3   ]          [ 2.24 ]
```

Where `1.73 = sqrt(3)`, `2.24 = sqrt(5)`, `3.87 = sqrt(15)`.

Verify a few entries:

```text
A_{10} = sqrt(2*1+1) * sqrt(2*0+1) = sqrt(3) * sqrt(1) = sqrt(3) = 1.73   correct
A_{20} = sqrt(2*2+1) * sqrt(2*0+1) = sqrt(5) * sqrt(1) = sqrt(5) = 2.24   correct
A_{21} = sqrt(2*2+1) * sqrt(2*1+1) = sqrt(5) * sqrt(3) = sqrt(15) = 3.87  correct
A_{22} = 2 + 1 = 3                                                          correct
```

### What the three coefficients represent

Suppose at time t the input history is a ramp: `u(s) = s/t` for s in `[0, t]`, which rises linearly from 0 at the start to 1 now.

Substituting `xi = (2s/t) - 1`, the ramp in the `[-1, 1]` coordinate is `u = (xi + 1) / 2`.

**Coefficient 0 (average, using P_0 = 1):**

```text
x_0 = sqrt(1) * (1/2) * integral from -1 to 1 of:  (xi+1)/2 * 1  d_xi

    = (1/2) * (1/2) * [ integral of xi from -1 to 1 + integral of 1 from -1 to 1 ]

    = (1/4) * [ 0 + 2 ]   = 1/2 = 0.5
```

The average of a ramp from 0 to 1 is 0.5. Correct.

**Coefficient 1 (trend, using P_1 = xi):**

```text
x_1 = sqrt(3) * (1/2) * integral from -1 to 1 of:  (xi+1)/2 * xi  d_xi

    = sqrt(3)/4 * integral of (xi^2 + xi) d_xi from -1 to 1

    = sqrt(3)/4 * [ integral of xi^2 (= 2/3) + integral of xi (= 0) ]

    = sqrt(3)/4 * 2/3 = sqrt(3)/6 = 1.732/6 ≈ 0.289
```

Positive, confirming the signal is rising.

**Coefficient 2 (curvature, using P_2 = (1/2)(3xi^2 - 1)):**

```text
x_2 = sqrt(5) * (1/2) * integral from -1 to 1 of:  (xi+1)/2 * (1/2)(3xi^2 - 1)  d_xi

    = (sqrt(5)/4) * (1/2) * integral of (xi+1)(3xi^2-1) d_xi from -1 to 1
```

The integrand `(xi + 1)(3xi^2 - 1) = 3xi^3 + 3xi^2 - xi - 1`. Integrating term by term:

```text
integral of 3xi^3 from -1 to 1 = 0   (odd function)
integral of 3xi^2 from -1 to 1 = 3 * 2/3 = 2
integral of -xi   from -1 to 1 = 0   (odd function)
integral of -1    from -1 to 1 = -2
```

Total: `0 + 2 + 0 - 2 = 0`. So `x_2 = 0`.

No curvature in a straight ramp. Correct.

The state is:

```text
x = (0.5,  0.289,  0.0)
```

Read as: "average one-half, rising trend, no curvature." That is a perfect three-number description of a linear ramp.

### One update step

Suppose a new input `u = 1` arrives at time t, with step size `Delta = 0.1`. Using the simplified ODE `d/dt x = -A x + B u` (absorbing the 1/t into Delta):

Compute `A * x`:

```text
Row 0:  1 * 0.5 + 0 + 0 = 0.5
Row 1:  1.73 * 0.5 + 2 * 0.289 + 0 = 0.865 + 0.578 = 1.443
Row 2:  2.24 * 0.5 + 3.87 * 0.289 + 3 * 0 = 1.12 + 1.118 = 2.238
```

Compute `-A*x + B*u`:

```text
Row 0:  -0.5 + 1 * 1 = 0.5
Row 1:  -1.443 + 1.73 * 1 = 0.287
Row 2:  -2.238 + 2.24 * 1 = 0.002
```

Update:

```text
x_new = x_old + Delta * (d/dt x)

= (0.5,    0.289, 0.0) + 0.1 * (0.5,   0.287,  0.002)
= (0.55,   0.318, 0.0)
```

The average went up (the new input of 1 is above the current average of 0.5). The trend nudged up slightly. Curvature stayed near zero. The state updated sensibly using only matrix-vector arithmetic.

---

## 9. Discretization: from continuous equations to token steps

### The gap between theory and practice

The HiPPO derivation produced a continuous-time differential equation:

```text
d/dt x(t) = A * x(t) + B * u(t)
```

But a language model does not receive a continuous signal. It receives discrete tokens `u_1, u_2, u_3, ...` at fixed time steps. We need to convert the continuous-time equation into a discrete-time recurrence.

### The exact solution: zero-order hold

Assume the input `u` is held constant at value `u_k` over the time interval from step k-1 to step k. This assumption is called the Zero-Order Hold (ZOH). It is exactly correct for token sequences, where the input genuinely is constant between steps.

Under this assumption, the exact solution to the differential equation over one step of width Delta is:

```text
x_k = A_bar * x_{k-1} + B_bar * u_k
```

where:

```text
A_bar = exp(Delta * A)

B_bar = (Delta * A)^{-1} * (exp(Delta * A) - I) * Delta * B
```

Here `exp(M)` is the matrix exponential, defined by the power series:

```text
exp(M) = I + M + M^2/2! + M^3/3! + M^4/4! + ...
```

This is the same formula as the ordinary exponential `e^x = 1 + x + x^2/2! + ...`, but applied to a matrix instead of a number.

**Why this formula?** The continuous ODE `d/dt x = A x` has the solution `x(t) = exp(A * t) * x(0)`. Over one step of width Delta, the state gets multiplied by `exp(A * Delta)`. The input contribution is the integral of the state solution over the step, which gives the `B_bar` formula.

### The resulting recurrence

The discrete model is:

```text
x_k = A_bar * x_{k-1} + B_bar * u_k     (state update)
y_k = C * x_k                            (output)
```

This is a simple linear recurrence. At every step: multiply the old state by A_bar, add B_bar times the new input, get the new state. The output C reads from the state to produce the prediction.

### Training mode vs inference mode

The recurrence above is sequential: you cannot compute step k before step k-1. This is fine for inference (generating text one token at a time) but slow for training (where you want to process all tokens in parallel on a GPU).

Because the recurrence is linear, unrolling it reveals a convolution. Starting from `x_0 = 0`:

```text
x_k = A_bar^{k-1} * B_bar * u_1
    + A_bar^{k-2} * B_bar * u_2
    + ...
    + A_bar^0 * B_bar * u_k
```

Therefore:

```text
y_k = C * x_k = sum from j=1 to k of:  C * A_bar^{k-j} * B_bar * u_j
```

This is a convolution of the input sequence with the kernel:

```text
K = (C*B_bar, C*A_bar*B_bar, C*A_bar^2*B_bar, ...)
```

Convolutions can be computed in parallel using the Fast Fourier Transform (FFT), giving `O(L log L)` training time over a sequence of L tokens.

| Mode | Formula | Cost | When used |
| --- | --- | --- | --- |
| Recurrent | x_k = A_bar x_{k-1} + B_bar u_k | O(N) per step, O(N) memory | Inference, generation |
| Convolutional | y = K convolved with u via FFT | O(L log L) total | Training |

This dual mode is one of the key advantages of SSMs over attention. Attention has O(L^2) cost at training time and growing memory cost at inference time. SSMs have O(L log L) training cost and constant O(N) memory at inference regardless of context length.

### The role of Delta

The step size Delta controls how much the state changes per token. A large Delta means the state evolves a lot per token (captures short-range, fast-changing patterns). A small Delta means slow evolution (long-range memory).

In the basic SSM, Delta is a fixed learned scalar. Mamba's central innovation, described in the next section, is to make Delta depend on the input token, so the model can dynamically decide how fast to update its memory.

---

## 10. Stability: why the memory never explodes

### The question

We apply the transition matrix A_bar to the state at every step, thousands or millions of times. Can the state grow without bound?

For a scalar `a` applied repeatedly, the answer depends on its magnitude: `a^k` grows if `|a| > 1` and shrinks if `|a| < 1`. The same logic applies to matrices, but we need "eigenvalues" instead of magnitude.

### What eigenvalues are

An eigenvector `v` of a matrix `A` is a special direction: when you apply `A` to it, you get back the same direction scaled by a number `lambda`:

```text
A * v = lambda * v
```

The number `lambda` is the eigenvalue. Along this direction, the matrix acts exactly like multiplication by a scalar.

For a lower-triangular matrix, the eigenvalues are simply the diagonal entries (this follows from the determinant formula: `det(A - lambda * I) = product of (A_{nn} - lambda)` for triangular matrices).

### The stability condition

For a continuous-time system `d/dt x = M * x`, the state along eigenvector `v` evolves as `exp(lambda * t)`. This:

- decays to zero if `lambda < 0` (the real part is negative)
- stays constant if `lambda = 0`
- explodes if `lambda > 0`

For the discretized system `x_k = A_bar * x_{k-1}` with `A_bar = exp(Delta * A)`, the eigenvalues of `A_bar` are `exp(Delta * lambda)`. Since `|exp(Delta * lambda)| = exp(Delta * Re(lambda))`, the discrete system is stable exactly when the continuous eigenvalues have negative real part.

### HiPPO is stable by design

The HiPPO ODE is `d/dt x = -(1/t) A x + (1/t) B u`. The effective transition matrix is `-A`. Since `A` is lower-triangular with diagonal entries `1, 2, 3, ..., N`, the matrix `-A` has diagonal entries `-1, -2, -3, ..., -N`.

These are the eigenvalues of `-A`:

```text
eigenvalues of (-A) = {-1, -2, -3, ..., -N}
```

All eigenvalues are real and strictly negative. Therefore:

- every mode decays toward zero
- the state stays bounded for arbitrarily long input sequences
- the memory is stable

**Worked check for N = 3.** The effective matrix is:

```text
-A = [ -1      0       0  ]
     [ -1.73  -2       0  ]
     [ -2.24  -3.87   -3  ]
```

This is lower-triangular. The eigenvalues are the diagonal entries: `-1, -2, -3`. All negative. Stability confirmed directly from the structure.

**Why the lower-triangular shape matters.** The triangular structure serves two purposes simultaneously: it encodes the direction of information flow (coarse to fine, never backward), and it makes the eigenvalues trivially readable from the diagonal. Both properties fall directly out of the Legendre derivative identity from Section 5.

---

## 11. From HiPPO to S4 to Mamba

HiPPO provides the mathematical foundation. S4 and Mamba build on it in two directions.

### S4 (2021): making HiPPO fast

The HiPPO matrix A is an N-by-N dense lower-triangular matrix. Computing powers of A to build the convolution kernel `K = (C*B_bar, C*A_bar*B_bar, ...)` naively costs O(N^2) per kernel entry, which is expensive for large N.

S4 (Structured State Spaces) found a way to reparameterize A into a form where the kernel can be computed efficiently. The key insight is to write A as a diagonal matrix plus a low-rank correction:

```text
A = Lambda - P * Q^T
```

where `Lambda` is diagonal and `P`, `Q` are N-by-1 vectors. This "Diagonal Plus Low-Rank" (DPLR) structure enables computing the entire kernel via a Cauchy matrix inversion, which has known fast algorithms. The training cost drops to O(L log L) over sequences of length L, even for N-by-N state transitions.

S4 keeps HiPPO's memory theory unchanged. The state still stores Legendre projection coefficients. The speedup comes entirely from the algebraic restructuring of A.

A later simplification (S4D, 2022) showed that a purely diagonal approximation of the HiPPO matrix retains most of the performance gain. Most modern SSM implementations use diagonal A matrices.

### Mamba (2023): making the memory selective

The fundamental limitation of HiPPO and S4 is that the matrices A, B, C, and Delta are the same for every input token. The model processes "the" exactly the same way it processes "earthquake." This is called a Linear Time-Invariant (LTI) system.

This matters because some tokens carry important new information (a key name, a new fact) while others are filler (articles, prepositions). An ideal memory system would write important tokens strongly into the state and let filler tokens pass through with minimal effect.

Mamba introduces input-dependent parameters. At each step, the matrices B, C, and the step size Delta are computed from the current input:

```text
B_k = linear_B(u_k)         (a learned linear projection of the current input)
C_k = linear_C(u_k)
Delta_k = softplus(linear_Delta(u_k))
```

Now each token controls how strongly it writes into the state (through B_k and Delta_k) and how the state is read (through C_k). The model learns which tokens are worth remembering.

This breaks the convolutional interpretation (the kernel is no longer fixed). Mamba uses a hardware-aware parallel scan algorithm to compute the recurrence efficiently even though it is now input-dependent. The result: O(L) training time, O(N) inference memory, and the ability to selectively compress information.

HiPPO's matrix A remains the backbone. It still provides the coarse-to-fine structure and the stability guarantee. What Mamba adds is dynamic control over what gets written in and read out.

The through-line:

```text
HiPPO:  what should the state mean?     (optimal polynomial projection)
S4:     how do we compute it fast?      (DPLR reparameterization + FFT)
Mamba:  how do we make it selective?    (input-dependent B, C, Delta)
```

---

## 12. Symbol reference

| Symbol | Meaning |
| --- | --- |
| u(t) | input signal at time t (a single number) |
| x(t) | hidden state at time t (a list of N numbers) |
| A | N-by-N state transition matrix |
| B | N-by-1 input column |
| C | 1-by-N output row |
| P_n | n-th Legendre polynomial |
| g_n(s, t) | n-th basis function rescaled to the window [0, t] |
| weight(s, t) | the measure, assigning importance to each past moment |
| <f, g> | inner product of functions f and g |
| Delta | discretization step size (time per token) |
| A_bar, B_bar | discretized versions of A and B |
| exp(M) | matrix exponential of M |
| ZOH | Zero-Order Hold: input held constant between steps |

---

## 13. References

### Primary papers

1. Gu, A., Dao, T., Ermon, S., Rudra, A., and Re, C. (2020). HiPPO: Recurrent Memory with Optimal Polynomial Projections. NeurIPS 2020. arXiv:2008.07669.

2. Gu, A., Goel, K., and Re, C. (2021). Efficiently Modeling Long Sequences with Structured State Spaces (S4). ICLR 2022. arXiv:2111.00396.

3. Gu, A., Goel, K., Gupta, A., and Re, C. (2022). On the Parameterization and Initialization of Diagonal State Space Models (S4D). NeurIPS 2022. arXiv:2206.11893.

4. Gu, A., and Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.

5. Rush, A., and Karamcheti, S. The Annotated S4. srush.github.io/annotated-s4. A line-by-line JAX implementation with commentary.

6. Grootendorst, M. A Visual Guide to Mamba and State Space Models. maartengrootendorst.com/blog/mamba. Intuition-first walkthrough.

### Mathematical background

1. Abramowitz, M., and Stegun, I. A. (1964). Handbook of Mathematical Functions, Chapter 8 (Legendre Functions) and Chapter 22 (Orthogonal Polynomials). Classic reference for Legendre identities and the three-term recurrence.

2. Szego, G. (1939). Orthogonal Polynomials. American Mathematical Society. The definitive treatise on orthogonal polynomial families.

3. Boyce, W. E., and DiPrima, R. C. Elementary Differential Equations and Boundary Value Problems. For the matrix exponential, linear ODE solution theory, and Sturm-Liouville problems.

---

**A note on notation across sources.** Different papers differ by signs and scalings: some write `d/dt x = A x + B u`, others write `d/dt x = -A x + B u`. The structural facts (lower-triangular A, `sqrt(2n+1)` factors, eigenvalues `-(n+1)`, the projection interpretation) are the same in all conventions. Readers cross-referencing the original HiPPO paper should consult Appendix D, where the LegS matrices and their derivation are given in the paper's notation.
