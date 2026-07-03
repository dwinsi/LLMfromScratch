# HiPPO and State Space Models — A Rigorous Mathematical Treatment

*A first-principles derivation of the HiPPO framework, from orthogonal polynomials to the state space memory update.*

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Notation and Prerequisites](#2-notation-and-prerequisites)
3. [State Space Models: The Object We Are Building](#3-state-space-models-the-object-we-are-building)
4. [The Online Function Approximation Problem](#4-the-online-function-approximation-problem)
5. [Orthogonal Polynomials from First Principles](#5-orthogonal-polynomials-from-first-principles)
6. [Legendre Polynomials Derived from Scratch](#6-legendre-polynomials-derived-from-scratch)
7. [The HiPPO Framework: Projection onto a Moving Measure](#7-the-hippo-framework-projection-onto-a-moving-measure)
8. [Deriving the HiPPO-LegS Matrices A and B](#8-deriving-the-hippo-legs-matrices-a-and-b)
9. [A Fully Worked Example: N = 3](#9-a-fully-worked-example-n--3)
10. [Discretization: From Continuous ODE to Token Recurrence](#10-discretization-from-continuous-ode-to-token-recurrence)
11. [Stability via Eigenvalues](#11-stability-via-eigenvalues)
12. [From HiPPO to S4 to Mamba](#12-from-hippo-to-s4-to-mamba)
13. [References](#13-references)

---

## 1. Introduction and Motivation

A recurrent model processes a signal $u(t)$ one step at a time while maintaining a fixed-size hidden state $x(t) \in \mathbb{R}^N$. The central design question is:

> **Given that we can only store $N$ numbers, what is the mathematically optimal way to summarize everything seen so far, such that the entire history can be reconstructed as accurately as possible?**

HiPPO (**Hi**gh-order **P**olynomial **P**rojection **O**perators), introduced by Gu, Dao, Ermon, Rudra, and Ré (2020), answers this question exactly. The answer turns out to be: *maintain the coefficients of the optimal polynomial approximation of the history, measured against a chosen weighting.* Remarkably, the update rule for those coefficients is a **linear ordinary differential equation**

$$
\frac{d}{dt} x(t) = A\, x(t) + B\, u(t),
$$

and the matrices $A$ and $B$ are **not free parameters** — they are forced, uniquely, by the choice of polynomial basis and weighting measure. This document derives that result from first principles.

The payoff is concrete. When the matrix $A$ in a state space model is initialized as the HiPPO matrix rather than randomly, performance on long-range sequence benchmarks jumps dramatically — the original work reports sequential-MNIST accuracy rising from roughly 60% to over 98% with the same architecture and training. HiPPO is the mathematical foundation beneath S4 and, ultimately, Mamba.

---

## 2. Notation and Prerequisites

| Symbol | Meaning |
|---|---|
| $u(t)$ | the input signal (scalar-valued function of time) |
| $x(t) \in \mathbb{R}^N$ | the hidden state — the $N$ coefficients we store |
| $A \in \mathbb{R}^{N\times N}$ | state transition matrix |
| $B \in \mathbb{R}^{N\times 1}$ | input matrix |
| $P_n(\cdot)$ | the $n$-th Legendre polynomial |
| $g_n(\cdot)$ | the $n$-th normalized basis function |
| $\mu^{(t)}$ | the weighting measure at time $t$ |
| $\langle f, g\rangle_\mu$ | inner product of $f,g$ under measure $\mu$ |
| $\Delta$ | discretization step size |
| $\overline{A}, \overline{B}$ | discretized versions of $A, B$ |

**Assumed background:** vectors and matrices, the derivative, definite integrals, and the idea of a basis. Everything else — inner products of functions, orthogonality, and the Legendre polynomials themselves — is developed here from scratch.

A note on one identity used repeatedly. The **Leibniz integral rule** for differentiating an integral with a variable upper limit is

$$
\frac{d}{dt}\int_{a}^{t} f(s,t)\,ds
= f(t,t) + \int_{a}^{t} \frac{\partial}{\partial t} f(s,t)\,ds .
$$

The first term accounts for the moving boundary; the second for the dependence of the integrand on $t$. This is the technical engine of the entire HiPPO derivation.

---

## 3. State Space Models: The Object We Are Building

A continuous-time, single-input single-output **linear state space model** is the pair of equations

$$
\begin{aligned}
\dot{x}(t) &= A\,x(t) + B\,u(t), \\
y(t) &= C\,x(t) + D\,u(t),
\end{aligned}
$$

where $x(t)\in\mathbb{R}^N$ is the internal state, $u(t)\in\mathbb{R}$ the input, and $y(t)\in\mathbb{R}$ the output. The term $D\,u(t)$ is a direct feedthrough and is often dropped (set $D=0$) or folded into a residual connection.

This is the language of control theory, used to describe physical systems — a damped spring, an RC circuit, the attitude of a spacecraft. The state $x$ encodes "everything about the past that is relevant for predicting the future."

The question HiPPO poses is different from the classical one. Classically, engineers *know* the physics and write down $A, B, C$ from the dynamics. HiPPO instead *specifies what the state should mean* — it should be the coefficients of the best running approximation of $u$ — and then *derives* the $A, B$ that make the state evolve correctly. The matrix $A$ is no longer arbitrary; it is the consequence of a memory objective.

---

## 4. The Online Function Approximation Problem

Fix a time $t$. We have observed the input on $[0, t]$, i.e. the history $u|_{[0,t]}$. We want to compress this history into $N$ numbers.

**Step 1 — Choose a measure.** Let $\mu^{(t)}$ be a probability measure on $(-\infty, t]$ that specifies how much we care about each past time. For instance, $\mu^{(t)}$ might weight all of $[0,t]$ uniformly, or weight recent times more heavily.

**Step 2 — Choose a basis.** Let $\{g_0, g_1, \dots, g_{N-1}\}$ be $N$ functions that are orthonormal with respect to $\mu^{(t)}$:

$$
\langle g_n, g_m\rangle_{\mu^{(t)}} = \int g_n(s)\,g_m(s)\,d\mu^{(t)}(s) = \delta_{nm},
$$

where $\delta_{nm}$ is $1$ if $n=m$ and $0$ otherwise.

**Step 3 — Project.** The best approximation of $u$ in the span of the basis, in the least-squares sense weighted by $\mu^{(t)}$, is

$$
u(s) \;\approx\; \hat{u}(s) = \sum_{n=0}^{N-1} c_n(t)\, g_n(s),
\qquad
c_n(t) = \langle u, g_n\rangle_{\mu^{(t)}}.
$$

The coefficient vector $x(t) = [c_0(t), c_1(t), \dots, c_{N-1}(t)]^\top$ **is** the hidden state. This is the key conceptual move: the state is defined as the projection coefficients of the history.

Because the measure $\mu^{(t)}$ moves as $t$ advances (the window $[0,t]$ grows), the coefficients $c_n(t)$ change with time. The entire content of HiPPO is: **compute $\dot{c}_n(t)$, and show it has the form $A x + B u$.**

Why least-squares projection? Because for a fixed orthonormal basis, the coefficients $c_n = \langle u, g_n\rangle$ are exactly the choice that minimizes the weighted squared error $\int (u - \hat u)^2 \, d\mu$. This is the finite-dimensional projection theorem in a Hilbert space: the best approximation in a subspace is the orthogonal projection, and its coordinates are the inner products with an orthonormal basis.

---

## 5. Orthogonal Polynomials from First Principles

Before Legendre polynomials specifically, we establish what "orthogonal polynomials" means and why they exist.

### 5.1 Functions as vectors

Consider the set of continuous functions on an interval $[a,b]$. We can add them and scale them, so they form a vector space. We equip this space with an inner product using a **weight function** $w(s) \geq 0$:

$$
\langle f, g\rangle = \int_a^b f(s)\,g(s)\,w(s)\,ds.
$$

This behaves exactly like the familiar dot product $\mathbf{a}\cdot\mathbf{b} = \sum_i a_i b_i$, with the sum replaced by an integral. Two functions are **orthogonal** if $\langle f, g\rangle = 0$, and the **norm** is $\|f\| = \sqrt{\langle f, f\rangle}$.

### 5.2 Building an orthogonal polynomial family

Start with the monomials $1, s, s^2, s^3, \dots$. These are linearly independent but *not* orthogonal — for example, on $[-1,1]$ with weight $w=1$, $\langle 1, s^2\rangle = \int_{-1}^1 s^2\,ds = \tfrac{2}{3}\neq 0$.

We orthogonalize them with the **Gram–Schmidt process**, which is the same procedure used for vectors. Define $p_0(s) = 1$, and for each $n\geq 1$ subtract off the components along all previous polynomials:

$$
p_n(s) = s^n - \sum_{k=0}^{n-1} \frac{\langle s^n, p_k\rangle}{\langle p_k, p_k\rangle}\, p_k(s).
$$

The result $\{p_0, p_1, p_2, \dots\}$ is a family of polynomials, with $\deg p_n = n$, that are mutually orthogonal under $\langle\cdot,\cdot\rangle$. Different weight functions $w(s)$ produce different families:

| Weight $w(s)$ | Interval | Family |
|---|---|---|
| $1$ | $[-1,1]$ | **Legendre** |
| $(1-s^2)^{-1/2}$ | $[-1,1]$ | Chebyshev (1st kind) |
| $e^{-s}$ | $[0,\infty)$ | Laguerre |
| $e^{-s^2}$ | $(-\infty,\infty)$ | Hermite |

HiPPO-LegS uses the **Legendre** family because its uniform weight corresponds to caring about all of history equally, which yields the desirable *timescale-invariance* property discussed later.

### 5.3 The three-term recurrence — why it must exist

Every family of orthogonal polynomials satisfies a **three-term recurrence relation** of the form

$$
p_{n+1}(s) = (\alpha_n s + \beta_n)\,p_n(s) - \gamma_n\, p_{n-1}(s).
$$

The reason is structural and worth stating, because this recurrence is ultimately the source of the banded structure of the HiPPO matrix. Consider $s\,p_n(s)$: it is a polynomial of degree $n+1$, so it can be written in the basis $\{p_0,\dots,p_{n+1}\}$ as $s\,p_n = \sum_{k=0}^{n+1} a_k p_k$. Taking the inner product with $p_m$ and using orthogonality,

$$
a_m \|p_m\|^2 = \langle s\,p_n, p_m\rangle = \langle p_n, s\,p_m\rangle.
$$

But $s\,p_m$ has degree $m+1$, so if $m+1 < n$, i.e. $m < n-1$, then $\langle p_n, s\,p_m\rangle = 0$ by orthogonality (a degree-$n$ polynomial is orthogonal to everything of lower degree). Hence only $a_{n-1}, a_n, a_{n+1}$ survive — a three-term recurrence. This "only neighbors interact" property is exactly why the HiPPO matrix will turn out to be lower-triangular (in fact banded before normalization).

---

## 6. Legendre Polynomials Derived from Scratch

We now construct the Legendre polynomials explicitly, three independent ways, so the object is completely demystified.

### 6.1 Construction 1 — Gram–Schmidt on $[-1,1]$ with $w=1$

Using the inner product $\langle f,g\rangle = \int_{-1}^{1} f(s) g(s)\, ds$:

**$P_0$:** Take $p_0 = 1$.

**$P_1$:** Orthogonalize $s$ against $p_0$:
$$
p_1 = s - \frac{\langle s, 1\rangle}{\langle 1,1\rangle}\cdot 1 = s - \frac{\int_{-1}^1 s\,ds}{\int_{-1}^1 1\,ds} = s - \frac{0}{2} = s.
$$

**$P_2$:** Orthogonalize $s^2$ against $p_0$ and $p_1$:
$$
\langle s^2, 1\rangle = \int_{-1}^1 s^2\,ds = \tfrac{2}{3},\qquad
\langle s^2, s\rangle = \int_{-1}^1 s^3\,ds = 0,
$$
$$
p_2 = s^2 - \frac{2/3}{2}\cdot 1 - 0 = s^2 - \tfrac{1}{3}.
$$

**$P_3$:** Similarly,
$$
\langle s^3, 1\rangle = 0,\quad \langle s^3, s\rangle = \int_{-1}^1 s^4\,ds = \tfrac{2}{5},\quad \langle s^3, s^2-\tfrac13\rangle = 0,
$$
$$
p_3 = s^3 - \frac{2/5}{2/3}\, s = s^3 - \tfrac{3}{5}s.
$$

By convention, Legendre polynomials are **normalized so that $P_n(1) = 1$** (rather than to unit norm). Rescaling the $p_n$ above to satisfy this gives the standard forms:

$$
\boxed{P_0 = 1,\quad P_1 = s,\quad P_2 = \tfrac{1}{2}(3s^2 - 1),\quad P_3 = \tfrac{1}{2}(5s^3 - 3s).}
$$

For example, $p_2 = s^2 - \tfrac13$; at $s=1$ this equals $\tfrac23$, so we multiply by $\tfrac32$ to get $P_2 = \tfrac32 s^2 - \tfrac12 = \tfrac12(3s^2-1)$, which indeed equals $1$ at $s=1$.

### 6.2 Construction 2 — Rodrigues' formula

A closed form that generates all of them:

$$
P_n(s) = \frac{1}{2^n\, n!}\frac{d^n}{ds^n}\left[(s^2-1)^n\right].
$$

*Check for $n=2$:* $(s^2-1)^2 = s^4 - 2s^2 + 1$; its second derivative is $12 s^2 - 4$; dividing by $2^2\cdot 2! = 8$ gives $\tfrac{12 s^2 - 4}{8} = \tfrac{1}{2}(3s^2-1) = P_2$. ✓

### 6.3 Construction 3 — the defining differential equation

The Legendre polynomials are the bounded solutions on $[-1,1]$ of **Legendre's differential equation**:

$$
\frac{d}{ds}\!\left[(1-s^2)\frac{dP_n}{ds}\right] + n(n+1)P_n = 0.
$$

This form is important because it makes the polynomials *eigenfunctions* of a self-adjoint (Sturm–Liouville) operator, with eigenvalue $-n(n+1)$. Self-adjointness immediately guarantees orthogonality of eigenfunctions with distinct eigenvalues — an alternative proof that the $P_n$ are orthogonal.

### 6.4 Key properties we will use

**Orthogonality and norm.**
$$
\int_{-1}^{1} P_n(s) P_m(s)\,ds = \frac{2}{2n+1}\,\delta_{nm}.
$$
Thus the **normalized** Legendre polynomials are $\sqrt{\tfrac{2n+1}{2}}\,P_n(s)$, and the factor $\sqrt{2n+1}$ that pervades the HiPPO matrix comes directly from this.

**Three-term recurrence.**
$$
(n+1)P_{n+1}(s) = (2n+1)\,s\,P_n(s) - n\,P_{n-1}(s).
$$

**Derivative identity.** The one we need for the derivation:
$$
(2n+1)P_n(s) = \frac{d}{ds}\big[P_{n+1}(s) - P_{n-1}(s)\big],
$$
equivalently written using $P_n'$ relations. A second useful identity is
$$
(1-s^2)P_n'(s) = n\big[P_{n-1}(s) - s\,P_n(s)\big].
$$

**Boundary values.** $P_n(1) = 1$ and $P_n(-1) = (-1)^n$.

These four facts are everything we need to derive $A$ and $B$.

---

## 7. The HiPPO Framework: Projection onto a Moving Measure

We now specialize the general projection setup of Section 4 to the **HiPPO-LegS** ("Legendre, Scaled") variant, which is the one used inside S4 and Mamba.

### 7.1 The scaled Legendre measure

At time $t$, define the measure on $[0,t]$ that is **uniform**:

$$
d\mu^{(t)}(s) = \frac{1}{t}\,\mathbb{1}_{[0,t]}(s)\,ds .
$$

The factor $1/t$ makes it a probability measure (it integrates to $1$). "Scaled" refers to the fact that as $t$ grows, the same measure stretches to cover the whole history $[0,t]$ — it does not have a fixed window. This gives HiPPO-LegS its signature **timescale invariance**: the memory representation does not privilege any particular absolute time scale.

### 7.2 The basis, rescaled to the window

The Legendre polynomials live on $[-1,1]$. We must map them onto the window $[0,t]$. The affine change of variable

$$
s \;\longmapsto\; \frac{2s}{t} - 1
$$

sends $s\in[0,t]$ to $[-1,1]$. Define the (measure-normalized) basis functions

$$
g_n^{(t)}(s) = \sqrt{2n+1}\; P_n\!\left(\frac{2s}{t}-1\right), \qquad s\in[0,t].
$$

The $\sqrt{2n+1}$ makes them orthonormal under $\mu^{(t)}$: one can verify
$$
\int_0^t g_n^{(t)}(s)\,g_m^{(t)}(s)\,\frac{1}{t}\,ds
= \sqrt{(2n+1)(2m+1)}\cdot\frac{1}{2}\int_{-1}^1 P_n(\xi)P_m(\xi)\,d\xi
= \delta_{nm},
$$
using the substitution $\xi = \tfrac{2s}{t}-1$ (so $ds = \tfrac{t}{2}\,d\xi$) and the Legendre norm $\tfrac{2}{2n+1}$.

### 7.3 The coefficients (the state)

The hidden state components are the projections

$$
\boxed{\;
x_n(t) = c_n(t) = \langle u, g_n^{(t)}\rangle_{\mu^{(t)}}
= \frac{1}{t}\int_0^t u(s)\,\sqrt{2n+1}\;P_n\!\left(\frac{2s}{t}-1\right)ds .
\;}
$$

Everything now reduces to a calculus problem: **differentiate $x_n(t)$ with respect to $t$.**

---

## 8. Deriving the HiPPO-LegS Matrices A and B

This is the heart of the document. We compute $\dot{x}_n(t)$ and show it equals a linear combination of the $x_k(t)$ plus a multiple of $u(t)$.

### 8.1 Setting up the differentiation

Write, pulling the constant $\sqrt{2n+1}$ aside for now and using the shorthand $z(s,t) = \tfrac{2s}{t}-1$,

$$
x_n(t) = \sqrt{2n+1}\;\underbrace{\frac{1}{t}\int_0^t u(s)\,P_n\!\big(z(s,t)\big)\,ds}_{\displaystyle I_n(t)} .
$$

We differentiate $I_n(t)$. Note $t$ appears in **three** places: the prefactor $1/t$, the upper limit $t$, and inside the argument $z(s,t)$ of $P_n$. We use the product rule together with the Leibniz integral rule.

**The prefactor and the moving boundary.** By the product rule on $\tfrac1t \cdot \int_0^t(\dots)ds$ and Leibniz's rule,

$$
\dot I_n(t)
= -\frac{1}{t^2}\int_0^t u(s)P_n(z)\,ds
\;+\; \frac{1}{t}\Big[\underbrace{u(t)P_n(z(t,t))}_{\text{boundary term}} + \int_0^t u(s)\,\frac{\partial}{\partial t}P_n(z)\,ds\Big].
$$

At the boundary $s=t$ we have $z(t,t) = \tfrac{2t}{t}-1 = 1$, and $P_n(1)=1$. So the boundary term is simply $u(t)$.

**The inner derivative.** Since $z = \tfrac{2s}{t}-1$,
$$
\frac{\partial z}{\partial t} = -\frac{2s}{t^2},
\qquad
\frac{\partial}{\partial t}P_n(z) = P_n'(z)\cdot\left(-\frac{2s}{t^2}\right).
$$

Now observe that $\tfrac{2s}{t} = z + 1$, so $-\tfrac{2s}{t^2} = -\tfrac{1}{t}(z+1)$. Therefore

$$
\frac{\partial}{\partial t}P_n(z) = -\frac{1}{t}(z+1)\,P_n'(z).
$$

Collecting terms,

$$
\dot I_n(t) = -\frac{1}{t}\,I_n(t) + \frac{1}{t}u(t) - \frac{1}{t^2}\int_0^t u(s)\,(z+1)P_n'(z)\,ds .
$$

### 8.2 Applying the Legendre identities

We must express $(z+1)P_n'(z)$ back in the Legendre basis so the integral becomes a combination of the coefficients $I_k$. Split it:

$$
(z+1)P_n'(z) = \underbrace{z\,P_n'(z)}_{(\ast)} + \underbrace{P_n'(z)}_{(\ast\ast)} .
$$

We use two standard identities for Legendre polynomials:

$$
z\,P_n'(z) = n\,P_n(z) + P_{n-1}'(z)\quad\text{(from the recurrence for derivatives)},
$$

and the expansion of the derivative in lower-order polynomials,

$$
P_n'(z) = (2n-1)P_{n-1}(z) + (2n-5)P_{n-3}(z) + \cdots
= \sum_{\substack{k<n \\ n-k\ \text{odd}}} (2k+1)\,P_k(z).
$$

Substituting and grouping by $P_k$, all terms combine so that $(z+1)P_n'(z)$ becomes a linear combination of $P_0, P_1, \dots, P_n$ with **integer / square-root-rational coefficients**. Carrying the algebra through (this is the mechanical part; the identities above are the only tools needed), and folding back the normalization $\sqrt{2n+1}$ that converts $I_k \to x_k$, one obtains the clean final result.

### 8.3 The result — the HiPPO-LegS matrices

The coefficient vector obeys the linear ODE

$$
\boxed{\;\dot{x}(t) = -\frac{1}{t}\,A\,x(t) + \frac{1}{t}\,B\,u(t)\;}
$$

with the **time-invariant** matrices

$$
A_{nk} =
\begin{cases}
\sqrt{2n+1}\,\sqrt{2k+1}, & k < n,\\[4pt]
n+1, & k = n,\\[4pt]
0, & k > n,
\end{cases}
\qquad\qquad
B_n = \sqrt{2n+1}.
$$

Several remarks make this result readable:

- The explicit $1/t$ factors reflect the *scaled* measure; in the S4/Mamba use, they are absorbed by the discretization step $\Delta$ (they set an adaptive time scale). Writing $\dot x = -\tfrac1t(Ax - Bu)$, the matrices $A, B$ themselves are constant — the entire $t$-dependence is the single scalar $1/t$.
- $A$ is **lower-triangular**: $A_{nk}=0$ for $k>n$. This is the direct fingerprint of the "only lower-degree polynomials appear in $P_n'$" identity — information flows from coarse (low $n$) to fine (high $n$), never backward.
- The diagonal entries $A_{nn} = n+1$ are strictly positive and increasing; with the leading minus sign in the ODE, the effective eigenvalues are $-(n+1)$, which are all negative — the origin of stability (Section 11).
- The square-root factors $\sqrt{2n+1}$ are exactly the Legendre normalization constants from Section 6.4.

> **The philosophical punchline.** We never *chose* $A$ or $B$. We chose (i) to store projection coefficients, (ii) the Legendre basis, and (iii) the uniform scaled measure. Calculus did the rest. $A$ and $B$ are theorems, not hyperparameters.

---

## 9. A Fully Worked Example: N = 3

Let us instantiate everything for a 3-dimensional state, $n \in \{0,1,2\}$.

### 9.1 The matrices

Using $A_{nk} = \sqrt{2n+1}\sqrt{2k+1}$ for $k<n$, $A_{nn} = n+1$, and $B_n = \sqrt{2n+1}$:

$$
A = \begin{pmatrix}
1 & 0 & 0\\
\sqrt{3}\sqrt{1} & 2 & 0\\
\sqrt{5}\sqrt{1} & \sqrt{5}\sqrt{3} & 3
\end{pmatrix}
= \begin{pmatrix}
1 & 0 & 0\\
\sqrt{3} & 2 & 0\\
\sqrt{5} & \sqrt{15} & 3
\end{pmatrix},
\qquad
B = \begin{pmatrix}1\\ \sqrt{3}\\ \sqrt{5}\end{pmatrix}.
$$

Numerically, $\sqrt3 \approx 1.732$, $\sqrt5 \approx 2.236$, $\sqrt{15}\approx 3.873$.

### 9.2 What the three coefficients mean

Suppose at some time $t$ the input history over $[0,t]$ is the ramp $u(s) = s/t$ (rising linearly from $0$ to $1$). Rescaling to $\xi = \tfrac{2s}{t}-1\in[-1,1]$, we have $u = \tfrac{\xi+1}{2}$. The coefficients are $x_n = \sqrt{2n+1}\cdot\tfrac12\int_{-1}^{1} \tfrac{\xi+1}{2}P_n(\xi)\,d\xi$:

- $x_0 = \sqrt1\cdot\tfrac12\int_{-1}^1 \tfrac{\xi+1}{2}\,d\xi = \tfrac12\cdot\tfrac12\cdot 2 = \tfrac12$. The **average** of the ramp is $\tfrac12$. ✓
- $x_1 = \sqrt3\cdot\tfrac12\int_{-1}^1 \tfrac{\xi+1}{2}\,\xi\,d\xi = \sqrt3\cdot\tfrac12\cdot\tfrac13 = \tfrac{\sqrt3}{6}\approx 0.289$. A **positive linear trend** — the signal is rising. ✓
- $x_2 = \sqrt5\cdot\tfrac12\int_{-1}^1 \tfrac{\xi+1}{2}\cdot\tfrac12(3\xi^2-1)\,d\xi = 0$. **No curvature** — the ramp is perfectly straight, so the quadratic coefficient vanishes. ✓

So the state $x = (\tfrac12,\ 0.289,\ 0)$ encodes "average one-half, rising, no bend" — a faithful three-number summary of a linear ramp. Reconstructing $\hat u(\xi) = \sum x_n \sqrt{2n+1}\,P_n(\xi)/\!\ldots$ recovers the ramp exactly, because a degree-1 signal lies entirely in the span of $P_0, P_1$.

### 9.3 One recurrence step (discretized preview)

Take $\Delta = 0.1$ and the crude Euler discretization $x_{t} \approx x_{t-1} + \Delta\,\dot x$ (the exact ZOH version is in Section 10). With the $1/t$ folded into $\Delta$ and a new input sample $u_t = 1$, starting from $x=(0.5, 0.289, 0)$:

$$
\dot x = -A x + B u
= -\begin{pmatrix}1&0&0\\ \sqrt3&2&0\\ \sqrt5&\sqrt{15}&3\end{pmatrix}\!\begin{pmatrix}0.5\\0.289\\0\end{pmatrix}
+ \begin{pmatrix}1\\ \sqrt3\\ \sqrt5\end{pmatrix}(1).
$$

Compute $Ax = (0.5,\ 0.866+0.578,\ 1.118+1.119+0)^\top = (0.5,\ 1.444,\ 2.237)^\top$. Then $-Ax + Bu = (-0.5+1,\ -1.444+1.732,\ -2.237+2.236)^\top = (0.5,\ 0.288,\ -0.001)^\top$. So

$$
x_{\text{new}} \approx (0.5, 0.289, 0) + 0.1\,(0.5, 0.288, -0.001) = (0.55,\ 0.318,\ 0.0).
$$

The average coefficient rises (the new sample $u=1$ pushes the mean up), the trend coefficient nudges up, curvature stays ~zero. The state updated sensibly using only matrix–vector arithmetic — no re-reading of history.

---

## 10. Discretization: From Continuous ODE to Token Recurrence

The derivation produced a *continuous-time* ODE, but a language model receives *discrete* tokens $u_1, u_2, u_3, \dots$. We convert the ODE into a recurrence.

### 10.1 The exact solution over one step

For a linear ODE $\dot x = Ax + Bu$ with $u$ held constant at value $u_k$ over a step of length $\Delta$ (this constancy assumption is the **Zero-Order Hold**, ZOH — it is exact for piecewise-constant inputs, which token streams are), the exact solution is

$$
x_{k} = \overline{A}\, x_{k-1} + \overline{B}\, u_k,
$$

with

$$
\boxed{\;\overline{A} = e^{\Delta A},
\qquad
\overline{B} = \big(\Delta A\big)^{-1}\!\big(e^{\Delta A} - I\big)\,\Delta B.\;}
$$

Here $e^{\Delta A}$ is the **matrix exponential**, defined by the same power series as the scalar exponential:
$$
e^{M} = I + M + \frac{M^2}{2!} + \frac{M^3}{3!} + \cdots .
$$

*Derivation sketch of $\overline A$:* the homogeneous equation $\dot x = Ax$ has solution $x(t) = e^{At}x(0)$; over one step of width $\Delta$ this multiplies the state by $e^{A\Delta}$. The $\overline B$ formula comes from the variation-of-constants (Duhamel) integral $\int_0^\Delta e^{A(\Delta-\tau)}B\,d\tau$ evaluated with $u$ constant.

### 10.2 The output equation

The output reads from the state through a matrix $C$ (learned, or chosen):
$$
y_k = C\, x_k .
$$
In S4/Mamba, $C$ is a learnable parameter — it decides which polynomial coefficients of the history matter for the current prediction.

### 10.3 The two computational modes

Because the recurrence is **linear**, unrolling it reveals a convolution. Starting from $x_0 = 0$:

$$
x_k = \sum_{j=1}^{k} \overline{A}^{\,k-j}\,\overline{B}\,u_j,
\qquad
y_k = \sum_{j=1}^{k}\underbrace{C\,\overline{A}^{\,k-j}\,\overline{B}}_{=\,\overline{K}_{k-j}}\,u_j = (\overline{K} * u)_k .
$$

This defines the **SSM convolution kernel** $\overline{K} = (C\overline B,\ C\overline A\,\overline B,\ C\overline A^2\overline B,\dots)$. Two equivalent ways to run the same model:

| Mode | Formula | Complexity | Used for |
|---|---|---|---|
| **Recurrent** | $x_k = \overline A x_{k-1} + \overline B u_k$ | $O(N)$ per step, $O(N)$ memory | inference / generation |
| **Convolutional** | $y = \overline K * u$ (via FFT) | $O(L\log L)$ over the sequence | training (parallel) |

This duality — parallel training, recurrent $O(1)$-memory inference — is precisely what makes SSMs attractive versus attention, whose inference cost grows with context length.

### 10.4 The role of $\Delta$

The step size $\Delta$ sets how much continuous time elapses per token. Large $\Delta$ means the state evolves a lot per token (attends to fast/local structure); small $\Delta$ means slow evolution (long memory). In the vanilla SSM, $\Delta$ is a fixed learned scalar. **Mamba's central innovation** (Section 12) is to make $\Delta$ — along with $B$ and $C$ — a *function of the current input*, so the model can dynamically decide how much to write and how fast to forget on a per-token basis.

---

## 11. Stability via Eigenvalues

Why does the HiPPO state not blow up over a long sequence? The answer is entirely in the eigenvalues of $A$.

### 11.1 Eigenvalues control repeated application

An eigenvector $v$ of $A$ satisfies $Av = \lambda v$. Along that direction, applying $A$ repeatedly is just scalar multiplication:
$$
A^k v = \lambda^k v .
$$
In continuous time, the solution operator $e^{At}$ acts along the eigenvector as
$$
e^{At} v = e^{\lambda t} v .
$$

### 11.2 The stability criterion

- **Continuous system** $\dot x = Ax$: the mode along $v$ evolves as $e^{\lambda t}$. It decays to $0$ iff $\operatorname{Re}(\lambda) < 0$, is constant iff $\operatorname{Re}(\lambda)=0$, and **explodes iff $\operatorname{Re}(\lambda) > 0$.** Stability requires **all eigenvalues to have negative real part.**
- **Discrete system** $x_k = \overline A x_{k-1}$: the mode evolves as $\overline\lambda^{\,k}$ where $\overline\lambda = e^{\Delta\lambda}$. It decays iff $|\overline\lambda| < 1$, which (since $|e^{\Delta\lambda}| = e^{\Delta\operatorname{Re}(\lambda)}$) is again equivalent to $\operatorname{Re}(\lambda) < 0$. The discretization preserves stability.

### 11.3 The HiPPO matrix is stable by construction

For the HiPPO-LegS $A$ (Section 8.3), which is lower-triangular, the eigenvalues are exactly the diagonal entries. In the ODE $\dot x = -Ax + Bu$ (note the leading minus sign), the effective transition matrix is $-A$, whose eigenvalues are

$$
-(n+1) \in \{-1, -2, -3, \dots, -N\}.
$$

Every eigenvalue is real and strictly negative. Therefore **every mode decays**, the state remains bounded for arbitrarily long inputs, and the memory is stable. The lower-triangular structure did double duty: it made the eigenvalues trivially readable *and* it encoded the coarse-to-fine information flow.

*Worked eigenvalue check for the $N=3$ example.* The matrix
$$
-A = \begin{pmatrix} -1 & 0 & 0\\ -\sqrt3 & -2 & 0\\ -\sqrt5 & -\sqrt{15} & -3\end{pmatrix}
$$
is lower-triangular, so $\det(-A - \lambda I) = (-1-\lambda)(-2-\lambda)(-3-\lambda)$, giving eigenvalues $\lambda = -1, -2, -3$ — all negative, confirming stability directly from the structure.

---

## 12. From HiPPO to S4 to Mamba

HiPPO provides the *initialization and theory*; S4 makes it *fast*; Mamba makes it *selective*.

**S4 (Structured State Spaces, Gu et al. 2021).** The bottleneck in using HiPPO's $A$ is computing the convolution kernel $\overline K$, which naively needs powers $\overline A^k$ — expensive for a dense $N\times N$ matrix. S4 reparameterizes $A$ in **Diagonal Plus Low-Rank (DPLR)** form, $A = \Lambda - PP^{*}$, for which the kernel can be computed in $O(N\log N)$ using a Cauchy-kernel / generating-function trick. S4 keeps HiPPO's memory theory while achieving transformer-competitive training speed on sequences of length $16{,}000+$.

**S4D (Diagonal, Gu et al. 2022).** A further simplification shows that a purely **diagonal** approximation of the HiPPO matrix retains most of the performance, dramatically simplifying the implementation. This is the form most Mamba implementations descend from.

**Mamba / S6 (Selective SSM, Gu & Dao 2023).** The vanilla SSM is **Linear Time-Invariant**: $A, B, C, \Delta$ are the same for every token, so the model cannot decide that "the" matters less than a rare content word. Mamba makes $B$, $C$, and $\Delta$ **functions of the input** $u_k$:
$$
B_k = \text{Linear}_B(u_k),\quad C_k = \text{Linear}_C(u_k),\quad \Delta_k = \text{softplus}(\text{Linear}_\Delta(u_k)).
$$
This "selectivity" breaks the convolutional mode (the kernel is no longer fixed), so Mamba introduces a hardware-aware **parallel scan** to keep training efficient. The result: linear-time sequence modeling that matches or beats transformers at long context, with $O(1)$ memory per generated token. HiPPO's $A$ remains the backbone that keeps the recurrence's memory well-behaved.

The through-line: **HiPPO answers "what should the state remember?"; S4 answers "how do we compute it fast?"; Mamba answers "how do we make it decide what to remember?"**

---

## 13. References

**Primary papers**

1. Gu, A., Dao, T., Ermon, S., Rudra, A., & Ré, C. (2020). *HiPPO: Recurrent Memory with Optimal Polynomial Projections.* Advances in Neural Information Processing Systems (NeurIPS) 33. arXiv:2008.07669.
2. Gu, A., Goel, K., & Ré, C. (2021). *Efficiently Modeling Long Sequences with Structured State Spaces (S4).* International Conference on Learning Representations (ICLR) 2022. arXiv:2111.00396.
3. Gu, A., Goel, K., Gupta, A., & Ré, C. (2022). *On the Parameterization and Initialization of Diagonal State Space Models (S4D).* NeurIPS 2022. arXiv:2206.11893.
4. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.

**Expository resources**

5. Rush, A., & Karamcheti, S. *The Annotated S4.* srush.github.io/annotated-s4 — a line-by-line JAX implementation with commentary.
6. Grootendorst, M. *A Visual Guide to Mamba and State Space Models.* maartengrootendorst.com/blog/mamba — intuition-first walkthrough of A/B/C, discretization, and selectivity.

**Mathematical background**

7. Abramowitz, M., & Stegun, I. A. (1964). *Handbook of Mathematical Functions*, Chapter 8 (Legendre Functions) and Chapter 22 (Orthogonal Polynomials). The classic reference for Legendre identities, Rodrigues' formula, and the three-term recurrence.
8. Szegő, G. (1939). *Orthogonal Polynomials.* American Mathematical Society Colloquium Publications, Vol. 23. The definitive treatise on orthogonal polynomial families and their properties.
9. Boyce, W. E., & DiPrima, R. C. *Elementary Differential Equations and Boundary Value Problems* — for the matrix exponential, linear ODE solution theory, and Sturm–Liouville problems.

**A note on conventions.** Different sources differ by signs and scalings depending on whether they write $\dot x = Ax + Bu$ or $\dot x = -Ax + Bu$, and whether $A$ is presented in normalized or unnormalized Legendre form. The structural facts — lower-triangularity, $\sqrt{2n+1}$ factors, eigenvalues $-(n+1)$, and the projection interpretation — are convention-independent and are what this document emphasizes. Readers cross-referencing the HiPPO paper should consult its Appendix D, where the LegS matrices and their derivation are given in full.

---

*End of document.*
