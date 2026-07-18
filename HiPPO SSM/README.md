# Legendre Polynomials and HiPPO: Code Implementation

This folder contains runnable Python that builds Legendre polynomials and the HiPPO memory operator from scratch. Nothing here relies on a black-box library for the mathematics. Every polynomial is constructed step by step from its definition, and every matrix entry is computed by the formula rather than looked up.

The code is the executable companion to the three markdown documents in this folder:

- `Legendre_polynomials_complete.md` derives the polynomials mathematically
- `HiPPO_SSM_explained.md` derives the HiPPO matrices A and B
- `HiPPO_for_beginners.md` explains the intuition without formulas

Running the code lets you see the theory working numerically.

---

## Files

| File | What it does |
| --- | --- |
| `legendre.py` | Builds Legendre polynomials four independent ways and verifies all their properties |
| `hippo.py` | Builds the HiPPO memory matrices, encodes a signal into fixed-size memory, and reconstructs it |
| `visualize.py` | Generates three figures showing the polynomials, reconstruction quality, and memory evolution |
| `requirements.txt` | Dependencies: numpy, scipy, matplotlib |

---

## Quick start

```bash
pip install -r requirements.txt

python legendre.py      # build and verify Legendre polynomials from scratch
python hippo.py         # build and verify the HiPPO memory operator
python visualize.py     # generate figures into the figures/ folder
```

Each script prints its verification results as it runs, so you can follow along with the mathematics.

---

## What `legendre.py` does

This script answers the question: are there really four independent ways to build the same polynomials?

It constructs P0 through P5 using each method, then checks that all four methods produce numerically identical results.

### The four methods

**Method 1: Gram-Schmidt.** Start with the simple powers `1, x, x^2, x^3, ...` and remove each one's overlap with the previous polynomials. The inner product of two polynomials is computed exactly (by integrating the product term by term), so orthogonality checks come out to machine zero rather than just "small."

**Method 2: Rodrigues' formula.** Apply the formula:

```text
P_n = (1 / (2^n * n!)) * (d^n/dx^n) [(x^2 - 1)^n]
```

Polynomials are stored as coefficient arrays, so differentiation is done directly on the coefficients rather than symbolically. You can trace every step.

**Method 3: Three-term recurrence.** Use the relation:

```text
(n+1) * P_{n+1} = (2n+1) * x * P_n - n * P_{n-1}
```

Starting from `P0 = 1` and `P1 = x`, this generates every subsequent polynomial in two multiplications and one subtraction.

**Method 4: Generating function.** Expand `1 / sqrt(1 - 2xt + t^2)` as a power series and extract the coefficient of `t^n`. This is a numerical cross-check.

### The property verifications

After building the polynomials, the script verifies every key property from the mathematical document:

```text
Orthogonality:  integral of P_n * P_m from -1 to 1 = 0 for n not equal to m
Norm:           integral of P_n^2 from -1 to 1 = 2 / (2n+1)
Endpoints:      P_n(1) = 1 and P_n(-1) = (-1)^n for all n
Parity:         P_n(-x) = (-1)^n * P_n(x)
Recurrence:     (2n+1)*P_n = P_{n+1}' - P_{n-1}' holds for all tested n
```

If you have read the derivations in `Legendre_polynomials_complete.md`, running this script confirms that the formulas are not just theoretical: they hold numerically, to the precision of floating-point arithmetic.

---

## What `hippo.py` does

This script builds the HiPPO-LegS matrices from their formulas and then demonstrates the memory compression in action.

### Building the matrices

The HiPPO A matrix has entries:

```text
A_{nk} = sqrt(2n+1) * sqrt(2k+1)   when k < n
A_{nn} = n + 1
A_{nk} = 0                         when k > n
```

The B vector has entries:

```text
B_n = sqrt(2n+1)
```

The script constructs these matrices numerically for a chosen N (the number of memory slots), then prints:

- The full matrix, so you can see the lower-triangular structure.
- The eigenvalues of `-A`, which are `-1, -2, ..., -N`. All negative, confirming the memory is provably stable.

### Encoding a signal

The script feeds a test signal through the HiPPO recurrence:

```text
new state = A_bar * old state + B_bar * current input
```

The signal has features at multiple time scales: a slow trend, a medium-speed bump, and a fast wiggle. The entire signal of length 100 (or 1000, or 10000: the choice does not matter) is compressed into exactly N numbers. The compression uses the same amount of memory regardless of how long the signal is.

### Reconstructing the signal

After encoding, the script reconstructs the signal from the stored N coefficients. The reconstruction quality depends on N:

- N = 4: the overall shape is captured but fine detail is lost.
- N = 8: the main features are visible.
- N = 16: the reconstruction closely follows the original.
- N = 32: essentially perfect for a smooth signal.

This is the core demonstration: a fixed-size memory can represent an arbitrary-length history with reconstruction accuracy that improves as you allow more coefficients.

### Fixed memory size regardless of sequence length

The script runs the same encoding on signals of length 100, 1000, and 10000. In all cases, the final memory state is N numbers. The memory requirement does not grow with sequence length. This is the practical advantage of HiPPO over attention: at inference time, a model using HiPPO-based memory uses the same amount of storage after reading 100 tokens as after reading 10000 tokens.

---

## What `visualize.py` does

This script generates three PNG figures and saves them in the `figures/` folder. The same figures appear in the markdown documents.

**Figure 1: `1_legendre_polynomials.png`**

Plots P0 through P5 over the interval from -1 to 1. You can see the key features directly:

- Each polynomial is bounded between -1 and +1.
- All polynomials pass through `+1` at `x = 1`.
- Pn crosses zero exactly n times.
- Even-numbered polynomials are symmetric (same shape left and right). Odd-numbered ones are antisymmetric (flipped).

**Figure 2: `2_hippo_reconstruction.png`**

Shows the same test signal and its reconstruction at four values of N: 4, 8, 16, and 32. The gap between the original and the reconstruction visibly shrinks as N grows. This is the "compression quality" picture.

**Figure 3: `3_memory_evolution.png`**

Shows how the memory coefficients (x0, x1, x2, ...) evolve in real time as the signal is read left to right, one sample at a time. You can watch the coefficients respond to events in the signal: a sudden spike causes the trend and curvature coefficients to react, then they gradually settle as the spike becomes part of the history.

---

## Connection to Mamba

The code here implements the foundation that Mamba builds on. The HiPPO A matrix you see printed and verified in `hippo.py` is the same backbone used in Mamba.

A full Mamba layer adds three things on top of what is here:

**1. A learnable C matrix and output projection.** Instead of reading all coefficients for reconstruction, the model learns which combinations of coefficients are useful for prediction.

**2. Input-dependent B, C, and step size Delta.** In this code, B and Delta are fixed: every input token is processed identically. Mamba makes these depend on the current token, so the model can choose to write an important token strongly into memory and let unimportant tokens pass through almost unchanged.

**3. A parallel scan.** Because B and Delta now vary per token, the simple convolution trick no longer applies. Mamba uses an associative parallel scan to compute the recurrence efficiently across a whole sequence in `O(log L)` parallel steps rather than `O(L)` sequential ones.

The `README_mamba.md` file in this folder explains these three additions in detail.

---

## Notes on the implementation

**Polynomial representation.** Polynomials are stored as NumPy arrays of coefficients `[a0, a1, a2, ...]` where `a0` is the constant term, `a1` is the coefficient of x, and so on. This is NumPy's standard convention. Multiplication, differentiation, addition, and evaluation are all done directly on these arrays, so you can inspect the state of a polynomial at any step.

**Exact inner products.** The inner product of two polynomials is computed by integrating the product term by term analytically. For a product of degree m and degree n, the integral from -1 to 1 of `x^k` is `2/(k+1)` when k is even and `0` when k is odd. This is evaluated exactly, not numerically sampled. As a result, the orthogonality check `integral of P_n * P_m = 0` produces results at machine precision (around 1e-16) rather than just "small compared to 1".

**Discretization.** The HiPPO recurrence uses the generalized bilinear transform with the LegS adaptive step size `1/t`. This is the standard formulation from the HiPPO paper. A small numerical difference from the exact Zero-Order Hold (ZOH) discretization is expected and does not affect the memory demonstration.
