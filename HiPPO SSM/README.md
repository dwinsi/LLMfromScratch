# Legendre Polynomials & HiPPO — Code Implementation

Runnable Python that implements everything in the companion documents
(`Legendre_polynomials_complete.md` and `HiPPO_SSM_explained.md`) from scratch,
with verification and visualization.

Nothing here uses a black-box library for the math itself — the Legendre
polynomials and the HiPPO matrices are built by hand from their definitions,
then checked against known results.

---

## Files

| File | What it does |
|------|--------------|
| `legendre.py` | Legendre polynomials built four independent ways (Gram–Schmidt, Rodrigues, recurrence, generating function), with full property verification |
| `hippo.py` | The HiPPO memory operator: the derived LegS matrices, discretization, encoding a signal into fixed-size memory, and reconstructing it |
| `visualize.py` | Generates three figures (polynomials, reconstruction quality, memory evolution) |
| `requirements.txt` | Dependencies (numpy, scipy, matplotlib) |

---

## Quick start

```bash
pip install -r requirements.txt

python legendre.py     # build & verify Legendre polynomials from scratch
python hippo.py        # build & verify the HiPPO memory operator
python visualize.py    # generate figures into ./figures/
```

---

## What each script proves

### `legendre.py`

Builds $P_0$ through $P_5$ **four different ways** and confirms they all agree:

1. **Gram–Schmidt** — orthogonalize the monomials $1, x, x^2, \dots$ using an exact polynomial inner product.
2. **Rodrigues' formula** — $P_n = \frac{1}{2^n n!}\frac{d^n}{dx^n}(x^2-1)^n$, with the derivatives done symbolically on coefficient arrays.
3. **Three-term recurrence** — $(n+1)P_{n+1} = (2n+1)xP_n - nP_{n-1}$.
4. **Generating function** — numerical cross-check.

Then it verifies every property from the document:
- Orthogonality: $\int_{-1}^1 P_n P_m\,dx = 0$ for $n \neq m$
- Norm: $\int_{-1}^1 P_n^2\,dx = \frac{2}{2n+1}$
- Endpoints: $P_n(1) = 1$, $P_n(-1) = (-1)^n$
- The recurrence identity holds

### `hippo.py`

Builds the **HiPPO-LegS matrices** exactly as derived in the document:

$$
A_{nk} = \begin{cases} \sqrt{2n+1}\sqrt{2k+1} & k<n \\ n+1 & k=n \\ 0 & k>n \end{cases}
\qquad B_n = \sqrt{2n+1}
$$

Then it demonstrates:
- **The matrix is lower-triangular** (printed to confirm).
- **Eigenvalues of $-A$ are $-1, -2, -3, -4$** — all negative, so the memory is provably stable.
- **Encoding**: feed a test signal (trend + bump + wiggle) through the recurrence and store only $N$ numbers.
- **Reconstruction**: rebuild the signal from those $N$ coefficients; RMSE drops as $N$ grows.
- **Fixed memory**: signal of length 100, 1000, or 10000 all compress to the same $N$ numbers.

### `visualize.py`

Produces three PNGs in `./figures/`:

1. **`1_legendre_polynomials.png`** — the first six polynomials; each has one more root than the last.
2. **`2_hippo_reconstruction.png`** — the same signal reconstructed at $N = 4, 8, 16, 32$. At $N=4$ the sharp bump is missed; by $N=32$ the reconstruction is essentially perfect.
3. **`3_memory_evolution.png`** — the coefficients evolving in real time as the signal is read left to right.

---

## The connection to Mamba

This code implements the *foundation*. A full Mamba layer adds three things on top of what's here:

1. **Learnable $C$ and output projection** — read the stored coefficients into a prediction.
2. **Input-dependent $B$, $C$, and step size $\Delta$** — the "selectivity" that lets the model choose what to remember per token.
3. **A parallel scan** — to run the selective recurrence efficiently during training.

The stable, structured memory you see here (the HiPPO $A$ matrix) remains the backbone in all of them.

---

## Notes on the implementation

- Polynomials are stored as coefficient arrays `[a0, a1, a2, ...]` (low-to-high), NumPy's convention. Multiplication, differentiation, and evaluation are done on these arrays directly, so you can trace every operation.
- The inner product for polynomials is computed **exactly** (term-by-term integration), not numerically — so orthogonality checks come out to machine zero, not just "small".
- The HiPPO discretization uses the generalized bilinear transform with the LegS adaptive step $1/t$, which is the standard formulation. A tiny numerical difference from a pure ZOH is expected and does not affect the demonstration.
