"""
selective_ssm_numpy.py
----------------------
The SELECTIVE state space mechanism (the heart of Mamba, "S6"),
implemented in pure NumPy so it runs anywhere — no PyTorch needed.

This shows the actual algorithm with no framework magic:
  - B, C, and the step size Delta are computed FROM the input (selectivity)
  - the recurrence is run with an associative SCAN
  - forgetting/remembering is now content-dependent, unlike vanilla SSM

For the trainable version with autograd, see mamba_block_torch.py.
This file is about SEEING the mechanism clearly.

    python selective_ssm_numpy.py
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════
#  The discretization: continuous (A, B) -> discrete (Abar, Bbar)
#  using Zero-Order Hold, but now Delta VARIES per token.
# ══════════════════════════════════════════════════════════════════

def discretize(A, B, delta):
    """
    A:     (N,)       diagonal state matrix (one value per state dim)
    B:     (L, N)     input matrix, DIFFERENT for each timestep (selective!)
    delta: (L,)       step size, DIFFERENT for each timestep (selective!)

    Returns
    Abar:  (L, N)     discretized A, one per timestep
    Bbar:  (L, N)     discretized B, one per timestep

    ZOH formulas (diagonal case):
        Abar = exp(delta * A)
        Bbar = (Abar - 1) / A * B    ~=  delta * B   (the standard simplification)
    """
    # delta[:, None] broadcasts step size across the N state dims
    Abar = np.exp(delta[:, None] * A[None, :])          # (L, N)
    Bbar = delta[:, None] * B                            # (L, N)
    return Abar, Bbar


# ══════════════════════════════════════════════════════════════════
#  The SELECTIVE SCAN — run the recurrence  h_t = Abar_t * h_{t-1} + Bbar_t * u_t
#
#  Two implementations:
#   (1) sequential   — the obvious loop, O(L)
#   (2) parallel scan — associative combine, the trick that makes
#                        Mamba trainable fast on GPUs
#  They give identical results; we verify that below.
# ══════════════════════════════════════════════════════════════════

def selective_scan_sequential(Abar, Bbar, u, C):
    """
    The straightforward recurrence.
    Abar: (L, N)   Bbar: (L, N)   u: (L,)   C: (L, N)
    Returns y: (L,)
    """
    L, N = Abar.shape
    h = np.zeros(N)
    ys = np.zeros(L)
    for t in range(L):
        h = Abar[t] * h + Bbar[t] * u[t]     # update state (elementwise, diagonal A)
        ys[t] = C[t] @ h                     # read output through selective C
    return ys


def selective_scan_parallel(Abar, Bbar, u, C):
    """
    The PARALLEL SCAN. Key insight: the recurrence
        h_t = a_t * h_{t-1} + b_t
    (with a_t = Abar_t and b_t = Bbar_t * u_t) is an ASSOCIATIVE operation,
    so we can compute all h_t together using a prefix-scan instead of a loop.

    The associative combine for two segments (a1,b1) then (a2,b2) is:
        (a2*a1,  a2*b1 + b2)

    Here we use the simple Hillis-Steele inclusive scan (O(L log L) steps,
    each fully vectorized). This is the CPU-friendly cousin of the
    hardware-aware scan Mamba uses on GPUs.
    """
    L, N = Abar.shape
    a = Abar.copy()                 # (L, N) multiplicative coefficients
    b = (Bbar * u[:, None]).copy()  # (L, N) additive terms

    # Hillis-Steele inclusive scan over the time axis
    shift = 1
    while shift < L:
        # combine element t with element (t - shift)
        a_prev = np.zeros_like(a)
        b_prev = np.zeros_like(b)
        a_prev[shift:] = a[:-shift]
        b_prev[shift:] = b[:-shift]
        # identity element for the "gap" positions is (a=1, b=0)
        a_prev[:shift] = 1.0
        b_prev[:shift] = 0.0
        # associative combine:  new = current ∘ previous
        #   a_new = a_curr * a_prev
        #   b_new = a_curr * b_prev + b_curr
        b = a * b_prev + b
        a = a * a_prev
        shift *= 2

    h_all = b                       # (L, N) — all hidden states at once
    y = np.sum(C * h_all, axis=1)   # read output: (L,)
    return y


# ══════════════════════════════════════════════════════════════════
#  The SELECTIVE mechanism — B, C, Delta are FUNCTIONS OF THE INPUT
#  This is the one idea that turns a vanilla SSM into Mamba.
# ══════════════════════════════════════════════════════════════════

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)   # stable softplus


class SelectiveSSM_numpy:
    """
    A minimal selective SSM demonstrating the core mechanism.
    Weights are random (this is for SEEING the forward pass, not training).
    """
    def __init__(self, d_model, N=16, seed=0):
        rng = np.random.default_rng(seed)
        self.N = N
        # A is fixed and structured (HiPPO-inspired: negative for stability).
        # Diagonal, initialized as -(1..N) like the S4D real init.
        self.A = -(np.arange(1, N + 1)).astype(float)          # (N,)
        # Projections that make B, C, Delta INPUT-DEPENDENT.
        self.W_B     = rng.standard_normal((d_model, N)) * 0.1  # x -> B
        self.W_C     = rng.standard_normal((d_model, N)) * 0.1  # x -> C
        self.W_delta = rng.standard_normal(d_model) * 0.1       # x -> delta
        self.d_model = d_model

    def forward(self, x, use_parallel=True):
        """
        x: (L, d_model) input sequence.
        We treat the mean over channels as the scalar signal u for this demo.
        Returns y: (L,)
        """
        L = x.shape[0]
        # --- selectivity: compute B, C, delta FROM the input ---
        B_sel     = x @ self.W_B                       # (L, N)  input-dependent B
        C_sel     = x @ self.W_C                       # (L, N)  input-dependent C
        delta_sel = softplus(x @ self.W_delta)         # (L,)    input-dependent step > 0

        u = x.mean(axis=1)                             # (L,) scalar signal per step

        Abar, Bbar = discretize(self.A, B_sel, delta_sel)

        if use_parallel:
            return selective_scan_parallel(Abar, Bbar, u, C_sel)
        return selective_scan_sequential(Abar, Bbar, u, C_sel)


# ══════════════════════════════════════════════════════════════════
#  Demonstration
# ══════════════════════════════════════════════════════════════════

def demo():
    print("=" * 70)
    print("  SELECTIVE SSM (the Mamba mechanism) — NumPy")
    print("=" * 70)

    d_model, N, L = 8, 16, 64
    model = SelectiveSSM_numpy(d_model, N, seed=1)

    rng = np.random.default_rng(42)
    x = rng.standard_normal((L, d_model))

    # --- 1. parallel scan == sequential scan ? ---
    print("\n  [1] Parallel scan matches the sequential recurrence?")
    y_seq = model.forward(x, use_parallel=False)
    y_par = model.forward(x, use_parallel=True)
    max_diff = np.max(np.abs(y_seq - y_par))
    print(f"      max difference = {max_diff:.2e}   ->  {'MATCH' if max_diff < 1e-10 else 'MISMATCH'}")

    # --- 2. selectivity: B, C, delta actually depend on input ---
    print("\n  [2] Are B, C, Delta actually input-dependent (selective)?")
    x2 = rng.standard_normal((L, d_model))
    B1 = x  @ model.W_B
    B2 = x2 @ model.W_B
    d1 = softplus(x  @ model.W_delta)
    d2 = softplus(x2 @ model.W_delta)
    print(f"      B differs for different inputs: {not np.allclose(B1, B2)}")
    print(f"      Delta differs for different inputs: {not np.allclose(d1, d2)}")
    print(f"      Delta range on this input: [{d1.min():.3f}, {d1.max():.3f}]  (all > 0)")

    # --- 3. stability: A is negative -> state stays bounded ---
    print("\n  [3] Is the state stable (A negative)?")
    print(f"      A diagonal = {model.A.astype(int).tolist()}")
    print(f"      all negative -> {np.all(model.A < 0)}  ->  discretized |Abar| < 1, stable")

    # --- 4. selective copying: the task Mamba was designed to win ---
    print("\n  [4] Why selectivity matters: a content-aware memory.")
    print("      A vanilla SSM treats every token the same. A selective SSM")
    print("      can make Delta large (write hard) or small (skip) per token,")
    print("      based on the token's content. That is the whole advantage.")

    # show delta reacting to a 'marker' token
    x_marked = rng.standard_normal((L, d_model)) * 0.1
    x_marked[10] += 5.0    # a big, distinctive 'important' token at position 10
    delta_marked = softplus(x_marked @ model.W_delta)
    print(f"\n      Delta at ordinary positions ~ {np.median(delta_marked):.3f}")
    print(f"      Delta at the 'important' token (pos 10) = {delta_marked[10]:.3f}")
    print(f"      -> the model can allocate more 'write strength' to it")

    print("\n" + "=" * 70)
    print("  Selective SSM working: input-dependent B/C/Delta + parallel scan.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
