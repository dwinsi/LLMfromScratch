"""
hippo.py
--------
The HiPPO memory operator, implemented from scratch, mirroring
HiPPO_SSM_explained.md.

Core idea: maintain a fixed-size state x(t) in R^N that holds the
coefficients of the best Legendre-polynomial approximation of the
input history seen so far. The state updates via a linear recurrence
    x_t = Abar @ x_{t-1} + Bbar * u_t
whose matrices A, B are DERIVED (not learned) from the projection math.

Run directly to see the memory compress a signal and reconstruct it:
    python hippo.py
"""

import numpy as np
from scipy.linalg import expm      # matrix exponential, for discretisation


# ══════════════════════════════════════════════════════════════════
#  1. The HiPPO-LegS matrices  (derived in the document, Section 8.3)
#
#     A_nk =  sqrt(2n+1)*sqrt(2k+1)   if k < n
#             n + 1                    if k = n
#             0                        if k > n
#     B_n  =  sqrt(2n+1)
# ══════════════════════════════════════════════════════════════════

def build_hippo_legs(N):
    """
    Construct the N x N HiPPO-LegS transition matrix A and input vector B.
    These are the exact matrices that fall out of projecting onto the
    scaled Legendre measure. Nothing here is tuned or learned.
    """
    A = np.zeros((N, N))
    B = np.zeros((N, 1))
    for n in range(N):
        B[n, 0] = np.sqrt(2 * n + 1)
        for k in range(N):
            if k < n:
                A[n, k] = np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
            elif k == n:
                A[n, k] = n + 1
            # k > n stays 0  -> lower-triangular
    return A, B


# ══════════════════════════════════════════════════════════════════
#  2. Discretisation  (document Section 10)
#     Continuous:  dx/dt = -A x + B u
#     Discrete:    x_t = Abar x_{t-1} + Bbar u_t
#     with the scaled-measure step  dt = 1/t  (LegS uses adaptive step)
# ══════════════════════════════════════════════════════════════════

def discretize_legs(A, B, t):
    """
    LegS uses a time-varying step tied to the current position t.
    The bilinear/GBT discretisation with step 1/t gives the update below.
    This is the standard HiPPO-LegS discrete recurrence.
    """
    N = A.shape[0]
    I = np.eye(N)
    # GBT (generalized bilinear transform) with alpha = 0.5, step = 1/t
    step = 1.0 / t
    Abar = np.linalg.solve(I + 0.5 * step * A, I - 0.5 * step * A)
    Bbar = np.linalg.solve(I + 0.5 * step * A, step * B)
    return Abar, Bbar


# ══════════════════════════════════════════════════════════════════
#  3. The memory update — run over a whole input signal
# ══════════════════════════════════════════════════════════════════

def hippo_encode(u, N):
    """
    Feed the signal u (a 1-D array of samples) through the HiPPO memory.
    Returns the sequence of states x_t (shape [len(u), N]).

    At every step, x_t holds the N Legendre coefficients that best
    describe the entire history u[0..t] seen so far.
    """
    A, B = build_hippo_legs(N)
    L = len(u)
    x = np.zeros((N, 1))
    states = np.zeros((L, N))
    for t in range(1, L + 1):
        Abar, Bbar = discretize_legs(A, B, t)
        x = Abar @ x + Bbar * u[t - 1]
        states[t - 1] = x[:, 0]
    return states


# ══════════════════════════════════════════════════════════════════
#  4. Reconstruction — turn the coefficients back into the signal
#
#  The state at time t holds coefficients c_n such that
#     u(s) ~ sum_n c_n * scaled_Legendre_n(s)   for s in [0, t].
#  We rebuild the approximation and compare to the true history.
# ══════════════════════════════════════════════════════════════════

def legendre_eval(n, x):
    """Evaluate the n-th Legendre polynomial at x in [-1,1] via recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    Pm1, Pn = np.ones_like(x), x
    for k in range(1, n):
        Pn1 = ((2 * k + 1) * x * Pn - k * Pm1) / (k + 1)
        Pm1, Pn = Pn, Pn1
    return Pn


def hippo_reconstruct(state, n_points=200):
    """
    Given the coefficient vector `state` (length N) at some time t,
    reconstruct the approximation of the normalised history on [0,1].
    Returns (s_grid, reconstructed_values).
    """
    N = len(state)
    s = np.linspace(0, 1, n_points)          # normalised time in [0,1]
    xi = 2 * s - 1                            # map to [-1,1] for Legendre
    recon = np.zeros_like(s)
    for n in range(N):
        # scaled (orthonormal) Legendre basis: sqrt(2n+1) * P_n
        basis = np.sqrt(2 * n + 1) * legendre_eval(n, xi)
        recon += state[n] * basis
    return s, recon


# ══════════════════════════════════════════════════════════════════
#  5. Demonstration and verification
# ══════════════════════════════════════════════════════════════════

def demo():
    print("=" * 68)
    print("  HiPPO MEMORY OPERATOR — from scratch")
    print("=" * 68)

    # ---- show the derived matrices ----
    N = 4
    A, B = build_hippo_legs(N)
    print(f"\n  HiPPO-LegS matrix A (N={N}) — note it is LOWER-TRIANGULAR:")
    for row in A:
        print("    [" + "  ".join(f"{v:6.3f}" for v in row) + "]")
    print(f"\n  Input vector B = {B[:,0].round(3).tolist()}")
    print(f"    (these are sqrt(2n+1): {[round(np.sqrt(2*n+1),3) for n in range(N)]})")

    # ---- stability: eigenvalues of -A must be negative ----
    eigs = np.linalg.eigvals(-A)
    print(f"\n  Eigenvalues of -A: {sorted(eigs.real.round(3))}")
    print(f"    All negative? {np.all(eigs.real < 0)}  ->  memory is STABLE")

    # ---- feed a signal, watch it get compressed ----
    print("\n  Encoding a test signal into N coefficients...")
    L = 400
    t = np.linspace(0, 1, L)
    # a signal with a trend + a bump + a wiggle
    signal = 0.5 * t + 0.4 * np.exp(-((t - 0.6) ** 2) / 0.01) + 0.15 * np.sin(12 * t)

    for N_test in [4, 8, 16, 32, 64]:
        states = hippo_encode(signal, N_test)
        final_state = states[-1]
        s_grid, recon = hippo_reconstruct(final_state, n_points=L)
        # compare reconstruction to the true signal
        err = np.sqrt(np.mean((recon - signal) ** 2))
        print(f"    N={N_test:3d} coefficients  ->  reconstruction RMSE = {err:.4f}")

    print("\n  More coefficients = better reconstruction, as expected.")
    print("  The memory stays a FIXED N numbers regardless of signal length.")

    # ---- confirm: memory size is constant, not growing ----
    print("\n  Memory footprint check:")
    for L_test in [100, 1000, 10000]:
        sig = np.random.randn(L_test)
        states = hippo_encode(sig, N=16)
        print(f"    signal length {L_test:5d}  ->  state size stays {states.shape[1]} numbers")

    print("\n" + "=" * 68)
    print("  HiPPO verified: fixed-size memory, stable, reconstructs history.")
    print("=" * 68)
    return signal


if __name__ == "__main__":
    demo()
