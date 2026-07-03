"""
visualize.py
------------
Generate figures for the Legendre / HiPPO implementation:
  1. The first six Legendre polynomials plotted on [-1, 1]
  2. HiPPO reconstructing a signal from N coefficients, at increasing N
  3. The HiPPO memory "compressing" history as it reads a signal

    python visualize.py     # saves PNGs into ./figures/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless backend
import matplotlib.pyplot as plt

from legendre import legendre_recurrence
from numpy.polynomial import polynomial as Pnp
from hippo import hippo_encode, hippo_reconstruct


FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

# a clean, consistent palette
COLORS = ["#7F77DD", "#1D9E75", "#BA7517", "#D85A30", "#4A7FC0", "#B0508A"]


def plot_legendre_polynomials():
    """Figure 1 — the first six Legendre polynomials."""
    polys = legendre_recurrence(5)
    x = np.linspace(-1, 1, 400)

    fig, ax = plt.subplots(figsize=(8, 5))
    for n in range(6):
        y = Pnp.polyval(x, polys[n])
        ax.plot(x, y, color=COLORS[n], linewidth=2, label=f"$P_{n}(x)$")

    ax.axhline(0, color="#999", linewidth=0.8)
    ax.axvline(0, color="#999", linewidth=0.8)
    ax.axhline(1, color="#ccc", linewidth=0.6, linestyle="--")
    ax.axhline(-1, color="#ccc", linewidth=0.6, linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("$P_n(x)$")
    ax.set_title("The first six Legendre polynomials on [-1, 1]\n"
                 "(each has n roots — one 'wiggle' more than the last)")
    ax.legend(loc="lower right", ncol=2)
    ax.set_ylim(-1.3, 1.3)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "1_legendre_polynomials.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


def plot_hippo_reconstruction():
    """Figure 2 — HiPPO reconstructing a signal at increasing N."""
    L = 400
    t = np.linspace(0, 1, L)
    signal = (0.5 * t
              + 0.4 * np.exp(-((t - 0.6) ** 2) / 0.01)
              + 0.15 * np.sin(12 * t))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, N in zip(axes.flat, [4, 8, 16, 32]):
        states = hippo_encode(signal, N)
        s_grid, recon = hippo_reconstruct(states[-1], n_points=L)
        err = np.sqrt(np.mean((recon - signal) ** 2))

        ax.plot(t, signal, color="#333", linewidth=2, label="true signal")
        ax.plot(s_grid, recon, color="#D85A30", linewidth=2,
                linestyle="--", label=f"HiPPO (N={N})")
        ax.set_title(f"N = {N} coefficients   ·   RMSE = {err:.4f}")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.15)
        ax.set_xlabel("normalised time")

    fig.suptitle("HiPPO reconstructs a signal from a fixed number of coefficients\n"
                 "More coefficients → sharper reconstruction (the bump needs high N)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIGDIR, "2_hippo_reconstruction.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


def plot_memory_evolution():
    """Figure 3 — how the coefficients evolve as the signal is read."""
    L = 400
    t = np.linspace(0, 1, L)
    signal = (0.5 * t
              + 0.4 * np.exp(-((t - 0.6) ** 2) / 0.01)
              + 0.15 * np.sin(12 * t))

    N = 8
    states = hippo_encode(signal, N)     # shape [L, N]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # left: the input signal
    ax1.plot(t, signal, color="#333", linewidth=2)
    ax1.set_title("Input signal being read left → right")
    ax1.set_xlabel("time")
    ax1.grid(alpha=0.15)

    # right: each coefficient's value over time
    for n in range(N):
        ax2.plot(t, states[:, n], color=COLORS[n % len(COLORS)],
                 linewidth=1.6, label=f"$c_{n}$")
    ax2.set_title(f"The {N} HiPPO coefficients evolving as history grows")
    ax2.set_xlabel("time")
    ax2.set_ylabel("coefficient value")
    ax2.legend(loc="upper left", ncol=2, fontsize=8)
    ax2.grid(alpha=0.15)

    fig.suptitle("As the model reads the signal, it continuously updates a "
                 "fixed set of coefficients — its compressed memory",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(FIGDIR, "3_memory_evolution.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


if __name__ == "__main__":
    print("Generating figures...")
    plot_legendre_polynomials()
    plot_hippo_reconstruction()
    plot_memory_evolution()
    print("Done. See the ./figures/ folder.")
