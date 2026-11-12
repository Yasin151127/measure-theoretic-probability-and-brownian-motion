"""Basic simulation tools for standard Brownian motion."""
from __future__ import annotations
import numpy as np


def simulate_brownian(T: float = 1.0, n_steps: int = 500,
                      n_paths: int = 20, seed: int = 2026):
    """Simulate Brownian motion on an equally spaced grid.

    The increments are independent N(0, dt). The returned paths have
    shape (n_steps + 1, n_paths), with W(0)=0 in the first row.
    """
    if T <= 0 or n_steps < 1 or n_paths < 1:
        raise ValueError("T, n_steps and n_paths must be positive.")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    dW = np.sqrt(dt) * rng.standard_normal((n_steps, n_paths))
    paths = np.vstack((np.zeros((1, n_paths)), np.cumsum(dW, axis=0)))
    times = np.linspace(0.0, T, n_steps + 1)
    return times, paths


def quadratic_variation(paths: np.ndarray) -> np.ndarray:
    """Return sum_k (W_{k+1}-W_k)^2 for each simulated path."""
    increments = np.diff(paths, axis=0)
    return np.sum(increments**2, axis=0)


def left_riemann_integral(paths: np.ndarray, T: float) -> np.ndarray:
    """Approximate integral_0^T W_t dt with a left-endpoint sum."""
    if paths.ndim != 2 or paths.shape[0] < 2 or T <= 0:
        raise ValueError("paths must be a 2D grid array and T must be positive.")
    dt = T / (paths.shape[0] - 1)
    return dt * np.sum(paths[:-1, :], axis=0)
