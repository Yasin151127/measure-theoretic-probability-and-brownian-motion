"""Regenerate all figures used in the report."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from brownian import simulate_brownian, left_riemann_integral

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Experiment 1: sample paths.
t, W = simulate_brownian(T=1.0, n_steps=500, n_paths=12, seed=2026)
plt.figure(figsize=(8.0, 4.5))
for j in range(W.shape[1]):
    plt.plot(t, W[:, j], lw=1.0, alpha=0.78)
plt.axhline(0, lw=0.7, alpha=0.55)
plt.xlabel(r"Time $t$"); plt.ylabel(r"$W_t$")
plt.title("Sample paths of standard Brownian motion")
plt.xlim(0, 1.0); plt.grid(True, alpha=0.22); plt.tight_layout()
plt.savefig(FIG / "01_brownian_paths.png", dpi=220, bbox_inches="tight")
plt.close()

# Experiment 2: increment distributions at two resolutions.
for n, label in [(25, "coarse"), (400, "fine")]:
    rng = np.random.default_rng(2027 + n)
    dt = 1.0 / n
    values = np.sqrt(dt) * rng.standard_normal(20000)
    x = np.linspace(-4*np.sqrt(dt), 4*np.sqrt(dt), 500)
    density = np.exp(-x*x/(2*dt)) / np.sqrt(2*np.pi*dt)
    plt.figure(figsize=(7.5, 4.3))
    plt.hist(values, bins=55, density=True, alpha=0.58,
             label="Simulated increments")
    plt.plot(x, density, lw=2,
             label=r"Theoretical density: $N(0,\Delta t)$")
    plt.xlabel(r"Increment $\Delta W$"); plt.ylabel("Density")
    plt.title(f"Brownian increments ({label} grid, n={n}, $\\Delta t$={dt:.4g})")
    plt.legend(frameon=False); plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG / f"02_increment_{label}.png", dpi=220,
                bbox_inches="tight")
    plt.close()

# Experiment 3: quadratic variation on increasingly fine partitions.
step_counts = [8, 16, 32, 64, 128, 256, 512, 1024]
rng = np.random.default_rng(2028)
means, sds, n_paths = [], [], 6000
for n in step_counts:
    dt = 1.0 / n
    dW = np.sqrt(dt) * rng.standard_normal((n, n_paths))
    qv = np.sum(dW**2, axis=0)
    means.append(qv.mean()); sds.append(qv.std(ddof=1))
plt.figure(figsize=(7.8, 4.5))
plt.errorbar(step_counts, means, yerr=sds, fmt="o-", capsize=3,
             label="Simulated mean ± one cross-path SD")
plt.axhline(1.0, ls="--", label=r"Theoretical mean $T=1$")
plt.xscale("log", base=2); plt.xticks(step_counts, [str(x) for x in step_counts])
plt.xlabel("Number of intervals, n"); plt.ylabel(r"$Q_n$")
plt.title("Quadratic variation on progressively finer partitions")
plt.legend(frameon=False, fontsize=8); plt.grid(True, alpha=0.22)
plt.tight_layout()
plt.savefig(FIG / "03_quadratic_variation.png", dpi=220,
            bbox_inches="tight")
plt.close()

# Experiment 4: ordinary time integral (not an Itô stochastic integral).
_, paths = simulate_brownian(T=1.0, n_steps=500,
                             n_paths=20000, seed=2029)
values = left_riemann_integral(paths, T=1.0)
x = np.linspace(np.quantile(values, 0.001), np.quantile(values, 0.999), 500)
variance = 1.0 / 3.0
pdf = np.exp(-x*x/(2*variance)) / np.sqrt(2*np.pi*variance)
plt.figure(figsize=(7.8, 4.5))
plt.hist(values, bins=65, density=True, alpha=0.58,
         label="Simulated Riemann sums")
plt.plot(x, pdf, lw=2, label=r"$N(0,T^3/3)$ limiting density")
plt.xlabel(r"Approximation to $I_T=\int_0^T W_t\,dt$")
plt.ylabel("Density")
plt.title("Distribution of the time integral of Brownian motion")
plt.legend(frameon=False); plt.grid(True, axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(FIG / "04_time_integral.png", dpi=220, bbox_inches="tight")
plt.close()

print(f"All report figures regenerated in {FIG}")
