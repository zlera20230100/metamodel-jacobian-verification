"""Reproducible checks for the scale-invariance/bias-blindness appendix.

The script produces ``fig_toy_simpat.pdf`` and ``fig_toy_simpat.png`` next
to itself.  It deliberately separates two statements that are easy to
conflate:

1. A positive component-wise rescaling cannot change signs, mean-aligned sign
   agreement, or sign-correctness labels, whereas it can change every
   magnitude ranking.
2. Mean-aligned sign agreement measures concentration around the ensemble
   population mean.  It cannot determine whether that mean is aligned with a
   reference gradient.

No project data are overwritten.  The only stochastic calculation is a small
Monte Carlo study with fixed seeds; the bias panel uses closed-form Gaussian
probabilities.
"""

from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
M = 10
N_COMPONENTS = 4_000
N_REPEATS = 48
BASE_SEED = 20260711
GAMMAS = np.linspace(-2.0, 2.0, 81)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based ranks with midranks for ties (NumPy-only)."""

    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        # Positions start+1, ..., stop have average (start+1+stop)/2.
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann--Whitney form of ROC AUC, including tied scores."""

    label = np.asarray(label, dtype=bool)
    n_pos = int(label.sum())
    n_neg = int(label.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC AUC requires both label classes")
    ranks = average_ranks(np.asarray(score, dtype=float))
    u_stat = ranks[label].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


def one_repeat(seed: int) -> tuple[np.ndarray, float, float]:
    """Return magnitude AUC(gamma), mean-aligned AUC, and wrong-sign fraction."""

    rng = np.random.default_rng(seed)
    r = np.exp(
        rng.uniform(np.log(0.05), np.log(5.0), size=N_COMPONENTS)
    )
    truth_sign = rng.choice(np.array([-1.0, 1.0]), size=N_COMPONENTS)

    # Canonical, unbiased ensemble: X_m = r*s* + epsilon_m, sigma=1.
    canonical = (
        r[None, :] * truth_sign[None, :]
        + rng.standard_normal((M, N_COMPONENTS))
    )
    ensemble_mean = canonical.mean(axis=0)
    mean_sign = np.sign(ensemble_mean)
    if np.any(mean_sign == 0):
        raise AssertionError("exact-zero ensemble mean must be sent to assessment")
    agreement = (np.sign(canonical) == mean_sign[None, :]).mean(axis=0)
    correct = mean_sign == truth_sign
    base_magnitude = np.mean(np.abs(canonical), axis=0)

    direction_auc = roc_auc(agreement, correct)
    magnitude_auc = np.empty(GAMMAS.size, dtype=float)
    for idx, gamma in enumerate(GAMMAS):
        scale = r**gamma
        magnitude_auc[idx] = roc_auc(scale * base_magnitude, correct)

    # Exact finite-sample invariance check under arbitrary positive scales.
    random_scale = np.exp(rng.normal(0.0, 2.0, size=N_COMPONENTS))
    rescaled = random_scale[None, :] * canonical
    rescaled_mean_sign = np.sign(rescaled.mean(axis=0))
    rescaled_agreement = (
        np.sign(rescaled) == rescaled_mean_sign[None, :]
    ).mean(axis=0)
    rescaled_correct = rescaled_mean_sign == truth_sign
    if not np.array_equal(agreement, rescaled_agreement):
        raise AssertionError("positive scaling changed sign agreement")
    if not np.array_equal(correct, rescaled_correct):
        raise AssertionError("positive scaling changed sign-correctness labels")

    return magnitude_auc, direction_auc, float(1.0 - correct.mean())


def normal_cdf(x: np.ndarray | float) -> np.ndarray:
    """Standard-normal CDF without a SciPy dependency."""

    x_arr = np.asarray(x, dtype=float)
    return np.vectorize(lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0))))(x_arr)


def interpolate_crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """Linear interpolation at the first crossing of y=level."""

    delta = y - level
    crossing = np.flatnonzero(delta[:-1] * delta[1:] <= 0.0)
    if crossing.size == 0:
        return float("nan")
    idx = int(crossing[0])
    if delta[idx] == 0.0:
        return float(x[idx])
    weight = -delta[idx] / (delta[idx + 1] - delta[idx])
    return float(x[idx] + weight * (x[idx + 1] - x[idx]))


def value_at(grid: np.ndarray, values: np.ndarray, point: float) -> float:
    return float(np.interp(point, grid, values))


def main() -> None:
    magnitude_runs = []
    direction_runs = []
    wrong_runs = []
    for repeat in range(N_REPEATS):
        mag_auc, dir_auc, wrong = one_repeat(BASE_SEED + repeat)
        magnitude_runs.append(mag_auc)
        direction_runs.append(dir_auc)
        wrong_runs.append(wrong)

    magnitude_runs = np.asarray(magnitude_runs)
    direction_runs = np.asarray(direction_runs)
    wrong_runs = np.asarray(wrong_runs)

    mag_mean = magnitude_runs.mean(axis=0)
    mag_lo, mag_hi = np.percentile(magnitude_runs, [2.5, 97.5], axis=0)
    dir_mean = float(direction_runs.mean())
    dir_lo, dir_hi = np.percentile(direction_runs, [2.5, 97.5])
    crossing = interpolate_crossing(GAMMAS, mag_mean, 0.5)

    # Bias panel.  Reference truth is g*=+1.  The aligned world has mu>0;
    # the shared-bias world has mu<0 because b<-g*.  Reflection flips every
    # member sign and the ensemble-mean sign together, so it leaves the
    # mean-aligned agreement score unchanged.
    r_grid = np.linspace(0.0, 1.8, 300)
    agreement_limit = normal_cdf(r_grid)
    correct_aligned = normal_cdf(np.sqrt(M) * r_grid)
    correct_flipped = normal_cdf(-np.sqrt(M) * r_grid)

    if not np.allclose(correct_aligned + correct_flipped, 1.0, atol=1e-12):
        raise AssertionError("aligned and reflected correctness must sum to one")

    # Explicit paired non-identifiability check at finite M.
    check_rng = np.random.default_rng(BASE_SEED + 10_000)
    check_r = np.exp(check_rng.uniform(np.log(0.05), np.log(5.0), 10_000))
    aligned_draws = check_r[None, :] + check_rng.standard_normal((M, 10_000))
    flipped_draws = -aligned_draws
    aligned_mean_sign = np.sign(aligned_draws.mean(axis=0))
    flipped_mean_sign = np.sign(flipped_draws.mean(axis=0))
    a_aligned = (
        np.sign(aligned_draws) == aligned_mean_sign[None, :]
    ).mean(axis=0)
    a_flipped = (
        np.sign(flipped_draws) == flipped_mean_sign[None, :]
    ).mean(axis=0)
    if not np.array_equal(a_aligned, a_flipped):
        raise AssertionError("reflection must leave sign agreement unchanged")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )

    blue = "#0077BB"
    orange = "#EE7733"
    red = "#CC3311"
    grey = "#666666"

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15))

    ax = axes[0]
    ax.fill_between(GAMMAS, mag_lo, mag_hi, color=orange, alpha=0.16, lw=0)
    ax.plot(GAMMAS, mag_mean, color=orange, label=r"magnitude: $r^\gamma\,\overline{|X|}$")
    ax.fill_between(
        GAMMAS,
        np.full_like(GAMMAS, dir_lo),
        np.full_like(GAMMAS, dir_hi),
        color=blue,
        alpha=0.12,
        lw=0,
    )
    ax.axhline(
        dir_mean,
        color=blue,
        ls="--",
        label=r"mean-aligned agreement $a_{\mu,M}$",
    )
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    ax.axvline(0.0, color=grey, lw=0.8, ls=":")
    ax.scatter(
        [-1.0, -0.2, 1.0],
        [value_at(GAMMAS, mag_mean, x) for x in (-1.0, -0.2, 1.0)],
        color=orange,
        edgecolor="black",
        linewidth=0.35,
        s=24,
        zorder=4,
    )
    ax.annotate(
        rf"chance crossing $\gamma\approx{crossing:.2f}$",
        xy=(crossing, 0.5),
        xytext=(-1.78, 0.57),
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": grey},
        fontsize=7.7,
        color=grey,
    )
    ax.text(-1.94, 0.505, "chance", fontsize=7.5, va="bottom")
    ax.set_xlim(GAMMAS.min(), GAMMAS.max())
    ax.set_ylim(0.10, 0.92)
    ax.set_xlabel(r"scale--SNR exponent $\gamma$ in $c(r)=r^\gamma$")
    ax.set_ylabel("AUC for ensemble-mean sign correctness")
    ax.set_title("(a) Magnitude inversion has a data-dependent crossing", loc="left")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    ax.plot(
        r_grid,
        agreement_limit,
        color=grey,
        ls="--",
        label=r"large-$M$ limit $\Phi(r)$ (both regimes)",
    )
    ax.plot(
        r_grid,
        correct_aligned,
        color=blue,
        label=r"$P(C=1\mid r)$, $\mu g^\star>0$",
    )
    ax.plot(
        r_grid,
        correct_flipped,
        color=red,
        ls="-.",
        label=r"$P(C=1\mid r)$, $\mu g^\star<0$",
    )
    mark_r = 1.0
    mark_a = float(normal_cdf(mark_r))
    mark_good = float(normal_cdf(np.sqrt(M) * mark_r))
    mark_bad = 1.0 - mark_good
    ax.plot([mark_r, mark_r], [mark_bad, mark_good], color="#BBBBBB", lw=0.8)
    ax.scatter(
        [mark_r, mark_r, mark_r],
        [mark_a, mark_good, mark_bad],
        c=[grey, blue, red],
        edgecolor="black",
        linewidth=0.35,
        s=24,
        zorder=4,
    )
    ax.text(
        1.03,
        0.48,
        r"same limiting score;" "\n" r"opposite truth alignment",
        fontsize=7.7,
        color=grey,
        va="center",
    )
    ax.set_xlim(0.0, 1.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"population concentration $r=|\mu|/\sigma$")
    ax.set_ylabel("probability / limiting score")
    ax.set_title("(b) Mean-aligned agreement is scale-free but bias-blind", loc="left")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.01, 0.03))

    for axis in axes:
        axis.tick_params(direction="out", length=3.5, width=0.8)

    fig.tight_layout(w_pad=2.2)
    pdf_path = HERE / "fig_toy_simpat.pdf"
    png_path = HERE / "fig_toy_simpat.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    print(f"M={M}, components/repeat={N_COMPONENTS}, repeats={N_REPEATS}")
    print(
        "wrong-sign fraction: "
        f"mean={wrong_runs.mean():.4f}, "
        f"95% MC range=[{np.percentile(wrong_runs, 2.5):.4f}, "
        f"{np.percentile(wrong_runs, 97.5):.4f}]"
    )
    print(
        "mean-aligned agreement AUC (invariant in gamma): "
        f"mean={dir_mean:.4f}, 95% MC range=[{dir_lo:.4f}, {dir_hi:.4f}]"
    )
    for gamma in (-1.0, -0.2, 0.0, 1.0):
        idx = int(np.argmin(np.abs(GAMMAS - gamma)))
        print(
            f"magnitude AUC at gamma={GAMMAS[idx]:+.1f}: "
            f"mean={mag_mean[idx]:.4f}, "
            f"95% MC range=[{mag_lo[idx]:.4f}, {mag_hi[idx]:.4f}]"
        )
    print(f"model-specific magnitude chance crossing: gamma~{crossing:.3f}")
    print(
        "finite-M reflection check: aligned and bias-flipped agreement arrays "
        "are exactly identical"
    )
    print(f"saved {pdf_path.name} and {png_path.name}")


if __name__ == "__main__":
    main()
