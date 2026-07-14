#!/usr/bin/env python3
"""Finite-member sensitivity audit for the external metamodel ensembles.

The audit enumerates every subset of the ten archived members for M in
{3, 5, 7, 9, 10}.  It quantifies member-selection sensitivity only; the
subsets are not independent training repeats.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.stats import rankdata


FILES = (
    "external_members_tmm_easy.npz",
    "external_members_tmm_stressed.npz",
    "external_members_heat_poisson_stressed.npz",
)
M_VALUES = (3, 5, 7, 9, 10)


def auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cosine_mean(pred: np.ndarray, ref: np.ndarray) -> float:
    numerator = np.sum(pred * ref, axis=1)
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(ref, axis=1)
    values = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return float(np.nanmean(values))


def quantile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q)) if values.notna().any() else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--analysis-dir", type=Path, default=here)
    parser.add_argument("--package-dir", type=Path, default=here)
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    package_dir = args.package_dir.resolve()

    rows: list[dict] = []
    for filename in FILES:
        with np.load(package_dir / filename, allow_pickle=True) as archive:
            members = np.asarray(archive["member_gradients"], dtype=float)
            reference = np.asarray(archive["reference_gradients"], dtype=float)
            dataset = str(archive["dataset"].item())
        _, total_members, k = members.shape
        ref_sign = np.sign(reference)
        for m in M_VALUES:
            for subset_index, subset in enumerate(combinations(range(total_members), m)):
                selected = members[:, subset, :]
                mean = selected.mean(axis=1)
                std = selected.std(axis=1, ddof=0)
                mean_sign = np.sign(mean)
                correct = (mean_sign == ref_sign) & (ref_sign != 0)
                aligned = (np.sign(selected) == mean_sign[:, None, :]).mean(axis=1)
                aligned[mean_sign == 0] = 0.0
                snr = np.divide(np.abs(mean), std, out=np.full_like(mean, np.inf), where=std > 0)
                magnitude = np.abs(mean)
                trusted = aligned >= 0.9 - 1e-12
                accepted = int(trusted.sum())
                false_accepted = int((trusted & ~correct).sum())
                rows.append(
                    {
                        "dataset": dataset,
                        "M": m,
                        "subset_index": subset_index,
                        "members": ";".join(map(str, subset)),
                        "sign_auc": auc_binary(correct, aligned),
                        "snr_auc": auc_binary(correct, snr),
                        "magnitude_auc": auc_binary(correct, magnitude),
                        "tau09_coverage": float(trusted.mean()),
                        "tau09_false_trust_risk": false_accepted / accepted if accepted else float("nan"),
                        "zero_budget_cosine": cosine_mean(mean, reference),
                        "accepted_components": accepted,
                        "total_components": int(correct.size),
                        "K": k,
                    }
                )

    runs = pd.DataFrame.from_records(rows)
    summary_rows: list[dict] = []
    metrics = (
        "sign_auc",
        "snr_auc",
        "magnitude_auc",
        "tau09_coverage",
        "tau09_false_trust_risk",
        "zero_budget_cosine",
    )
    for (dataset, m), group in runs.groupby(["dataset", "M"], sort=False):
        record = {"dataset": dataset, "M": int(m), "subsets": len(group)}
        for metric in metrics:
            record[f"{metric}_median"] = quantile(group[metric], 0.50)
            record[f"{metric}_q10"] = quantile(group[metric], 0.10)
            record[f"{metric}_q90"] = quantile(group[metric], 0.90)
        summary_rows.append(record)
    summary = pd.DataFrame.from_records(summary_rows)

    for frame, name in (
        (runs, "subensemble_m_metrics.csv"),
        (summary, "subensemble_m_summary.csv"),
    ):
        analysis_path = analysis_dir / name
        frame.to_csv(analysis_path, index=False, float_format="%.12g")
        package_path = package_dir / name
        if analysis_path.resolve() != package_path.resolve():
            shutil.copy2(analysis_path, package_path)

    script_target = package_dir / Path(__file__).name
    if Path(__file__).resolve() != script_target.resolve():
        shutil.copy2(Path(__file__).resolve(), script_target)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
