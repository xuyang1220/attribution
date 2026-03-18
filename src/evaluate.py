# src/evaluate.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


EPS = 1e-12


def pehe(tau_true: np.ndarray, tau_hat: np.ndarray) -> float:
    """
    Precision in Estimation of Heterogeneous Effect:
        sqrt(mean((tau_hat - tau_true)^2))
    """
    tau_true = np.asarray(tau_true, dtype=float).reshape(-1)
    tau_hat = np.asarray(tau_hat, dtype=float).reshape(-1)
    _check_same_length(tau_true, tau_hat, "tau_true", "tau_hat")

    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def ate(tau: np.ndarray) -> float:
    tau = np.asarray(tau, dtype=float).reshape(-1)
    return float(np.mean(tau))


def ate_error(tau_true: np.ndarray, tau_hat: np.ndarray) -> float:
    tau_true = np.asarray(tau_true, dtype=float).reshape(-1)
    tau_hat = np.asarray(tau_hat, dtype=float).reshape(-1)
    _check_same_length(tau_true, tau_hat, "tau_true", "tau_hat")

    return float(np.mean(tau_hat) - np.mean(tau_true))


def rmse(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    x_true = np.asarray(x_true, dtype=float).reshape(-1)
    x_hat = np.asarray(x_hat, dtype=float).reshape(-1)
    _check_same_length(x_true, x_hat, "x_true", "x_hat")

    return float(np.sqrt(np.mean((x_hat - x_true) ** 2)))


def uplift_at_k(
    tau_true: np.ndarray,
    tau_hat: np.ndarray,
    frac: float = 0.1,
) -> float:
    """
    Mean true uplift among the top-k users ranked by predicted uplift.
    """
    tau_true = np.asarray(tau_true, dtype=float).reshape(-1)
    tau_hat = np.asarray(tau_hat, dtype=float).reshape(-1)
    _check_same_length(tau_true, tau_hat, "tau_true", "tau_hat")

    idx = _top_k_idx(tau_hat, frac=frac)
    return float(np.mean(tau_true[idx]))


def random_policy_uplift(tau_true: np.ndarray, frac: float = 0.1) -> float:
    """
    Expected mean uplift when treating a random frac of users.
    Under i.i.d. sampling, this is just the population mean uplift.
    """
    tau_true = np.asarray(tau_true, dtype=float).reshape(-1)
    _validate_frac(frac)
    return float(np.mean(tau_true))


def oracle_uplift_at_k(
    tau_true: np.ndarray,
    frac: float = 0.1,
) -> float:
    """
    Mean true uplift among the top-k users ranked by true uplift.
    """
    tau_true = np.asarray(tau_true, dtype=float).reshape(-1)
    idx = _top_k_idx(tau_true, frac=frac)
    return float(np.mean(tau_true[idx]))


def policy_value_from_probs(
    p0_true: np.ndarray,
    p1_true: np.ndarray,
    tau_hat: np.ndarray,
    frac: float = 0.1,
) -> float:
    """
    Treat top-k users ranked by tau_hat and compute expected conversion rate
    using oracle probabilities p0_true / p1_true.

    Policy:
      - treat top-k
      - do not treat the rest
    """
    p0_true = np.asarray(p0_true, dtype=float).reshape(-1)
    p1_true = np.asarray(p1_true, dtype=float).reshape(-1)
    tau_hat = np.asarray(tau_hat, dtype=float).reshape(-1)

    _check_same_length(p0_true, p1_true, "p0_true", "p1_true")
    _check_same_length(p0_true, tau_hat, "p0_true", "tau_hat")

    idx = _top_k_idx(tau_hat, frac=frac)

    policy_outcome = p0_true.copy()
    policy_outcome[idx] = p1_true[idx]
    return float(np.mean(policy_outcome))


def oracle_policy_value_from_probs(
    p0_true: np.ndarray,
    p1_true: np.ndarray,
    tau_true: np.ndarray,
    frac: float = 0.1,
) -> float:
    """
    Oracle policy value: treat top-k ranked by true uplift.
    """
    p0_true = np.asarray(p0_true, dtype=float).reshape(-1)
    p1_true = np.asarray(p1_true, dtype=float).reshape(-1)
    tau_true = np.asarray(tau_true, dtype=float).reshape(-1)

    _check_same_length(p0_true, p1_true, "p0_true", "p1_true")
    _check_same_length(p0_true, tau_true, "p0_true", "tau_true")

    idx = _top_k_idx(tau_true, frac=frac)

    policy_outcome = p0_true.copy()
    policy_outcome[idx] = p1_true[idx]
    return float(np.mean(policy_outcome))


def policy_gain_from_probs(
    p0_true: np.ndarray,
    p1_true: np.ndarray,
    tau_hat: np.ndarray,
    frac: float = 0.1,
) -> float:
    """
    Incremental conversion rate over treat-none baseline.
    """
    base = float(np.mean(np.asarray(p0_true, dtype=float)))
    value = policy_value_from_probs(p0_true=p0_true, p1_true=p1_true, tau_hat=tau_hat, frac=frac)
    return float(value - base)


def oracle_policy_gain_from_probs(
    p0_true: np.ndarray,
    p1_true: np.ndarray,
    tau_true: np.ndarray,
    frac: float = 0.1,
) -> float:
    base = float(np.mean(np.asarray(p0_true, dtype=float)))
    value = oracle_policy_value_from_probs(p0_true=p0_true, p1_true=p1_true, tau_true=tau_true, frac=frac)
    return float(value - base)


def cumulative_uplift_curve(
    tau_true: np.ndarray,
    tau_hat: np.ndarray,
    n_points: int = 20,
) -> pd.DataFrame:
    """
    Return a dataframe for an uplift / gain curve.

    Columns:
      - frac
      - mean_true_uplift_topk
      - oracle_mean_true_uplift_topk
      - random_mean_true_uplift
      - cumulative_gain_pred
      - cumulative_gain_oracle
      - cumulative_gain_random
    """
    tau_true = np.asarray(tau_true, dtype=float).reshape(-1)
    tau_hat = np.asarray(tau_hat, dtype=float).reshape(-1)
    _check_same_length(tau_true, tau_hat, "tau_true", "tau_hat")

    fracs = np.linspace(1 / n_points, 1.0, n_points)
    rows = []

    mean_random = float(np.mean(tau_true))

    for frac in fracs:
        idx_pred = _top_k_idx(tau_hat, frac=frac)
        idx_oracle = _top_k_idx(tau_true, frac=frac)

        mean_pred = float(np.mean(tau_true[idx_pred]))
        mean_oracle = float(np.mean(tau_true[idx_oracle]))

        # cumulative gain = average uplift in selected set * number selected
        k = len(idx_pred)
        gain_pred = mean_pred * k
        gain_oracle = mean_oracle * k
        gain_random = mean_random * k

        rows.append(
            {
                "frac": float(frac),
                "mean_true_uplift_topk": mean_pred,
                "oracle_mean_true_uplift_topk": mean_oracle,
                "random_mean_true_uplift": mean_random,
                "cumulative_gain_pred": gain_pred,
                "cumulative_gain_oracle": gain_oracle,
                "cumulative_gain_random": gain_random,
            }
        )

    return pd.DataFrame(rows)


def qini_curve_from_probs(
    p0_true: np.ndarray,
    p1_true: np.ndarray,
    tau_hat: np.ndarray,
    n_points: int = 20,
) -> pd.DataFrame:
    """
    Probability-based Qini-like curve using oracle p1 - p0.

    cumulative_incremental_conversions = sum_{treated_by_policy}(p1_true - p0_true)

    Columns:
      - frac
      - pred_incremental_conversions
      - oracle_incremental_conversions
      - random_incremental_conversions
    """
    p0_true = np.asarray(p0_true, dtype=float).reshape(-1)
    p1_true = np.asarray(p1_true, dtype=float).reshape(-1)
    tau_hat = np.asarray(tau_hat, dtype=float).reshape(-1)

    _check_same_length(p0_true, p1_true, "p0_true", "p1_true")
    _check_same_length(p0_true, tau_hat, "p0_true", "tau_hat")

    tau_true = p1_true - p0_true
    fracs = np.linspace(1 / n_points, 1.0, n_points)
    rows = []

    mean_tau = float(np.mean(tau_true))

    for frac in fracs:
        idx_pred = _top_k_idx(tau_hat, frac=frac)
        idx_oracle = _top_k_idx(tau_true, frac=frac)
        k = len(idx_pred)

        pred_inc = float(np.sum(tau_true[idx_pred]))
        oracle_inc = float(np.sum(tau_true[idx_oracle]))
        random_inc = float(mean_tau * k)

        rows.append(
            {
                "frac": float(frac),
                "pred_incremental_conversions": pred_inc,
                "oracle_incremental_conversions": oracle_inc,
                "random_incremental_conversions": random_inc,
            }
        )

    return pd.DataFrame(rows)


def calibration_by_uplift_bin(
    tau_true: np.ndarray,
    tau_hat: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Bin by predicted uplift and compare mean predicted uplift vs mean true uplift.
    """
    tau_true = np.asarray(tau_true, dtype=float).reshape(-1)
    tau_hat = np.asarray(tau_hat, dtype=float).reshape(-1)
    _check_same_length(tau_true, tau_hat, "tau_true", "tau_hat")

    df = pd.DataFrame({"tau_true": tau_true, "tau_hat": tau_hat})
    df["bin"] = pd.qcut(df["tau_hat"], q=n_bins, labels=False, duplicates="drop")

    out = (
        df.groupby("bin", observed=False)
        .agg(
            n=("tau_true", "size"),
            tau_true_mean=("tau_true", "mean"),
            tau_hat_mean=("tau_hat", "mean"),
            tau_true_std=("tau_true", "std"),
            tau_hat_std=("tau_hat", "std"),
        )
        .reset_index()
    )
    return out


@dataclass
class EvaluationSummary:
    pehe: float
    ate_true: float
    ate_hat: float
    ate_error: float
    uplift_at_10: float
    oracle_uplift_at_10: float
    policy_value_at_10: float
    oracle_policy_value_at_10: float
    policy_gain_at_10: float
    oracle_policy_gain_at_10: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "pehe": self.pehe,
            "ate_true": self.ate_true,
            "ate_hat": self.ate_hat,
            "ate_error": self.ate_error,
            "uplift_at_10": self.uplift_at_10,
            "oracle_uplift_at_10": self.oracle_uplift_at_10,
            "policy_value_at_10": self.policy_value_at_10,
            "oracle_policy_value_at_10": self.oracle_policy_value_at_10,
            "policy_gain_at_10": self.policy_gain_at_10,
            "oracle_policy_gain_at_10": self.oracle_policy_gain_at_10,
        }


def summarize_uplift_eval(
    tau_true: np.ndarray,
    tau_hat: np.ndarray,
    p0_true: np.ndarray,
    p1_true: np.ndarray,
    frac: float = 0.1,
) -> EvaluationSummary:
    tau_true = np.asarray(tau_true, dtype=float).reshape(-1)
    tau_hat = np.asarray(tau_hat, dtype=float).reshape(-1)
    p0_true = np.asarray(p0_true, dtype=float).reshape(-1)
    p1_true = np.asarray(p1_true, dtype=float).reshape(-1)

    _check_same_length(tau_true, tau_hat, "tau_true", "tau_hat")
    _check_same_length(tau_true, p0_true, "tau_true", "p0_true")
    _check_same_length(tau_true, p1_true, "tau_true", "p1_true")

    return EvaluationSummary(
        pehe=pehe(tau_true, tau_hat),
        ate_true=ate(tau_true),
        ate_hat=ate(tau_hat),
        ate_error=ate_error(tau_true, tau_hat),
        uplift_at_10=uplift_at_k(tau_true, tau_hat, frac=frac),
        oracle_uplift_at_10=oracle_uplift_at_k(tau_true, frac=frac),
        policy_value_at_10=policy_value_from_probs(p0_true, p1_true, tau_hat, frac=frac),
        oracle_policy_value_at_10=oracle_policy_value_from_probs(p0_true, p1_true, tau_true, frac=frac),
        policy_gain_at_10=policy_gain_from_probs(p0_true, p1_true, tau_hat, frac=frac),
        oracle_policy_gain_at_10=oracle_policy_gain_from_probs(p0_true, p1_true, tau_true, frac=frac),
    )


def print_eval_summary(summary: EvaluationSummary) -> None:
    d = summary.to_dict()
    for k, v in d.items():
        print(f"{k:28s}: {v:.6f}")


def _top_k_idx(score: np.ndarray, frac: float) -> np.ndarray:
    score = np.asarray(score, dtype=float).reshape(-1)
    _validate_frac(frac)

    n = len(score)
    k = max(1, int(np.ceil(n * frac)))

    # descending
    return np.argsort(-score)[:k]


def _validate_frac(frac: float) -> None:
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac must be in (0, 1], got {frac}")


def _check_same_length(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"{a_name} and {b_name} must have the same length: {len(a)} != {len(b)}")