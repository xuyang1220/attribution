# src/budget_sweep.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.uplift_bidding import (
    BiddingPolicyConfig,
    compare_policies,
    make_bid_cvr,
    make_bid_uplift,
    simulate_policy_with_alpha_search,
)


@dataclass
class PolicySpec:
    name: str
    score: np.ndarray
    bid_fn: Callable
    base_config: BiddingPolicyConfig


def run_budget_sweep(
    df: pd.DataFrame,
    target_spends: list[float] | np.ndarray,
    policies: list[PolicySpec],
    rel_tol: float = 1e-4,
    max_iter: int = 40,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    For each target spend, match each policy to that spend via alpha search.
    Returns one row per (target_spend, policy).
    """
    rows: list[dict] = []

    for target_spend in target_spends:
        if verbose:
            print(f"\n=== target_spend={target_spend:.2f} ===")

        for policy in policies:
            result = simulate_policy_with_alpha_search(
                df=df,
                score=policy.score,
                bid_fn=policy.bid_fn,
                base_config=policy.base_config,
                target_spend=float(target_spend),
                policy_name=policy.name,
                alpha_low=0.0,
                alpha_high=1.0,
                max_iter=max_iter,
                rel_tol=rel_tol,
                verbose=False,
            )
            result["target_spend_requested"] = float(target_spend)
            rows.append(result)

            if verbose:
                print(
                    f"{policy.name:24s} "
                    f"spend={result['spend']:.2f} "
                    f"inc_conv={result['expected_incremental_conversions']:.2f} "
                    f"inc_cpa={result['incremental_cpa']:.2f} "
                    f"alpha={result['alpha']:.6f}"
                )

    return pd.DataFrame(rows)