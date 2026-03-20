
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd

@dataclass
class BiddingPolicyConfig:
    value_per_conversion: float = 100.0
    alpha: float = 1.0
    bid_cap: float | None = None


def make_bid_cvr(
    p1_score: np.ndarray,
    config: BiddingPolicyConfig,
) -> np.ndarray:
    p1_score = np.asarray(p1_score, dtype=float)
    bids = config.alpha * p1_score * config.value_per_conversion
    if config.bid_cap is not None:
        bids = np.minimum(bids, config.bid_cap)
    return bids


def make_bid_uplift(
    tau_score: np.ndarray,
    config: BiddingPolicyConfig,
) -> np.ndarray:
    tau_score = np.asarray(tau_score, dtype=float)
    bids = config.alpha * np.clip(tau_score, 0.0, None) * config.value_per_conversion
    if config.bid_cap is not None:
        bids = np.minimum(bids, config.bid_cap)
    return bids


def simulate_policy_outcomes(
    df: pd.DataFrame,
    bids: np.ndarray,
    policy_name: str,
    sample_realized: bool = False,
    seed: int = 42,
) -> Dict[str, float]:
    """
    df must contain:
      market_price, p0_true, p1_true, tau_true
    """
    rng = np.random.default_rng(seed)

    market_price = df["market_price"].to_numpy(dtype=float)
    p0_true = df["p0_true"].to_numpy(dtype=float)
    p1_true = df["p1_true"].to_numpy(dtype=float)
    tau_true = df["tau_true"].to_numpy(dtype=float)

    bids = np.asarray(bids, dtype=float)
    won = bids >= market_price

    spend = float(np.sum(market_price[won]))
    impressions_won = int(np.sum(won))
    win_rate = float(np.mean(won))

    expected_treated_conversions = float(np.sum(p1_true[won]))
    expected_baseline_conversions = float(np.sum(p0_true[won]))
    expected_incremental_conversions = float(np.sum(tau_true[won]))

    out = {
        "policy": policy_name,
        "n_auctions": int(len(df)),
        "impressions_won": impressions_won,
        "win_rate": win_rate,
        "spend": spend,
        "expected_treated_conversions": expected_treated_conversions,
        "expected_baseline_conversions": expected_baseline_conversions,
        "expected_incremental_conversions": expected_incremental_conversions,
        "observed_cpa": spend / max(expected_treated_conversions, 1e-12),
        "incremental_cpa": spend / max(expected_incremental_conversions, 1e-12),
        "incremental_conv_per_1k": expected_incremental_conversions / max(spend, 1e-12) * 1000.0,
    }

    if sample_realized:
        y1 = rng.binomial(1, p1_true[won])
        y0 = rng.binomial(1, p0_true[won])

        out["realized_treated_conversions"] = int(np.sum(y1))
        out["realized_baseline_conversions"] = int(np.sum(y0))
        out["realized_incremental_conversions"] = int(np.sum(y1) - np.sum(y0))

    return out


def compare_policies(results: list[Dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(results)


@dataclass
class AlphaSearchResult:
    alpha: float
    spend: float
    rel_error: float
    iterations: int


def _simulate_spend_only(
    df: pd.DataFrame,
    bids: np.ndarray,
) -> float:
    market_price = df["market_price"].to_numpy(dtype=float)
    bids = np.asarray(bids, dtype=float)
    won = bids >= market_price
    return float(np.sum(market_price[won]))


def find_alpha_for_target_spend(
    df: pd.DataFrame,
    score: np.ndarray,
    bid_fn: Callable[[np.ndarray, BiddingPolicyConfig], np.ndarray],
    base_config: BiddingPolicyConfig,
    target_spend: float,
    alpha_low: float = 0.0,
    alpha_high: float = 1.0,
    max_iter: int = 40,
    rel_tol: float = 1e-3,
    expand_factor: float = 2.0,
    max_alpha_high: float = 1e6,
    verbose: bool = False,
) -> AlphaSearchResult:
    """
    Binary search alpha so policy spend matches target_spend.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain market_price.
    score : np.ndarray
        Model score used by bid_fn, e.g. p1_score or tau_score.
    bid_fn : callable
        Usually make_bid_cvr or make_bid_uplift.
    base_config : BiddingPolicyConfig
        value_per_conversion / bid_cap are taken from here.
        alpha from this config is ignored during search.
    target_spend : float
        Desired total spend.
    """
    if target_spend < 0:
        raise ValueError("target_spend must be non-negative.")
    if alpha_low < 0 or alpha_high < 0:
        raise ValueError("alpha bounds must be non-negative.")
    if alpha_high <= alpha_low:
        raise ValueError("alpha_high must be > alpha_low.")

    score = np.asarray(score, dtype=float)

    def spend_at(alpha: float) -> float:
        cfg = BiddingPolicyConfig(
            value_per_conversion=base_config.value_per_conversion,
            alpha=alpha,
            bid_cap=base_config.bid_cap,
        )
        bids = bid_fn(score, cfg)
        return _simulate_spend_only(df, bids)

    # Edge case: zero target spend
    if target_spend == 0:
        return AlphaSearchResult(alpha=0.0, spend=0.0, rel_error=0.0, iterations=0)

    # Expand upper bound until spend(alpha_high) >= target_spend
    spend_low = spend_at(alpha_low)
    spend_high = spend_at(alpha_high)

    while spend_high < target_spend and alpha_high < max_alpha_high:
        alpha_low = alpha_high
        spend_low = spend_high
        alpha_high *= expand_factor
        spend_high = spend_at(alpha_high)
        if verbose:
            print(f"[expand] alpha_high={alpha_high:.6f}, spend_high={spend_high:.6f}")

        # If bid cap prevents further spend growth, stop expanding
        if np.isclose(spend_high, spend_low, rtol=1e-10, atol=1e-10):
            break

    # If still below target, best achievable is alpha_high
    if spend_high < target_spend:
        print("WARNING: target spend not reachable due to bid cap / market limit")
        rel_error = abs(spend_high - target_spend) / max(target_spend, 1e-12)
        return AlphaSearchResult(
            alpha=alpha_high,
            spend=spend_high,
            rel_error=rel_error,
            iterations=0,
        )

    best_alpha = alpha_high
    best_spend = spend_high
    best_rel_error = abs(spend_high - target_spend) / max(target_spend, 1e-12)

    for it in range(1, max_iter + 1):
        alpha_mid = 0.5 * (alpha_low + alpha_high)
        spend_mid = spend_at(alpha_mid)
        rel_error = abs(spend_mid - target_spend) / max(target_spend, 1e-12)

        if rel_error < best_rel_error:
            best_alpha = alpha_mid
            best_spend = spend_mid
            best_rel_error = rel_error

        if verbose:
            print(
                f"[iter {it:02d}] alpha_low={alpha_low:.6f} "
                f"alpha_mid={alpha_mid:.6f} alpha_high={alpha_high:.6f} "
                f"spend_mid={spend_mid:.6f} rel_error={rel_error:.6f}"
            )

        if rel_error <= rel_tol:
            return AlphaSearchResult(
                alpha=alpha_mid,
                spend=spend_mid,
                rel_error=rel_error,
                iterations=it,
            )

        if spend_mid < target_spend:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid

    return AlphaSearchResult(
        alpha=best_alpha,
        spend=best_spend,
        rel_error=best_rel_error,
        iterations=max_iter,
    )


def simulate_policy_with_alpha_search(
    df: pd.DataFrame,
    score: np.ndarray,
    bid_fn: Callable[[np.ndarray, BiddingPolicyConfig], np.ndarray],
    base_config: BiddingPolicyConfig,
    target_spend: float,
    policy_name: str,
    sample_realized: bool = False,
    seed: int = 42,
    alpha_low: float = 0.0,
    alpha_high: float = 1.0,
    max_iter: int = 40,
    rel_tol: float = 1e-3,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Search alpha to match target spend, then simulate final outcomes.
    """
    search = find_alpha_for_target_spend(
        df=df,
        score=score,
        bid_fn=bid_fn,
        base_config=base_config,
        target_spend=target_spend,
        alpha_low=alpha_low,
        alpha_high=alpha_high,
        max_iter=max_iter,
        rel_tol=rel_tol,
        verbose=verbose,
    )

    final_cfg = BiddingPolicyConfig(
        value_per_conversion=base_config.value_per_conversion,
        alpha=search.alpha,
        bid_cap=base_config.bid_cap,
    )
    bids = bid_fn(score, final_cfg)

    result = simulate_policy_outcomes(
        df=df,
        bids=bids,
        policy_name=policy_name,
        sample_realized=sample_realized,
        seed=seed,
    )
    result["alpha"] = search.alpha
    result["target_spend"] = target_spend
    result["spend_rel_error"] = search.rel_error
    result["alpha_search_iterations"] = search.iterations
    return result