from __future__ import annotations

import numpy as np
import pandas as pd

from src.auction_data import simulate_auction_uplift_data
from src.uplift_bidding import (
    BiddingPolicyConfig,
    make_bid_cvr,
    make_bid_uplift,
    simulate_policy_outcomes,
    simulate_policy_with_alpha_search,
    compare_policies,
)

def plot_policy_comparison(summary_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    metrics = [
        "spend",
        "expected_treated_conversions",
        "expected_incremental_conversions",
        "incremental_cpa",
    ]

    for metric in metrics:
        plt.figure(figsize=(6, 4))
        plt.bar(summary_df["policy"], summary_df[metric])
        plt.title(metric)
        plt.grid(alpha=0.25)
        plt.show()

def main():
    df = simulate_auction_uplift_data(n=100_000, seed=42)

    # First-pass: oracle-like scores with mild noise
    rng = np.random.default_rng(123)

    p1_score = np.clip(df["p1_true"].to_numpy() + rng.normal(0, 0.01, len(df)), 1e-4, 1 - 1e-4)
    tau_score = np.clip(df["tau_true"].to_numpy() + rng.normal(0, 0.01, len(df)), -0.05, 0.25)

    # Tune alpha so spend is in roughly comparable range
    cvr_cfg = BiddingPolicyConfig(value_per_conversion=120.0, alpha=1.0, bid_cap=40.0)
    uplift_cfg = BiddingPolicyConfig(value_per_conversion=120.0, alpha=1.8, bid_cap=40.0)

    bids_cvr = make_bid_cvr(p1_score, cvr_cfg)
    bids_uplift = make_bid_uplift(tau_score, uplift_cfg)

    res_cvr = simulate_policy_with_alpha_search(df, p1_score, make_bid_cvr, cvr_cfg, target_spend=100_000, policy_name="CVR bidding")
    res_uplift = simulate_policy_with_alpha_search(df, tau_score, make_bid_uplift, uplift_cfg, target_spend=100_000, policy_name="Uplift bidding")

    summary = compare_policies([res_cvr, res_uplift])

    # plot_policy_comparison(summary)
    print(summary.T)


if __name__ == "__main__":
    main()