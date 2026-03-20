# src/run_budget_sweep.py

from __future__ import annotations

import numpy as np
import pandas as pd

from src.auction_data import simulate_auction_uplift_data
from src.budget_sweep import PolicySpec, run_budget_sweep
from src.plots import (
    plot_budget_sweep_incremental_frontier,
    plot_budget_sweep_efficiency,
    plot_budget_sweep_metric,
)
from src.uplift_bidding import (
    BiddingPolicyConfig,
    make_bid_cvr,
    make_bid_uplift,
)


def main():
    df = simulate_auction_uplift_data(n=100_000, seed=42)

    rng = np.random.default_rng(123)

    # first-pass noisy oracle scores
    p1_score = np.clip(
        df["p1_true"].to_numpy() + rng.normal(0, 0.01, len(df)),
        1e-4,
        1 - 1e-4,
    )
    tau_score = np.clip(
        df["tau_true"].to_numpy() + rng.normal(0, 0.01, len(df)),
        -0.05,
        0.25,
    )

    policies = [
        PolicySpec(
            name="CVR bidding",
            score=p1_score,
            bid_fn=make_bid_cvr,
            base_config=BiddingPolicyConfig(
                value_per_conversion=120.0,
                alpha=1.0,
                bid_cap=40.0,
            ),
        ),
        PolicySpec(
            name="Uplift bidding",
            score=tau_score,
            bid_fn=make_bid_uplift,
            base_config=BiddingPolicyConfig(
                value_per_conversion=120.0,
                alpha=1.0,
                bid_cap=40.0,
            ),
        ),
    ]

    target_spends = [50_000, 75_000, 100_000, 150_000, 200_000, 300_000, 500_000]

    sweep_df = run_budget_sweep(
        df=df,
        target_spends=target_spends,
        policies=policies,
        rel_tol=1e-4,
        max_iter=40,
        verbose=True,
    )

    print("\n=== Sweep results ===")
    print(
        sweep_df[
            [
                "policy",
                "target_spend_requested",
                "spend",
                "expected_treated_conversions",
                "expected_incremental_conversions",
                "observed_cpa",
                "incremental_cpa",
                "incremental_conv_per_1k",
                "alpha",
                "spend_rel_error",
            ]
        ]
        .sort_values(["target_spend_requested", "policy"])
        .to_string(index=False)
    )

    # Save plots
    plot_budget_sweep_incremental_frontier(
        sweep_df,
        save_path="images/budget_frontier_incremental_conversions.png",
        show=True,
    )

    plot_budget_sweep_efficiency(
        sweep_df,
        efficiency_col="incremental_conv_per_1k",
        save_path="images/budget_frontier_incremental_per_1k.png",
        show=True,
    )

    plot_budget_sweep_metric(
        sweep_df,
        metric="incremental_cpa",
        x_col="spend",
        title="Incremental CPA vs spend",
        save_path="images/budget_frontier_incremental_cpa.png",
        show=True,
    )

    plot_budget_sweep_metric(
        sweep_df,
        metric="observed_cpa",
        x_col="spend",
        title="Observed CPA vs spend",
        save_path="images/budget_frontier_observed_cpa.png",
        show=True,
    )

    plot_budget_sweep_metric(
        sweep_df,
        metric="expected_treated_conversions",
        x_col="spend",
        title="Gross treated conversions vs spend",
        save_path="images/budget_frontier_gross_conversions.png",
        show=True,
    )


if __name__ == "__main__":
    main()