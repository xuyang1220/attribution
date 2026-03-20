from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_auction_uplift_data(n=100_000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    intent = rng.normal(0, 1, n)
    price_sens = rng.normal(0, 1, n)
    visits = np.clip(rng.poisson(2.0, n), 0, 20)
    recency = np.clip(rng.exponential(5.0, n), 0, 30)
    mobile = rng.binomial(1, 0.65, n)
    creative_match = rng.uniform(0, 1, n)
    remarketing = rng.binomial(1, sigmoid(-0.5 + 0.8 * np.log1p(visits) - 0.12 * recency + 0.4 * intent))

    logit_p0 = (
        -2.8
        + 1.2 * intent
        + 0.35 * np.log1p(visits)
        - 0.25 * np.log1p(recency)
        + 0.5 * remarketing
        - 0.35 * price_sens
        + 0.25 * creative_match
    )
    p0_true = sigmoid(logit_p0)

    tau_true = (
        0.01
        + 0.06 * remarketing
        + 0.04 * sigmoid(1.5 * intent)
        + 0.03 * creative_match
        - 0.03 * sigmoid(1.2 * price_sens)
        - 0.02 * (recency > 12).astype(float)
    )
    tau_true = np.clip(tau_true, -0.03, 0.20)
    p1_true = np.clip(p0_true + tau_true, 1e-4, 1 - 1e-4)

    # Market price correlated with intent / value / competition
    log_market = (
        1.5
        + 0.35 * intent
        + 0.15 * np.log1p(visits)
        + 0.25 * remarketing
        + 0.10 * creative_match
        + rng.normal(0, 0.5, n)
    )
    market_price = np.exp(log_market)

    df = pd.DataFrame({
        "intent": intent,
        "price_sens": price_sens,
        "visits": visits,
        "recency": recency,
        "mobile": mobile,
        "creative_match": creative_match,
        "remarketing": remarketing,
        "p0_true": p0_true,
        "p1_true": p1_true,
        "tau_true": tau_true,
        "market_price": market_price,
    })
    return df