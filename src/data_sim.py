import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_uplift_data(n=100_000, seed=42):
    rng = np.random.default_rng(seed)

    intent = rng.normal(0, 1, n)
    price_sens = rng.normal(0, 1, n)
    visits = np.clip(rng.poisson(2.0, n), 0, 20)
    recency = np.clip(rng.exponential(5.0, n), 0, 30)
    mobile = rng.binomial(1, 0.65, n)
    geo_tier = rng.integers(0, 3, n)
    creative_match = rng.uniform(0, 1, n)

    remarketing_logit = -0.5 + 0.8 * np.log1p(visits) - 0.12 * recency + 0.4 * intent
    remarketing = rng.binomial(1, sigmoid(remarketing_logit))

    X = pd.DataFrame({
        "intent": intent,
        "price_sens": price_sens,
        "visits": visits,
        "recency": recency,
        "mobile": mobile,
        "geo_tier": geo_tier,
        "creative_match": creative_match,
        "remarketing": remarketing,
    })

    # untreated conversion
    logit_p0 = (
        -2.8
        + 1.2 * intent
        + 0.35 * np.log1p(visits)
        - 0.25 * np.log1p(recency)
        + 0.5 * remarketing
        - 0.35 * price_sens
        + 0.25 * creative_match
        + 0.1 * (geo_tier == 2)
    )
    p0 = sigmoid(logit_p0)

    # heterogeneous treatment effect on probability scale
    tau = (
        0.01
        + 0.06 * remarketing
        + 0.04 * sigmoid(1.5 * intent)
        + 0.03 * creative_match
        - 0.03 * sigmoid(1.2 * price_sens)
        - 0.02 * (recency > 12).astype(float)
    )
    tau = np.clip(tau, -0.03, 0.20)
    p1 = np.clip(p0 + tau, 1e-4, 1 - 1e-4)

    # biased treatment assignment
    logit_e = (
        -0.3
        + 1.0 * remarketing
        + 0.8 * intent
        + 0.4 * np.log1p(visits)
        - 0.15 * recency
        + 0.25 * mobile
    )
    e = sigmoid(logit_e)
    t = rng.binomial(1, e)

    y0 = rng.binomial(1, p0)
    y1 = rng.binomial(1, p1)
    y = np.where(t == 1, y1, y0)

    df = X.copy()
    df["treatment"] = t
    df["outcome"] = y
    df["p0_true"] = p0
    df["p1_true"] = p1
    df["tau_true"] = p1 - p0
    df["propensity_true"] = e

    return df