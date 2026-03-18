# src/baselines.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class TClassifierConfig:
    model_type: str = "lightgbm"   # "lightgbm" or "logreg"
    random_state: int = 42

    # LightGBM params
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    # Logistic regression params
    C: float = 1.0
    max_iter: int = 1000


class TLearner:
    """
    T-learner for binary treatment / binary outcome.

    Fits:
      - one model on treated group to estimate p1(x)
      - one model on control group to estimate p0(x)
    """

    def __init__(self, config: TClassifierConfig | None = None) -> None:
        self.config = config or TClassifierConfig()
        self.model_t1 = None
        self.model_t0 = None

    def _make_model(self):
        if self.config.model_type == "lightgbm":
            return LGBMClassifier(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                num_leaves=self.config.num_leaves,
                min_child_samples=self.config.min_child_samples,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=self.config.random_state,
                verbose=-1,
            )

        if self.config.model_type == "logreg":
            return LogisticRegression(
                C=self.config.C,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
            )

        raise ValueError(f"Unsupported model_type: {self.config.model_type}")

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> "TLearner":
        X = np.asarray(X)
        treatment = np.asarray(treatment).reshape(-1)
        outcome = np.asarray(outcome).reshape(-1)

        if len(X) != len(treatment) or len(X) != len(outcome):
            raise ValueError("X, treatment, outcome must have same length.")

        mask_t1 = treatment == 1
        mask_t0 = treatment == 0

        if mask_t1.sum() == 0 or mask_t0.sum() == 0:
            raise ValueError("Both treated and control samples are required.")

        self.model_t1 = self._make_model()
        self.model_t0 = self._make_model()

        self.model_t1.fit(X[mask_t1], outcome[mask_t1])
        self.model_t0.fit(X[mask_t0], outcome[mask_t0])

        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if self.model_t1 is None or self.model_t0 is None:
            raise ValueError("TLearner is not fitted.")

        X = np.asarray(X)

        p1_hat = self.model_t1.predict_proba(X)[:, 1]
        p0_hat = self.model_t0.predict_proba(X)[:, 1]
        tau_hat = p1_hat - p0_hat

        return {
            "p0_hat": p0_hat,
            "p1_hat": p1_hat,
            "tau_hat": tau_hat,
        }


def naive_ate_difference_in_means(
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> float:
    treatment = np.asarray(treatment).reshape(-1)
    outcome = np.asarray(outcome).reshape(-1)

    if len(treatment) != len(outcome):
        raise ValueError("treatment and outcome must have same length.")

    mask_t1 = treatment == 1
    mask_t0 = treatment == 0

    if mask_t1.sum() == 0 or mask_t0.sum() == 0:
        raise ValueError("Both treated and control samples are required.")

    return float(outcome[mask_t1].mean() - outcome[mask_t0].mean())