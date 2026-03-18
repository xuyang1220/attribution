# src/plots.py

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _maybe_savefig(save_path: str | Path | None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=160)


def plot_tau_histogram(
    tau_true: np.ndarray,
    bins: int = 50,
    title: str = "True uplift distribution",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    tau_true = np.asarray(tau_true, dtype=float).reshape(-1)

    plt.figure(figsize=(8, 5))
    plt.hist(tau_true, bins=bins)
    plt.xlabel("True uplift (tau)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(alpha=0.25)

    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_pred_vs_true_uplift(
    tau_true: np.ndarray,
    tau_hat: np.ndarray,
    sample_size: int | None = 5000,
    seed: int = 42,
    title: str = "Predicted uplift vs true uplift",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    tau_true = np.asarray(tau_true, dtype=float).reshape(-1)
    tau_hat = np.asarray(tau_hat, dtype=float).reshape(-1)

    if len(tau_true) != len(tau_hat):
        raise ValueError("tau_true and tau_hat must have the same length.")

    if sample_size is not None and len(tau_true) > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(tau_true), size=sample_size, replace=False)
        x = tau_true[idx]
        y = tau_hat[idx]
    else:
        x = tau_true
        y = tau_hat

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.35, s=12)

    xy_min = float(min(np.min(x), np.min(y)))
    xy_max = float(max(np.max(x), np.max(y)))
    plt.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--")

    corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
    plt.xlabel("True uplift")
    plt.ylabel("Predicted uplift")
    plt.title(f"{title}\nCorr={corr:.4f}")
    plt.grid(alpha=0.25)

    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_uplift_curve(
    uplift_curve_df: pd.DataFrame,
    title: str = "Cumulative uplift curve",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    required_cols = {
        "frac",
        "mean_true_uplift_topk",
        "oracle_mean_true_uplift_topk",
        "random_mean_true_uplift",
    }
    missing = required_cols - set(uplift_curve_df.columns)
    if missing:
        raise ValueError(f"uplift_curve_df missing columns: {missing}")

    plt.figure(figsize=(8, 5))
    plt.plot(
        uplift_curve_df["frac"],
        uplift_curve_df["mean_true_uplift_topk"],
        label="Predicted ranking",
    )
    plt.plot(
        uplift_curve_df["frac"],
        uplift_curve_df["oracle_mean_true_uplift_topk"],
        label="Oracle ranking",
    )
    plt.plot(
        uplift_curve_df["frac"],
        uplift_curve_df["random_mean_true_uplift"],
        label="Random baseline",
        linestyle="--",
    )

    plt.xlabel("Top fraction treated")
    plt.ylabel("Mean true uplift in selected set")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)

    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_cumulative_gain_curve(
    uplift_curve_df: pd.DataFrame,
    title: str = "Cumulative gain curve",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    required_cols = {
        "frac",
        "cumulative_gain_pred",
        "cumulative_gain_oracle",
        "cumulative_gain_random",
    }
    missing = required_cols - set(uplift_curve_df.columns)
    if missing:
        raise ValueError(f"uplift_curve_df missing columns: {missing}")

    plt.figure(figsize=(8, 5))
    plt.plot(
        uplift_curve_df["frac"],
        uplift_curve_df["cumulative_gain_pred"],
        label="Predicted ranking",
    )
    plt.plot(
        uplift_curve_df["frac"],
        uplift_curve_df["cumulative_gain_oracle"],
        label="Oracle ranking",
    )
    plt.plot(
        uplift_curve_df["frac"],
        uplift_curve_df["cumulative_gain_random"],
        label="Random baseline",
        linestyle="--",
    )

    plt.xlabel("Top fraction treated")
    plt.ylabel("Cumulative true uplift")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)

    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_qini_curve(
    qini_df: pd.DataFrame,
    title: str = "Qini-like curve",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    required_cols = {
        "frac",
        "pred_incremental_conversions",
        "oracle_incremental_conversions",
        "random_incremental_conversions",
    }
    missing = required_cols - set(qini_df.columns)
    if missing:
        raise ValueError(f"qini_df missing columns: {missing}")

    plt.figure(figsize=(8, 5))
    plt.plot(
        qini_df["frac"],
        qini_df["pred_incremental_conversions"],
        label="Predicted ranking",
    )
    plt.plot(
        qini_df["frac"],
        qini_df["oracle_incremental_conversions"],
        label="Oracle ranking",
    )
    plt.plot(
        qini_df["frac"],
        qini_df["random_incremental_conversions"],
        label="Random baseline",
        linestyle="--",
    )

    plt.xlabel("Top fraction treated")
    plt.ylabel("Incremental conversions")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)

    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_uplift_calibration(
    calib_df: pd.DataFrame,
    title: str = "Uplift calibration by predicted-uplift bin",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    required_cols = {"tau_true_mean", "tau_hat_mean"}
    missing = required_cols - set(calib_df.columns)
    if missing:
        raise ValueError(f"calib_df missing columns: {missing}")

    x = calib_df["tau_hat_mean"].to_numpy(dtype=float)
    y = calib_df["tau_true_mean"].to_numpy(dtype=float)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=40)

    xy_min = float(min(np.min(x), np.min(y)))
    xy_max = float(max(np.max(x), np.max(y)))
    plt.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--")

    for i, (_, row) in enumerate(calib_df.iterrows()):
        plt.annotate(str(i), (row["tau_hat_mean"], row["tau_true_mean"]))

    plt.xlabel("Mean predicted uplift")
    plt.ylabel("Mean true uplift")
    plt.title(title)
    plt.grid(alpha=0.25)

    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_propensity_histogram(
    propensity_hat: np.ndarray,
    propensity_true: np.ndarray | None = None,
    bins: int = 40,
    title: str = "Propensity distribution",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    propensity_hat = np.asarray(propensity_hat, dtype=float).reshape(-1)

    plt.figure(figsize=(8, 5))
    plt.hist(propensity_hat, bins=bins, alpha=0.7, label="Predicted propensity")

    if propensity_true is not None:
        propensity_true = np.asarray(propensity_true, dtype=float).reshape(-1)
        if len(propensity_true) != len(propensity_hat):
            raise ValueError("propensity_true and propensity_hat must have the same length.")
        plt.hist(propensity_true, bins=bins, alpha=0.5, label="True propensity")

    plt.xlabel("Propensity")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)

    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(
    history: list[dict],
    title: str = "DragonNet training history",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    hist_df = pd.DataFrame(history)
    if hist_df.empty:
        raise ValueError("history is empty.")

    required_cols = {"epoch", "train_loss", "val_loss"}
    missing = required_cols - set(hist_df.columns)
    if missing:
        raise ValueError(f"history missing columns: {missing}")

    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="Train loss")
    plt.plot(hist_df["epoch"], hist_df["val_loss"], label="Val loss")

    if "train_outcome_loss" in hist_df.columns and "val_outcome_loss" in hist_df.columns:
        plt.plot(hist_df["epoch"], hist_df["train_outcome_loss"], label="Train outcome", linestyle="--")
        plt.plot(hist_df["epoch"], hist_df["val_outcome_loss"], label="Val outcome", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)

    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_uplift_diagnostics(
    tau_true: np.ndarray,
    tau_hat: np.ndarray,
    uplift_curve_df: pd.DataFrame,
    qini_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    history: list[dict] | None = None,
    propensity_hat: np.ndarray | None = None,
    propensity_true: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Convenience wrapper to generate the full diagnostic set.
    """
    output_dir = Path(output_dir) if output_dir is not None else None

    plot_tau_histogram(
        tau_true=tau_true,
        save_path=(output_dir / "tau_histogram.png") if output_dir else None,
        show=show,
    )
    plot_pred_vs_true_uplift(
        tau_true=tau_true,
        tau_hat=tau_hat,
        save_path=(output_dir / "pred_vs_true_uplift.png") if output_dir else None,
        show=show,
    )
    plot_uplift_curve(
        uplift_curve_df=uplift_curve_df,
        save_path=(output_dir / "uplift_curve.png") if output_dir else None,
        show=show,
    )
    plot_cumulative_gain_curve(
        uplift_curve_df=uplift_curve_df,
        save_path=(output_dir / "cumulative_gain_curve.png") if output_dir else None,
        show=show,
    )
    plot_qini_curve(
        qini_df=qini_df,
        save_path=(output_dir / "qini_curve.png") if output_dir else None,
        show=show,
    )
    plot_uplift_calibration(
        calib_df=calib_df,
        save_path=(output_dir / "uplift_calibration.png") if output_dir else None,
        show=show,
    )

    if propensity_hat is not None:
        plot_propensity_histogram(
            propensity_hat=propensity_hat,
            propensity_true=propensity_true,
            save_path=(output_dir / "propensity_histogram.png") if output_dir else None,
            show=show,
        )

    if history is not None:
        plot_training_history(
            history=history,
            save_path=(output_dir / "training_history.png") if output_dir else None,
            show=show,
        )


def plot_pred_vs_true_uplift_compare(
    tau_true: np.ndarray,
    pred_dict: dict[str, np.ndarray],
    sample_size: int | None = 5000,
    seed: int = 42,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    pred_dict example:
      {
        "DragonNet": tau_hat_dragon,
        "T-learner": tau_hat_tlearner,
      }
    """
    tau_true = np.asarray(tau_true, dtype=float).reshape(-1)

    plt.figure(figsize=(7, 6))

    rng = np.random.default_rng(seed)
    if sample_size is not None and len(tau_true) > sample_size:
        idx = rng.choice(len(tau_true), size=sample_size, replace=False)
    else:
        idx = np.arange(len(tau_true))

    xy_min = float(np.min(tau_true[idx]))
    xy_max = float(np.max(tau_true[idx]))

    for name, tau_hat in pred_dict.items():
        tau_hat = np.asarray(tau_hat, dtype=float).reshape(-1)
        if len(tau_hat) != len(tau_true):
            raise ValueError(f"{name}: tau_hat length mismatch.")
        corr = np.corrcoef(tau_true, tau_hat)[0, 1]
        plt.scatter(
            tau_true[idx],
            tau_hat[idx],
            alpha=0.25,
            s=12,
            label=f"{name} (corr={corr:.3f})",
        )
        xy_min = min(xy_min, float(np.min(tau_hat[idx])))
        xy_max = max(xy_max, float(np.max(tau_hat[idx])))

    plt.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--")
    plt.xlabel("True uplift")
    plt.ylabel("Predicted uplift")
    plt.title("Predicted uplift vs true uplift")
    plt.legend()
    plt.grid(alpha=0.25)

    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_uplift_curve_compare(
    curve_dict: dict[str, pd.DataFrame],
    title: str = "Cumulative uplift curve",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    curve_dict example:
      {
        "DragonNet": dragon_curve_df,
        "T-learner": t_curve_df,
      }
    """
    plt.figure(figsize=(8, 5))

    oracle_drawn = False
    random_drawn = False

    for name, df in curve_dict.items():
        plt.plot(df["frac"], df["mean_true_uplift_topk"], label=name)

        if not oracle_drawn:
            plt.plot(
                df["frac"],
                df["oracle_mean_true_uplift_topk"],
                label="Oracle ranking",
            )
            oracle_drawn = True

        if not random_drawn:
            plt.plot(
                df["frac"],
                df["random_mean_true_uplift"],
                linestyle="--",
                label="Random baseline",
            )
            random_drawn = True

    plt.xlabel("Top fraction treated")
    plt.ylabel("Mean true uplift in selected set")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)

    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_qini_curve_compare(
    qini_dict: dict[str, pd.DataFrame],
    title: str = "Qini-like curve",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(8, 5))

    oracle_drawn = False
    random_drawn = False

    for name, df in qini_dict.items():
        plt.plot(df["frac"], df["pred_incremental_conversions"], label=name)

        if not oracle_drawn:
            plt.plot(
                df["frac"],
                df["oracle_incremental_conversions"],
                label="Oracle ranking",
            )
            oracle_drawn = True

        if not random_drawn:
            plt.plot(
                df["frac"],
                df["random_incremental_conversions"],
                linestyle="--",
                label="Random baseline",
            )
            random_drawn = True

    plt.xlabel("Top fraction treated")
    plt.ylabel("Incremental conversions")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)

    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_calibration_compare(
    calib_dict: dict[str, pd.DataFrame],
    title: str = "Uplift calibration comparison",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(6, 6))

    xy_min = float("inf")
    xy_max = float("-inf")

    for name, df in calib_dict.items():
        x = df["tau_hat_mean"].to_numpy(dtype=float)
        y = df["tau_true_mean"].to_numpy(dtype=float)
        plt.plot(x, y, marker="o", label=name)

        xy_min = min(xy_min, float(np.min(x)), float(np.min(y)))
        xy_max = max(xy_max, float(np.max(x)), float(np.max(y)))

    plt.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--")
    plt.xlabel("Mean predicted uplift")
    plt.ylabel("Mean true uplift")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)

    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()