import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_sim import simulate_uplift_data
from src.dragonnet import TrainConfig, fit_dragonnet, predict_dragonnet
from src.baselines import TLearner, TClassifierConfig, naive_ate_difference_in_means
from src.evaluate import (
    summarize_uplift_eval,
    print_eval_summary,
    cumulative_uplift_curve,
    qini_curve_from_probs,
    calibration_by_uplift_bin,
)
from src.plots import (
    plot_uplift_curve_compare,
    plot_qini_curve_compare,
    plot_pred_vs_true_uplift_compare,
    plot_calibration_compare,
    plot_training_history,
)

# 1. simulate data
df = simulate_uplift_data(n=100_000, seed=42)

feature_cols = [
    "intent",
    "price_sens",
    "visits",
    "recency",
    "mobile",
    "geo_tier",
    "creative_match",
    "remarketing",
]

X = df[feature_cols].to_numpy(dtype=np.float32)
t = df["treatment"].to_numpy(dtype=np.float32)
y = df["outcome"].to_numpy(dtype=np.float32)
p0_true = df["p0_true"].to_numpy(dtype=np.float32)
p1_true = df["p1_true"].to_numpy(dtype=np.float32)
tau_true = df["tau_true"].to_numpy(dtype=np.float32)

(
    X_train,
    X_tmp,
    t_train,
    t_tmp,
    y_train,
    y_tmp,
    p0_train,
    p0_tmp,
    p1_train,
    p1_tmp,
    tau_train,
    tau_tmp,
) = train_test_split(
    X, t, y, p0_true, p1_true, tau_true,
    test_size=0.3,
    random_state=42,
)

(
    X_val,
    X_test,
    t_val,
    t_test,
    y_val,
    y_test,
    p0_val,
    p0_test,
    p1_val,
    p1_test,
    tau_val,
    tau_test,
) = train_test_split(
    X_tmp, t_tmp, y_tmp, p0_tmp, p1_tmp, tau_tmp,
    test_size=0.5,
    random_state=42,
)

# 2. scale for DragonNet
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 3. DragonNet
config = TrainConfig(
    batch_size=1024,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=20,
    alpha_propensity=0.5,
    grad_clip_norm=5.0,
    device="cpu", # "cuda" if torch.cuda.is_available() else "cpu",
    verbose=True,
)

dragon_model, history = fit_dragonnet(
    X_train=X_train_scaled,
    t_train=t_train,
    y_train=y_train,
    X_val=X_val_scaled,
    t_val=t_val,
    y_val=y_val,
    hidden_dim=128,
    num_shared_layers=2,
    num_head_layers=1,
    dropout=0.1,
    config=config,
)

dragon_pred = predict_dragonnet(
    model=dragon_model,
    X=X_test_scaled,
    batch_size=4096,
    device=config.device,
)
tau_hat_dragon = dragon_pred["tau_hat"]

# 4. T-learner
tlearner = TLearner(
    TClassifierConfig(
        model_type="lightgbm",
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
    )
)
tlearner.fit(X_train, t_train, y_train)
t_pred = tlearner.predict(X_test)
tau_hat_t = t_pred["tau_hat"]

# 5. naive ATE
naive_ate = naive_ate_difference_in_means(t_test, y_test)
print(f"Naive observed ATE: {naive_ate:.6f}")
print(f"True ATE on test  : {tau_test.mean():.6f}")

# 6. summaries
print("\n=== DragonNet ===")
dragon_summary = summarize_uplift_eval(
    tau_true=tau_test,
    tau_hat=tau_hat_dragon,
    p0_true=p0_test,
    p1_true=p1_test,
    frac=0.10,
)
print_eval_summary(dragon_summary)

print("\n=== T-learner ===")
t_summary = summarize_uplift_eval(
    tau_true=tau_test,
    tau_hat=tau_hat_t,
    p0_true=p0_test,
    p1_true=p1_test,
    frac=0.10,
)
print_eval_summary(t_summary)

# 7. curve dfs
dragon_curve_df = cumulative_uplift_curve(
    tau_true=tau_test,
    tau_hat=tau_hat_dragon,
    n_points=20,
)
t_curve_df = cumulative_uplift_curve(
    tau_true=tau_test,
    tau_hat=tau_hat_t,
    n_points=20,
)

dragon_qini_df = qini_curve_from_probs(
    p0_true=p0_test,
    p1_true=p1_test,
    tau_hat=tau_hat_dragon,
    n_points=20,
)
t_qini_df = qini_curve_from_probs(
    p0_true=p0_test,
    p1_true=p1_test,
    tau_hat=tau_hat_t,
    n_points=20,
)

dragon_calib_df = calibration_by_uplift_bin(
    tau_true=tau_test,
    tau_hat=tau_hat_dragon,
    n_bins=10,
)
t_calib_df = calibration_by_uplift_bin(
    tau_true=tau_test,
    tau_hat=tau_hat_t,
    n_bins=10,
)

# 8. side-by-side plots
plot_pred_vs_true_uplift_compare(
    tau_true=tau_test,
    pred_dict={
        "DragonNet": tau_hat_dragon,
        "T-learner": tau_hat_t,
    },
    save_path="images/pred_vs_true_compare.png",
    show=True,
)

plot_uplift_curve_compare(
    curve_dict={
        "DragonNet": dragon_curve_df,
        "T-learner": t_curve_df,
    },
    save_path="images/uplift_curve_compare.png",
    show=True,
)

plot_qini_curve_compare(
    qini_dict={
        "DragonNet": dragon_qini_df,
        "T-learner": t_qini_df,
    },
    save_path="images/qini_curve_compare.png",
    show=True,
)

plot_calibration_compare(
    calib_dict={
        "DragonNet": dragon_calib_df,
        "T-learner": t_calib_df,
    },
    save_path="images/calibration_compare.png",
    show=True,
)

plot_training_history(
    history=history,
    save_path="images/dragonnet_training_history.png",
    show=True,
)