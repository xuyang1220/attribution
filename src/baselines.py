from lightgbm import LGBMClassifier


def fit_t_learner(X_train, t_train, y_train):
    model_t1 = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    model_t0 = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )

    model_t1.fit(X_train[t_train == 1], y_train[t_train == 1])
    model_t0.fit(X_train[t_train == 0], y_train[t_train == 0])

    return model_t1, model_t0


def predict_t_learner(models, X):
    model_t1, model_t0 = models
    p1_hat = model_t1.predict_proba(X)[:, 1]
    p0_hat = model_t0.predict_proba(X)[:, 1]
    tau_hat = p1_hat - p0_hat
    return p0_hat, p1_hat, tau_hat