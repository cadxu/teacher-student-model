import numpy as np


def _normalize_rows(X):
    X = np.asarray(X, dtype=float)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return X / row_sums


def augmentTabular(X_unlab, y=None, noise_std=0.1, flip_cat_prob=0.05, seed=None):
    rng = np.random.default_rng(seed)
    X_unlab = np.asarray(X_unlab, dtype=float)
    std_devs = np.std(X_unlab, axis=0) + 1e-8
    X_aug = X_unlab + rng.normal(0, noise_std, size=X_unlab.shape) * std_devs
    # BUG FIX: noise breaks the row-sums-to-1 compositional structure that
    # the rest of the pipeline assumes - renormalize before returning.
    return _normalize_rows(X_aug)


def weakAugment(X_unlab, noise_std=0.05, seed=None):
    return augmentTabular(X_unlab, noise_std=noise_std, seed=seed)


def compositionalCutmix(X_train, y_train, factor=10, weight=0.5, seed=None):
    rng = np.random.default_rng(seed)
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X = X_train.copy()
    y = y_train.copy()
    w = np.ones_like(y, dtype=float)

    for val in np.unique(y_train):
        idxs = y_train == val
        X_temp = X_train[idxs, :]
        n = X_temp.shape[0]
        n_aug = int(factor * n) - n
        if n < 2 or n_aug <= 0:
            continue

        idx1 = rng.choice(n, size=n_aug)
        idx2 = rng.choice(n, size=n_aug)
        p = rng.random(n_aug)
        mask = rng.binomial(1, p, [X_temp.shape[1], n_aug]).T

        X_aug = mask * X_temp[idx1, :] + (1 - mask) * X_temp[idx2, :]
        # BUG FIX: a per-feature 0/1 mask pulls features from two different
        # rows, so the result does NOT generally sum to 1 even though both
        # source rows did. Without this, cutmix rows silently leave the
        # simplex while every other part of the pipeline assumes they're on it.
        X_aug = _normalize_rows(X_aug)

        X = np.concatenate([X, X_aug], axis=0)
        y = np.concatenate([y, np.repeat(val, n_aug)])
        w = np.concatenate([w, np.repeat(weight / (1 - weight) * X_train.shape[0] / n_aug, n_aug)])

    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    return X[idx], y[idx], w[idx]


def aitchisonMixup(X_train, y_train, factor=10, weight=0.5, seed=None):
    rng = np.random.default_rng(seed)
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X = X_train.copy()
    y = y_train.copy()
    w = np.ones_like(y, dtype=float)

    for val in np.unique(y_train):
        idxs = y_train == val
        X_temp = X_train[idxs, :]
        n = X_temp.shape[0]
        n_aug = int(factor * n) - n
        if n < 2 or n_aug <= 0:
            continue

        lam = rng.random(n_aug).reshape([-1, 1])
        idx1 = rng.choice(n, size=n_aug)
        idx2 = rng.choice(n, size=n_aug)
        X_aug = lam * X_temp[idx1, :] + (1 - lam) * X_temp[idx2, :]
        # Convex combos of normalized rows stay normalized in theory; guard
        # against floating-point drift so downstream row-sum assumptions hold.
        X_aug = _normalize_rows(X_aug)

        X = np.concatenate([X, X_aug], axis=0)
        y = np.concatenate([y, np.repeat(val, n_aug)])
        w = np.concatenate([w, np.repeat(weight / (1 - weight) * X_train.shape[0] / n_aug, n_aug)])

    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    return X[idx], y[idx], w[idx]


def compositionalFeatureDropout(X_train, y_train, factor=10, weight=0.5, seed=None):
    rng = np.random.default_rng(seed)
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X = X_train.copy()
    y = y_train.copy()
    w = np.ones_like(y, dtype=float)

    for val in np.unique(y_train):
        idxs = y_train == val
        X_temp = X_train[idxs, :]
        n = X_temp.shape[0]
        n_aug = int(factor * n) - n
        if n < 1 or n_aug <= 0:
            continue

        p = rng.random(n_aug)
        idx = rng.choice(n, size=n_aug)
        mask = rng.binomial(1, p, [X_temp.shape[1], n_aug]).T

        X_new = X_temp[idx, :].copy()
        # BUG FIX: dropping a feature should zero it out (simulate absence),
        # not set it to 1 - setting to 1 manufactures a huge artificial spike
        # in that feature instead of removing it.
        X_new[mask.astype("bool")] = 0.0
        # Zeroing features changes the row sum, so renormalize the remaining
        # features back onto the simplex.
        X_aug = _normalize_rows(X_new)

        X = np.concatenate([X, X_aug], axis=0)
        y = np.concatenate([y, np.repeat(val, n_aug)])
        w = np.concatenate([w, np.repeat(weight / (1 - weight) * X_train.shape[0] / n_aug, n_aug)])

    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    return X[idx], y[idx], w[idx]