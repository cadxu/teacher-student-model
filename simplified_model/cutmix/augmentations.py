import numpy as np


def softmax(x, axis=None):
	x = np.asarray(x, dtype=float)
	x = x - np.max(x, axis=axis, keepdims=True)
	exp_x = np.exp(x)
	return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def WeakAugment(otu_matrix, noise_std=0.001, pseudocount=1e-8, seed=None):
	"""
	Augments a 2D OTU table (Samples x OTUs) by adding Gaussian noise in CLR space.

	Parameters:
	otu_matrix (numpy.ndarray): 2D array of raw counts or relative abundances.
								Rows are samples, columns are OTUs.
	noise_std (float): Standard deviation of the Gaussian noise.
	pseudocount (float): Small value to replace zeros before log transformation.
	seed (int): Random seed for reproducibility.

	Returns:
	numpy.ndarray: A 2D array of noisy relative abundances where each row sums to 1.
	"""
	# 1. Impute zeros
	mat = otu_matrix.astype(float)
	mat[mat == 0] = pseudocount

	p = mat / np.sum(mat, axis=1, keepdims=True)

	# 2. Centered Log-Ratio (CLR) Transformation
	# Calculate geometric mean for each row (sample)
	geom_means = np.exp(np.mean(np.log(p), axis=1, keepdims=True))

	# Log ratio of proportions against the geometric mean
	y = np.log(p / geom_means)

	# 3. Add Gaussian Noise
	# Draw independent noise for the entire matrix
	rng = np.random.default_rng(seed)
	noise = rng.normal(loc=0.0, scale=noise_std, size=y.shape)
	y_noisy = y + noise

	# 4. Inverse Transform (Softmax) back to the simplex
	# Applying softmax across the OTU axis (axis=1) closes the composition
	p_noisy = softmax(y_noisy, axis=1)

	return p_noisy



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

		idx1 = rng.choice(n, size=n_aug)
		idx2 = rng.choice(n, size=n_aug)
		p = rng.random(n_aug)
		mask = rng.binomial(1, p, [X_temp.shape[1], n_aug]).T
		X_aug = mask * X_temp[idx1, :] + (1 - mask) * X_temp[idx2, :]

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
		X_new[mask.astype("bool")] = 1
		X_aug = X_new

		X = np.concatenate([X, X_aug], axis=0)
		y = np.concatenate([y, np.repeat(val, n_aug)])
		w = np.concatenate([w, np.repeat(weight / (1 - weight) * X_train.shape[0] / n_aug, n_aug)])

	n = X.shape[0]
	idx = np.arange(n)
	rng.shuffle(idx)

	return X[idx], y[idx], w[idx]
