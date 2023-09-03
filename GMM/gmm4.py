import numpy as np
import matplotlib.pyplot as plt


class EM_GMM:
    def __init__(self, n_components, tol=1e-4, max_iter=100):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X):
        n_samples = X.shape[0]
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.weights = np.ones(self.n_components) / self.n_components
        self.covariances = np.array([np.identity(X.shape[1]) for _ in range(self.n_components)])
        log_likelihood = -np.inf
        for i in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)
            new_log_likelihood = self._compute_log_likelihood(X)
            if np.abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood
        return self

    def _e_step(self, X):
        return np.array([w * self._gaussian(X, m, c) for w, m, c in zip(self.weights, self.means, self.covariances)]).T

    def _m_step(self, X, resp):
        self.weights = resp.sum(axis=0) / resp.sum()
        self.means = np.dot(resp.T, X) / resp.sum(axis=0)[:, np.newaxis]
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(resp[:, k] * diff.T, diff) / resp[:, k].sum()

    def _gaussian(self, X, mean, cov):
        diff = X - mean
        return np.exp(-0.5 * np.sum(np.dot(diff, np.linalg.inv(cov)) * diff, axis=1)) / np.sqrt(np.linalg.det(cov))

    def _compute_log_likelihood(self, X):
        return np.sum(np.log(np.sum(self._e_step(X), axis=1)))


np.random.seed(0)
heights = np.concatenate([np.random.normal(160, 6, 60), np.random.normal(175, 6, 40)])[:, np.newaxis]
gmm = EM_GMM(n_components=2).fit(heights)
x = np.linspace(140, 200, 1000)[:, np.newaxis]
y = np.sum(gmm._e_step(x), axis=1)
plt.hist(heights, bins=20, density=True, alpha=0.6)
plt.plot(x, y)
plt.show()
