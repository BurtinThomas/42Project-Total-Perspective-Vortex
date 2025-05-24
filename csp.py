import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y):
        if X.ndim != 3:
            raise ValueError(f"input data must have shape (n_trials, n_channels, n_samples), but got {X.shape}")
        
        self.n_channels_ = X.shape[1]
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("my CSP is only for binary classification.")
        
        covs = []
        for label in classes:
            X_class = X[y == label]
            cov = np.mean([np.cov(trial) for trial in X_class], axis=0)
            covs.append(cov)
        
        composite_cov = sum(covs)
        eigvals, eigvecs = eigh(covs[1], composite_cov)
        
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        
        self.filters_ = np.hstack([eigvecs[:, :self.n_components], eigvecs[:, -self.n_components:]])
        return self

    def transform(self, X):
        if not hasattr(self, "filters_"):
            raise RuntimeError("CSP instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        if X.ndim != 3:
            raise ValueError(f"input data must have shape (n_trials, n_channels, n_samples), but got {X.shape}.")
        
        if X.shape[1] != self.n_channels_:
            raise ValueError(f"mismatch in number of channels. Expected {self.n_channels_}, but got {X.shape[1]}.")
        
        X_projected = np.asarray([self.filters_.T @ trial for trial in X])
        
        log_variances = np.log(np.var(X_projected, axis=2))
        return log_variances
