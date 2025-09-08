from __future__ import annotations
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans

class RegimeHMM:
    """
    Stable GaussianHMM wrapper:
      - KMeans initialization for means/covars
      - Diagonal covariances (robust to collinearity)
      - Multiple restarts; keep best score
      - No invalid 'w' param (hmmlearn supports only s,t,m,c)
    """
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "diag",
        n_iter: int = 200,
        tol: float = 1e-3,
        random_state: int = 42,
        verbose: bool = False,
        algorithm: str = "viterbi",
        n_restarts: int = 3,
        var_floor: float = 1e-4,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.algorithm = algorithm
        self.n_restarts = max(1, n_restarts)
        self.var_floor = var_floor
        self.model = self._new_model(random_state)

    def _new_model(self, seed: int) -> GaussianHMM:
        return GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=seed,
            verbose=False,          # keep console clean
            algorithm=self.algorithm,
            init_params="",         # we'll set params ourselves
            params="stmc",          # <-- VALID: update start/trans/mu/cov via EM
        )

    def _init_kmeans(self, model: GaussianHMM, X: np.ndarray):
        # KMeans for means_; per-cluster diagonal covariances with variance floor
        kmeans = KMeans(n_init=10, random_state=model.random_state, n_clusters=model.n_components)
        labels = kmeans.fit_predict(X)
        model.means_ = kmeans.cluster_centers_

        n_features = X.shape[1]
        covars = np.empty((model.n_components, n_features), dtype=float)
        for k in range(model.n_components):
            Xk = X[labels == k]
            v = np.var(X if Xk.shape[0] < 2 else Xk, axis=0)
            covars[k] = np.maximum(v, self.var_floor)
        model.covars_ = covars

        # mild persistence in transitions; uniform start
        model.startprob_ = np.full(model.n_components, 1.0 / model.n_components)
        A = np.full((model.n_components, model.n_components), 0.2 / (model.n_components - 1))
        np.fill_diagonal(A, 0.8)
        model.transmat_ = A

    def fit(self, X: np.ndarray):
        best, best_score = None, -np.inf
        base_seed = int(self.random_state)
        for j in range(self.n_restarts):
            m = self._new_model(base_seed + j)
            self._init_kmeans(m, X)
            m.fit(X)  # EM
            try:
                sc = m.score(X)
            except Exception:
                sc = -np.inf
            if sc > best_score:
                best, best_score = m, sc
        self.model = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def posterior(self, X: np.ndarray) -> np.ndarray:
        _, post = self.model.score_samples(X)
        return post
