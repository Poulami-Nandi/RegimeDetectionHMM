from __future__ import annotations
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans


class RegimeHMM:
    """
    A stable, finance-friendly wrapper around hmmlearn's GaussianHMM.

    Why this wrapper exists
    -----------------------
    - **Deterministic & robust init:** We use KMeans to initialize state means
      and diagonal covariances. This avoids random and often poor initializations
      that can cause EM to converge to bad local optima.
    - **Diagonal covariance:** Markets features (e.g., returns, vol, momentum)
      can be collinear; diagonal covariances reduce overfitting and improve EM stability.
    - **Multiple restarts:** Fit several models with different seeds; pick the one
      with the best log-likelihood. This is a cheap way to avoid unlucky inits.
    - **Valid `params`:** hmmlearn only accepts "s", "t", "m", "c" (startprob,
      transmat, means, covars). We explicitly set that, avoiding accidental misuse.

    Typical usage
    -------------
    >>> X = feature_matrix  # shape (n_samples, n_features)
    >>> model = RegimeHMM(n_components=4, n_restarts=5).fit(X)
    >>> states = model.predict(X)         # Viterbi path (hard labels)
    >>> post   = model.posterior(X)       # p(state | observation) (soft labels)

    Notes on finance context
    ------------------------
    - `n_components` is the number of *regimes* (e.g., bull, bear, high-vol-bull, chop).
    - Inputs X should be standardized/cleaned features (e.g., returns, vol proxy, trend).
    - Posterior probabilities are often smoothed downstream and converted to
      "candidate/confirmed" states via rule thresholds.
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
        """
        Parameters
        ----------
        n_components : int
            Number of hidden states (regimes).
        covariance_type : {"diag", "full"}
            Covariance structure per state. "diag" is usually more stable
            for financial features and prevents singular covariance issues.
        n_iter : int
            Maximum EM iterations.
        tol : float
            Convergence tolerance on the EM lower bound.
        random_state : int
            Base seed for reproducibility. Each restart increments this seed.
        verbose : bool
            If True, pass verbose to hmmlearn (we keep it False by default to
            avoid chatty logs; use your own logging around this wrapper instead).
        algorithm : {"viterbi", "map"}
            Decoding algorithm for `predict`. Viterbi gives the most likely path.
        n_restarts : int
            Number of independent initializations; best-scoring model is kept.
        var_floor : float
            Minimum variance on each feature for diagonal covariances (stability).
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.algorithm = algorithm
        self.n_restarts = max(1, n_restarts)
        self.var_floor = var_floor

        # Build an untrained HMM with configured hyperparameters.
        # We do *not* let hmmlearn auto-initialize parameters; we set them ourselves.
        self.model = self._new_model(random_state)

    def _new_model(self, seed: int) -> GaussianHMM:
        """
        Create a fresh GaussianHMM instance with our preferred settings.
        We suppress hmmlearn's internal init by setting init_params="",
        then specify params="stmc" so EM updates start, trans, means, covars.
        """
        return GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=seed,
            verbose=False,          # keep console clean; rely on external logging if needed
            algorithm=self.algorithm,
            init_params="",         # <-- do not let hmmlearn random-init parameters
            params="stmc",          # <-- VALID: EM updates startprob, transmat, means, covars
        )

    def _init_kmeans(self, model: GaussianHMM, X: np.ndarray) -> None:
        """
        Initialize model parameters from data using KMeans clustering.

        Steps
        -----
        1) **means_**   := KMeans cluster centroids (good starting points).
        2) **covars_**  := per-cluster diagonal variances with floor (robust).
        3) **startprob_** uniform (uninformative prior).
        4) **transmat_** with mild persistence (higher diagonal mass), which speeds
           EM and reflects typical regime persistence in markets.

        Parameters
        ----------
        model : GaussianHMM
            The HMM instance to initialize.
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        """
        # --- cluster for initial means ---
        kmeans = KMeans(
            n_init=10,
            random_state=model.random_state,
            n_clusters=model.n_components
        )
        labels = kmeans.fit_predict(X)
        model.means_ = kmeans.cluster_centers_

        # --- diagonal covariances per state with variance floor ---
        n_features = X.shape[1]
        covars = np.empty((model.n_components, n_features), dtype=float)

        for k in range(model.n_components):
            # Select points assigned to cluster k; if too few, fallback to global variance.
            Xk = X[labels == k]
            # A state with <2 points can't yield a reliable variance; use the whole sample variance.
            v = np.var(X if Xk.shape[0] < 2 else Xk, axis=0)
            # Apply floor to avoid zero/singular covariances that can break EM.
            covars[k] = np.maximum(v, self.var_floor)

        model.covars_ = covars

        # --- start probabilities: uniform (we don't prefer any regime a priori) ---
        model.startprob_ = np.full(model.n_components, 1.0 / model.n_components)

        # --- transition matrix: mild persistence prior ---
        # put 0.8 on diagonals (stay), distribute 0.2 over off-diagonals (switch)
        A = np.full((model.n_components, model.n_components), 0.2 / (model.n_components - 1))
        np.fill_diagonal(A, 0.8)
        model.transmat_ = A

    def fit(self, X: np.ndarray) -> "RegimeHMM":
        """
        Fit the HMM to data with multiple restarts; keep the best log-likelihood.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features). For finance,
            columns might be daily return, vol proxy, trend gap, etc.

        Returns
        -------
        self : RegimeHMM
            The wrapper with `.model` set to the best-scoring GaussianHMM.

        Notes
        -----
        - Each restart re-seeds the HMM (`random_state + j`), re-initializes
          with KMeans, then runs EM.
        - We guard `score(X)` to handle any numerical hiccups; if scoring fails
          for one restart, it simply gets an -inf score and is ignored.
        """
        best, best_score = None, -np.inf
        base_seed = int(self.random_state)

        for j in range(self.n_restarts):
            # Fresh model per restart
            m = self._new_model(base_seed + j)
            # Deterministic, data-driven initialization
            self._init_kmeans(m, X)

            # EM training
            m.fit(X)

            # Evaluate model log-likelihood; higher is better
            try:
                sc = m.score(X)
            except Exception:
                sc = -np.inf

            # Track the best-scoring restart
            if sc > best_score:
                best, best_score = m, sc

        # Keep the champion
        self.model = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Decode the most likely sequence of hidden states for X (Viterbi path).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) with integer state labels in [0, n_components-1].

        Notes
        -----
        This is a *hard* labeling. For regime probabilities, use `posterior`.
        """
        return self.model.predict(X)

    def posterior(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the per-time posterior state probabilities: P(state | observation).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Matrix of shape (n_samples, n_components) where each row sums to ~1.

        Notes
        -----
        These *soft* labels are what you usually smooth (EMA) and threshold to derive
        "candidate" vs "confirmed" regimes in downstream logic.
        """
        _, post = self.model.score_samples(X)
        return post
