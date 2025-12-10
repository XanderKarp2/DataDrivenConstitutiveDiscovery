# bindy.py
import numpy as np
from numpy.linalg import inv, slogdet


class BINDyRJ:
    """
    Bayesian Identification of Nonlinear Dynamics (BINDy-style)
    with:
      - RJMCMC bit-flip moves over model masks m \in {0,1}^D
      - Gaussian prior on coefficients xi
      - Inverse-gamma prior on noise sigma^2
      - Sparsity + complexity-aware model prior

    Model:
        Y = Theta @ xi + eps,  eps ~ N(0, sigma^2 I)
        xi ~ N(mu0, diag(Sigma0_diag))
        sigma^2 ~ IG(a0, b0)

    Theta: (N, D)
    Y    : (N,)
    """

    def __init__(
        self,
        Theta,
        Y,
        mu0=None,
        Sigma0_diag=None,
        term_complexity=None,
        a0=1e-3,
        b0=1e-3,
        model_prior="geo_complexity",
        theta_geom=0.95,
        complexity_weight=1.0,
        n_steps=6000,
        burn_in=1000,
        min_active=1,
        seed=0,
    ):
        """
        Parameters
        ----------
        Theta : ndarray, shape (N, D)
        Y     : ndarray, shape (N,)
        mu0   : (D,) prior mean for xi (default 0)
        Sigma0_diag : (D,) diag entries of prior covariance for xi.
            If None, we build a complexity-aware diagonal prior.
        term_complexity : (D,) nonnegative complexity measure
            (e.g. term.complexity from your TensorOp).
        a0, b0 : IG prior parameters for sigma^2.
        model_prior : {"flat", "geometric", "complexity", "geo_complexity"}
            - "flat": no preference
            - "geometric": p(m) ∝ (1-θ) θ^{#active}
            - "complexity": p(m) ∝ exp( - λ * sum_j m_j * complexity_j )
            - "geo_complexity": combination of both.
        theta_geom : geometric sparsity parameter (near 1 → sparser).
        complexity_weight : λ in the complexity penalty.
        n_steps : total RJMCMC steps.
        burn_in : discarded initial steps.
        min_active : minimum number of active terms in any model.
        seed : RNG seed.
        """
        self.Theta = np.asarray(Theta)
        self.Y = np.asarray(Y).flatten()
        self.N, self.D = self.Theta.shape

        self.n_steps = int(n_steps)
        self.burn_in = int(burn_in)
        self.min_active = int(min_active)

        # complexities
        if term_complexity is None:
            self.term_complexity = np.ones(self.D)
        else:
            self.term_complexity = np.asarray(term_complexity, dtype=float)

        # priors on xi
        if mu0 is None:
            self.mu0 = np.zeros(self.D)
        else:
            self.mu0 = np.asarray(mu0).flatten()

        if Sigma0_diag is None:
            # complexity-aware prior variance: more complex → stronger shrinkage
            # base variance ~ 1e3 scaled by (1 + λ * complexity)^(-2)
            base_var = 1e3
            self.Sigma0_diag = base_var / (1.0 + complexity_weight * self.term_complexity) ** 2
        else:
            self.Sigma0_diag = np.asarray(Sigma0_diag).flatten()

        self.a0 = float(a0)
        self.b0 = float(b0)

        self.model_prior = model_prior
        self.theta_geom = float(theta_geom)
        self.complexity_weight = float(complexity_weight)

        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Model prior log p(m)
    # ------------------------------------------------------------------
    def _log_p_model(self, m_mask):
        m_mask = np.asarray(m_mask, dtype=bool)
        d = int(np.sum(m_mask))
        if d < self.min_active:
            return -np.inf

        logp = 0.0
        if self.model_prior in ("geometric", "geo_complexity"):
            # geometric piece: p(m) ∝ (1 - θ) θ^d
            logp += np.log(1.0 - self.theta_geom) + d * np.log(self.theta_geom)

        if self.model_prior in ("complexity", "geo_complexity"):
            # complexity penalty: exp( - λ * Σ m_j * complexity_j )
            total_c = float(np.sum(self.term_complexity[m_mask]))
            logp += -self.complexity_weight * total_c

        if self.model_prior == "flat":
            logp = 0.0

        return logp

    # ------------------------------------------------------------------
    # Posterior for xi under given model m, sigma^2
    # ------------------------------------------------------------------
    def _posterior_params_for_model(self, m_mask, sigma2):
        m_mask = np.asarray(m_mask, dtype=bool)
        idx = np.where(m_mask)[0]
        d = len(idx)
        if d == 0:
            return None, None

        Theta_m = self.Theta[:, idx]                 # (N, d)
        Sigma0_m_diag = self.Sigma0_diag[idx]        # (d,)
        mu0_m = self.mu0[idx]                        # (d,)

        inv_Sigma0_m = np.diag(1.0 / Sigma0_m_diag)

        A = (Theta_m.T @ Theta_m) / sigma2 + inv_Sigma0_m
        Sigma_m = inv(A)

        rhs = inv_Sigma0_m @ mu0_m + (Theta_m.T @ self.Y) / sigma2
        mu_m = Sigma_m @ rhs

        return Sigma_m, mu_m

    # ------------------------------------------------------------------
    # Log "model evidence" like term for acceptance ratio
    # ------------------------------------------------------------------
    def _log_model_posterior_term(self, m_mask, sigma2):
        """
        Approximate term proportional to log p(Y | m, sigma^2) p(xi | m, sigma^2)
        up to constants that cancel in the RJMCMC ratio.

        We keep:
          -0.5 * log|Σ0_m| + 0.5 * log|Σ_m|
          +0.5 * μ_m^T Σ_m^{-1} μ_m
        """
        m_mask = np.asarray(m_mask, dtype=bool)
        idx = np.where(m_mask)[0]
        d = len(idx)

        if d == 0:
            return -1e9  # strongly discourage empty model

        Sigma_m, mu_m = self._posterior_params_for_model(m_mask, sigma2)
        if Sigma_m is None:
            return -np.inf

        Sigma0_m_diag = self.Sigma0_diag[idx]
        logdet_Sigma0 = np.sum(np.log(Sigma0_m_diag))

        sign_Sm, logdet_Sm = slogdet(Sigma_m)
        if sign_Sm <= 0:
            return -np.inf

        inv_Sigma_m = inv(Sigma_m)
        quad = float(mu_m.T @ inv_Sigma_m @ mu_m)

        return -0.5 * logdet_Sigma0 + 0.5 * logdet_Sm + 0.5 * quad

    # ------------------------------------------------------------------
    # Gibbs-like update for sigma^2
    # ------------------------------------------------------------------
    def _sample_sigma2(self, m_mask, sigma2_current):
        """
        Approximate IG posterior for sigma^2 using residuals at μ_m.
        """
        m_mask = np.asarray(m_mask, dtype=bool)
        idx = np.where(m_mask)[0]
        d = len(idx)

        if d == 0:
            rss = float(self.Y.T @ self.Y)
        else:
            Sigma_m, mu_m = self._posterior_params_for_model(m_mask, sigma2_current)
            Theta_m = self.Theta[:, idx]
            y_hat = Theta_m @ mu_m
            resid = self.Y - y_hat
            rss = float(resid.T @ resid)

        a_post = self.a0 + self.N / 2.0
        b_post = self.b0 + 0.5 * rss

        scale_post = 1.0 / b_post
        gamma_sample = self.rng.gamma(shape=a_post, scale=scale_post)
        sigma2_sample = 1.0 / gamma_sample
        return sigma2_sample

    # ------------------------------------------------------------------
    # Main RJMCMC loop
    # ------------------------------------------------------------------
    def sample(self, init_model="full", init_sigma2=1.0):
        """
        Run RJMCMC and return model masks and sigma^2 samples
        (after burn-in).

        init_model : "full", "empty", or boolean mask array.
        """
        if isinstance(init_model, str):
            if init_model == "full":
                m = np.ones(self.D, dtype=bool)
            elif init_model == "empty":
                m = np.zeros(self.D, dtype=bool)
                # ensure >= min_active
                true_indices = np.arange(self.D)
                self.rng.shuffle(true_indices)
                m[true_indices[: self.min_active]] = True
            else:
                raise ValueError("Unknown init_model string.")
        else:
            m = np.asarray(init_model, dtype=bool)
            if m.shape[0] != self.D:
                raise ValueError("init_model mask has wrong length")

        sigma2 = float(init_sigma2)

        masks = []
        sigma2s = []

        log_post_m = self._log_model_posterior_term(m, sigma2)
        log_p_m = self._log_p_model(m)

        for step in range(self.n_steps):
            # ---- RJMCMC bit-flip proposal ----
            k = self.rng.integers(0, self.D)
            m_prop = m.copy()
            m_prop[k] = ~m_prop[k]

            # enforce minimum active terms
            if np.sum(m_prop) < self.min_active:
                # skip proposal
                pass
            else:
                log_post_m_prop = self._log_model_posterior_term(m_prop, sigma2)
                log_p_m_prop = self._log_p_model(m_prop)

                log_alpha = (log_p_m_prop + log_post_m_prop) - (log_p_m + log_post_m)
                if np.log(self.rng.random()) < log_alpha:
                    m = m_prop
                    log_post_m = log_post_m_prop
                    log_p_m = log_p_m_prop

            # ---- update sigma^2 ----
            sigma2 = self._sample_sigma2(m, sigma2)

            # ---- save after burn-in ----
            if step >= self.burn_in:
                masks.append(m.copy())
                sigma2s.append(sigma2)

        return np.array(masks, dtype=bool), np.array(sigma2s, dtype=float)

    # ------------------------------------------------------------------
    # Posterior-mean xi from samples
    # ------------------------------------------------------------------
    def compute_posterior_mean_xi(self, masks, sigma2s):
        """
        For each sample (m, sigma^2), compute posterior mean μ_m
        embedded into full-length xi, then average over samples.
        """
        masks = np.asarray(masks, dtype=bool)
        sigma2s = np.asarray(sigma2s, dtype=float)
        n_samples, D = masks.shape
        assert D == self.D

        xi_samples = np.zeros((n_samples, D))

        for s in range(n_samples):
            m_mask = masks[s]
            sigma2 = sigma2s[s]
            idx = np.where(m_mask)[0]
            if len(idx) == 0:
                continue
            Sigma_m, mu_m = self._posterior_params_for_model(m_mask, sigma2)
            xi_s = np.zeros(D)
            xi_s[idx] = mu_m
            xi_samples[s] = xi_s

        return np.mean(xi_samples, axis=0)
