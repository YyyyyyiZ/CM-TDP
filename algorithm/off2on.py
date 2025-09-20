import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import logistic, norm
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import rbf_kernel
import warnings
warnings.filterwarnings("ignore")


class Offline2Online:
    def __init__(self, F_dist='logistic', noise_B=1.0, W=1.0):
        """
        Offline-to-online cross-market pricing transfer algorithm

        Parameters:
        F_dist: Noise distribution type ('logistic' or 'normal')
        noise_B: Support boundary for noise [-B, B]
        W: Bound for feature/parameter magnitudes (for u_F calculation)
        """
        self.F_dist = F_dist
        self.noise_B = noise_B
        self.W = W
        self._init_F_functions()
        self.u_F = self._compute_uF()  # Precompute u_F constant

    def _init_F_functions(self):
        """Initialize distribution and pricing functions"""
        if self.F_dist == 'logistic':
            self.F = logistic.cdf
            self.F_pdf = logistic.pdf
            self.F_deriv = lambda x: self.F(x) * (1 - self.F(x))
        elif self.F_dist == 'normal':
            self.F = norm.cdf
            self.F_pdf = norm.pdf
            self.F_deriv = norm.pdf
        else:
            raise ValueError("Supported distributions: 'logistic' or 'normal'")

        self.h = lambda u: self._solve_h(u)

    def _solve_h(self, u):
        def phi(z):
            return z - (1 - self.F(z)) / (self.F_pdf(z) + 1e-10)

        res = minimize_scalar(
            lambda z: np.abs(phi(z) + u),
            bounds=(-self.noise_B, self.noise_B),
            method='bounded'
        )
        return u + res.x

    def _compute_uF(self):
        """Compute u_F = max{ log'F(-2W), -log'(1-F(2W)) }"""
        x1, x2 = -2 * self.W, 2 * self.W
        if self.F_dist == 'logistic':
            term1 = 1 - self.F(x1)  # For logistic: log'(F(x)) = 1-F(x)
            term2 = self.F(x2)  # For logistic: -log'(1-F(x)) = F(x)
        else:
            term1 = self.F_deriv(x1) / (self.F(x1) + 1e-10)
            term2 = self.F_deriv(x2) / (1 - self.F(x2) + 1e-10)
        return max(term1, term2)

    def _get_lambda(self, t: int, d: int) -> float:
        """Compute λ = 4u_F * sqrt(log(d)/t)"""
        return 4 * self.u_F * np.sqrt(np.log(d) / t)

    def fit(self,
            source_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
            target_X: np.ndarray,
            target_p: np.ndarray,
            n0: int = 200) -> Dict[str, np.ndarray]:
        """
        Execute offline-to-online pricing transfer

        Parameters:
        source_data: List of (p_k, X_k, y_k) tuples for K source markets
        target_X: Target market features [n_samples, n_features]
        n0: Number of transfer learning episodes

        Returns:
        {
            'prices': Price decisions,
            'betas': Parameter estimates,
            'lambdas': Lambda values used
        }
        """
        # 1. Aggregate source market data
        p_all = np.concatenate([d[0] for d in source_data])
        X_all = np.concatenate([d[1] for d in source_data])
        y_all = np.concatenate([d[2] for d in source_data])
        d = X_all.shape[1]

        # 2. Initial warm-up estimate (Eq.9)
        beta_init = np.zeros(d)

        res = minimize(
            fun=self._negative_log_likelihood,
            x0=beta_init,
            args=(X_all, p_all, y_all),
            method='L-BFGS-B',
            bounds=[(-self.W ,self.W)] * d,
            options={'maxiter': 1000}
        )

        if not res.success:
            raise ValueError(f"Optimization failed: {res.message}")
        beta_ag = res.x

        # Initialize storage
        T = len(target_X)
        prices = np.zeros(T)
        betas = []
        lambdas = []
        collected_data = []

        # 3. Initial pricing (m=1)
        prices[0] = self.h(np.dot(target_X[0], beta_ag))
        y_1 = (target_p[0] >= self.h(np.dot(target_X[0], beta_ag))).astype(int)
        collected_data.append((prices[0], target_X[0], y_1))

        # Episode-based processing
        m = 2
        while True:
            prev_start_t = 2 ** (m - 2)
            prev_end_t = 2 ** (m - 1) - 1
            start_t = 2 ** (m - 1)
            end_t = min(2 ** m - 1, T)


            # 4. Transfer learning phase (m <= n0)
            if 2**(m-1) <= n0:
                prev_data = collected_data[:1] if m < 2 else collected_data[prev_start_t-1:prev_end_t]

                # Compute lambda for current episode
                lambda_m = self._get_lambda(len(prev_data), d)
                lambdas.append(lambda_m)

                # Compute delta_m with L1 regularization (Eq.12)
                delta_m = self._compute_delta(beta_ag, prev_data, lambda_m)
                beta_m = beta_ag + delta_m

            # 5. Pure online phase (m > n0)
            else:
                prev_data = collected_data[prev_start_t-1:prev_end_t]
                beta_init = np.zeros(d)

                res = minimize(
                    fun=self._negative_log_likelihood,
                    x0=beta_init,
                    args=(np.array([d[1] for d in prev_data]),
                          np.array([d[0] for d in prev_data]),
                          np.array([d[2] for d in prev_data]),),
                    method='L-BFGS-B',
                    bounds=[(-self.W ,self.W)] * d,
                    options={'maxiter': 1000}
                )

                if not res.success:
                    raise ValueError(f"Optimization failed: {res.message}")
                beta_m = res.x
            betas.append(beta_m)

            # 6. Apply pricing
            for t in range(start_t-1, end_t):
                prices[t] = self.h(np.dot(target_X[t], beta_m))
                y_t = (target_p[t] >= self.h(np.dot(target_X[t], beta_m))).astype(int)
                collected_data.append((prices[t], target_X[t], y_t))

            if end_t == T:
                break
            m += 1

        return {
            'prices': prices,
            'betas': betas,
            'lambdas': lambdas
        }

    def _compute_delta(self,
                       beta_ag: np.ndarray,
                       episode_data: List[Tuple[float, np.ndarray, int]],
                       lambda_k: float) -> np.ndarray:
        """Compute delta_m with L1 regularization (Eq.12)"""

        def loss(delta):
            beta = beta_ag + delta
            loss_val = 0.0
            for p, x, y in episode_data:
                u = p - np.dot(x, beta)
                prob = self.F(u)
                epsilon = 1e-10
                loss_val -= y * np.log(max(1 - prob, epsilon)) + (1 - y) * np.log(max(prob, epsilon))
            return loss_val / len(episode_data) + lambda_k * np.linalg.norm(delta, 1)

        res = minimize(
            loss,
            x0=np.zeros_like(beta_ag),
            method='L-BFGS-B',
            bounds=[(-self.W, self.W)] * len(beta_ag)
        )
        return res.x

    def _negative_log_likelihood(self, beta, X, p, y):
        """Negative loglikelihood (Eq.10)"""
        utility = p - X @ beta
        prob1 = 1 - self.F(utility)  # P(y=1)
        prob0 = self.F(utility)  # P(y=0)

        epsilon = 1e-10
        loss = -np.sum(
            (y == 1) * np.log(np.maximum(prob1, epsilon)) +
            (y == 0) * np.log(np.maximum(prob0, epsilon))
        )
        return loss / len(y)



class Offline2OnlineRKHS:
    def __init__(self, F_dist='logistic', noise_B=1.0, R=1.0, h=1.0,
                 kernel='rbf', gamma=1.0, eta=1.0):
        """
        Offline-to-online cross-market pricing transfer algorithm with RKHS utility model

        Parameters:
        F_dist: Noise distribution type ('logistic' or 'normal')
        noise_B: Support boundary for noise [-B, B]
        R: Bound for RKHS norm of target function
        h: Bound for task similarity (Assumption 7)
        kernel: Kernel type ('rbf' or 'linear')
        gamma: Kernel parameter (for RBF kernel)
        eta: Smoothness parameter for task similarity
        """
        self.F_dist = F_dist
        self.noise_B = noise_B
        self.R = R
        self.h = h
        self.eta = eta
        self.kernel = kernel
        self.gamma = gamma
        self._init_F_functions()
        self._init_kernel()

    def _init_F_functions(self):
        """Initialize distribution and pricing functions"""
        if self.F_dist == 'logistic':
            self.F = logistic.cdf
            self.F_pdf = logistic.pdf
            self.F_deriv = lambda x: self.F(x) * (1 - self.F(x))
        elif self.F_dist == 'normal':
            self.F = norm.cdf
            self.F_pdf = norm.pdf
            self.F_deriv = norm.pdf
        else:
            raise ValueError("Supported distributions: 'logistic' or 'normal'")

        self.h_func = lambda u: self._solve_h(u)

    def _init_kernel(self):
        """Initialize kernel function"""
        if self.kernel == 'rbf':
            self.kernel_func = lambda X1, X2: rbf_kernel(X1, X2, gamma=self.gamma)
        elif self.kernel == 'linear':
            self.kernel_func = lambda X1, X2: np.dot(X1, X2.T)
        else:
            raise ValueError("Supported kernels: 'rbf' or 'linear'")

    def _solve_h(self, u):
        """Solve for h(u) = u + φ^{-1}(-u)"""

        def phi(z):
            return z - (1 - self.F(z)) / (self.F_pdf(z) + 1e-10)

        res = minimize_scalar(
            lambda z: np.abs(phi(z) + u),
            bounds=(-self.noise_B, self.noise_B),
            method='bounded'
        )
        return u + res.x

    def fit(self,
            source_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
            target_X: np.ndarray,
            target_p: np.ndarray,
            n0: int = 200) -> Dict[str, np.ndarray]:
        """
        Execute RKHS offline-to-online pricing transfer

        Parameters:
        source_data: List of (p_k, X_k, y_k) tuples for K source markets
        target_X: Target market features [n_samples, n_features]
        target_p: Target market prices [n_samples]
        n0: Number of transfer learning episodes

        Returns:
        {
            'prices': Price decisions,
            'alphas': Dual weights for each episode,
            'utilities': Predicted utilities
        }
        """
        # 1. Aggregate source market data
        p_source = np.concatenate([d[0] for d in source_data])
        X_source = np.concatenate([d[1] for d in source_data])
        y_source = np.concatenate([d[2] for d in source_data])

        # 2. Initial warm-up estimate using source data
        K_source = self.kernel_func(X_source, X_source)
        alpha_ag = self._aggregate_source_estimate(K_source, p_source, y_source)

        # Initialize storage
        T = len(target_X)
        prices = np.zeros(T)
        alphas = []
        utilities = np.zeros(T)
        collected_data = []

        # 3. Initial pricing (m=1)
        K_init = self.kernel_func(target_X[0:1], X_source)
        utility = np.dot(K_init, alpha_ag)
        prices[0] = self.h_func(utility)
        utilities[0] = utility
        y_1 = (target_p[0] >= prices[0]).astype(int)
        collected_data.append((prices[0], target_X[0], y_1))
        alphas.append(alpha_ag.copy())

        # Episode-based processing
        m = 2
        while True:
            prev_start_t = 2 ** (m - 2)
            prev_end_t = 2 ** (m - 1) - 1
            start_t = 2 ** (m - 1)
            end_t = min(2 ** m - 1, T)

            prev_data = collected_data[:1] if m < 2 else collected_data[prev_start_t - 1:prev_end_t]
            X_prev = np.array([d[1] for d in prev_data])
            p_prev = np.array([d[0] for d in prev_data])
            y_prev = np.array([d[2] for d in prev_data])

            # 4. Transfer learning phase (m <= n0)
            if 2 ** (m - 1) <= n0:
                # Compute combined kernel matrix
                X_combined = np.vstack([X_source, X_prev])
                K_combined = self.kernel_func(X_combined, X_combined)

                # Compute regularization parameters
                lambda1, lambda2 = self._get_lambda(len(X_source), len(X_prev))

                # Optimize dual weights with transfer learning
                alpha_m = self._compute_transfer_estimate(
                    alpha_ag, K_combined,
                    p_source, y_source, X_source,
                    p_prev, y_prev, X_prev,
                    lambda1, lambda2
                )

            # 5. Pure online phase (m > n0)
            else:
                # Only use target market data
                K_target = self.kernel_func(X_prev, X_prev)
                alpha_m = self._optimize_alpha(K_target, p_prev, y_prev)

            alphas.append(alpha_m.copy())

            # 6. Apply pricing for current episode
            for t in range(start_t - 1, end_t):
                # Compute utility using all relevant data
                if 2 ** (m - 1) <= n0:
                    X_hist = np.vstack([X_source, np.array([d[1] for d in prev_data])])
                else:
                    X_hist = np.array([d[1] for d in prev_data])

                K_t = self.kernel_func(target_X[t:t + 1], X_hist)
                utility = np.dot(K_t, alpha_m[:len(X_hist)])
                utilities[t] = utility

                # Set price and observe response
                prices[t] = self.h_func(utility)
                y_t = (target_p[t] >= prices[t]).astype(int)
                collected_data.append((prices[t], target_X[t], y_t))

            if end_t == T:
                break
            m += 1

        return {
            'prices': prices,
            'alphas': alphas,
            'utilities': utilities
        }

    def _aggregate_source_estimate(self, K_source, p_source, y_source):
        """Compute initial RKHS estimate from source markets"""
        n = len(p_source)

        def loss(alpha):
            utilities = np.dot(K_source, alpha)
            probs = self.F(p_source - utilities)
            log_probs = np.where(y_source == 1,
                                 np.log(1 - probs + 1e-10),
                                 np.log(probs + 1e-10))
            return -np.mean(log_probs) + 0.5 * self.R ** 2 * np.dot(alpha, np.dot(K_source, alpha))

        res = minimize(
            loss,
            x0=np.zeros(n),
            method='L-BFGS-B',
            bounds=[(-self.R, self.R)] * n
        )
        return res.x

    def _compute_transfer_estimate(self, alpha_ag, K_combined,
                                   p_source, y_source, X_source,
                                   p_target, y_target, X_target,
                                   lambda1, lambda2):
        """Compute corrected RKHS estimate with transfer learning"""
        n_source = len(p_source)
        n_target = len(p_target)

        def loss(alpha):
            # Source term
            source_utilities = np.dot(K_combined[:n_source, :n_source], alpha[:n_source])  # Only use source portion
            source_probs = self.F(p_source - source_utilities)
            source_loss = -np.mean(np.where(y_source == 1,
                                            np.log(1 - source_probs + 1e-10),
                                            np.log(source_probs + 1e-10)))

            # Target term
            target_utilities = np.dot(K_combined[n_source:], alpha)  # Use full alpha for target
            target_probs = self.F(p_target - target_utilities)
            target_loss = -np.mean(np.where(y_target == 1,
                                            np.log(1 - target_probs + 1e-10),
                                            np.log(target_probs + 1e-10)))

            # RKHS norm regularization (on full alpha)
            rkhs_reg = 0.5 * lambda1 * np.dot(alpha, np.dot(K_combined, alpha))

            # Task similarity regularization (only on overlapping target portion)
            min_len = min(len(alpha_ag), len(alpha[n_source:]))
            task_sim_reg = lambda2 * np.linalg.norm(alpha[n_source:n_source + min_len] - alpha_ag[:min_len], 1)

            return source_loss + target_loss + rkhs_reg + task_sim_reg

        # Initialize with alpha_ag for source portion and zeros for target portion
        alpha_init = np.concatenate([alpha_ag[:n_source], np.zeros(n_target)])

        res = minimize(
            loss,
            x0=alpha_init,
            method='L-BFGS-B',
            bounds=[(-self.R, self.R)] * (n_source + n_target)
        )
        return res.x

    def _optimize_alpha(self, K, p, y):
        """Optimize dual weights using only target market data"""
        n = len(p)

        def loss(alpha):
            utilities = np.dot(K, alpha)
            probs = self.F(p - utilities)
            log_probs = np.where(y == 1,
                                 np.log(1 - probs + 1e-10),
                                 np.log(probs + 1e-10))
            return -np.mean(log_probs) + 0.5 * self.R ** 2 * np.dot(alpha, np.dot(K, alpha))

        res = minimize(
            loss,
            x0=np.zeros(n),
            method='L-BFGS-B',
            bounds=[(-self.R, self.R)] * n
        )
        return res.x

    def _get_lambda(self, n_source, n_target, alpha=0.5):
        """Compute regularization parameters λ1 and λ2"""
        lambda1 = (n_source * self.R ** 2) ** (-2 * alpha / (2 * alpha + 1))
        lambda2 = (n_target * self.h) ** (-2 * alpha / (2 * alpha + 1))
        return lambda1, lambda2
