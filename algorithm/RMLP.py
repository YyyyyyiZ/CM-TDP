import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import logistic, norm
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import rbf_kernel
import warnings
warnings.filterwarnings("ignore")


class RMLP:
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
            target_X: np.ndarray,
            target_p: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute offline-to-online pricing transfer

        Parameters:
        target_X: Target market features [n_samples, n_features]
        n0: Number of transfer learning episodes

        Returns:
        {
            'prices': Price decisions,
            'betas': Parameter estimates,
            'lambdas': Lambda values used
        }
        """

        # Initialize storage
        T = len(target_X)
        prices = np.zeros(T)
        betas = []
        lambdas = []
        collected_data = []

        # 1. Initial pricing (m=1)
        d = target_X.shape[1]
        collected_data.append((0, target_X[0], 0))

        # 2. Episode-based processing
        m = 2
        while True:
            prev_start_t = 2 ** (m - 2)
            prev_end_t = 2 ** (m - 1) - 1
            start_t = 2 ** (m - 1)
            end_t = min(2 ** m - 1, T)

            prev_data = collected_data[prev_start_t-1:prev_end_t]
            beta_init = np.zeros(d)

            res = minimize(
                fun=self._negative_log_likelihood,
                x0=beta_init,
                args=(
                    np.array([d[1] for d in prev_data]),
                    np.array([d[0] for d in prev_data]),
                    np.array([d[2] for d in prev_data])),
                method='L-BFGS-B',
                bounds=[(-self.W ,self.W)] * d,
                options={'maxiter': 1000}
            )

            if not res.success:
                raise ValueError(f"Optimization failed: {res.message}")

            beta_m = res.x

            betas.append(beta_m)

            # 3. Apply pricing
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



class RMLP_RKHS:
    def __init__(self, F_dist='logistic', noise_B=1.0, R=1.0,
                 kernel='rbf', gamma=1.0):
        """
        RKHS-based pricing algorithm without transfer learning

        Parameters:
        F_dist: Noise distribution type ('logistic' or 'normal')
        noise_B: Support boundary for noise [-B, B]
        R: Bound for RKHS norm of target function
        kernel: Kernel type ('rbf' or 'linear')
        gamma: Kernel parameter (for RBF kernel)
        """
        self.F_dist = F_dist
        self.noise_B = noise_B
        self.R = R
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
            target_X: np.ndarray,
            target_p: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute RKHS pricing without transfer learning

        Parameters:
        target_X: Target market features [n_samples, n_features]
        target_p: Target market prices [n_samples]

        Returns:
        {
            'prices': Price decisions,
            'alphas': Dual weights for each episode,
            'utilities': Predicted utilities
        }
        """
        T = len(target_X)
        prices = np.zeros(T)
        alphas = []
        utilities = np.zeros(T)
        collected_data = []

        # Initial state (m=1)
        prices[0] = 0  # Initial price
        alpha_m = np.zeros(1)  # Initial dual weight (single point)
        alphas.append(alpha_m.copy())
        collected_data.append((0, target_X[0], 0))

        # Episode-based processing
        m = 2
        while True:
            prev_start_t = 2 ** (m - 2)
            prev_end_t = 2 ** (m - 1) - 1
            start_t = 2 ** (m - 1)
            end_t = min(2 ** m - 1, T)

            # Get previous episode data
            prev_data = collected_data[prev_start_t - 1:prev_end_t]
            X_prev = np.array([d[1] for d in prev_data])
            p_prev = np.array([d[0] for d in prev_data])
            y_prev = np.array([d[2] for d in prev_data])

            # Compute kernel matrix for previous data
            K_prev = self.kernel_func(X_prev, X_prev)

            # Optimize dual weights (alpha) for current episode
            alpha_m = self._optimize_alpha(K_prev, p_prev, y_prev)
            alphas.append(alpha_m.copy())

            # Apply pricing for current episode
            for t in range(start_t - 1, end_t):
                # Compute utility using all historical data
                X_hist = np.array([d[1] for d in prev_data])
                K_t = self.kernel_func(target_X[t:t + 1], X_hist)
                utility = np.dot(K_t, alpha_m[:t])  # Use first t weights
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

    def _optimize_alpha(self, K, p, y):
        """Optimize dual weights using current episode data"""
        n = len(p)

        def loss(alpha):
            # Compute utilities
            utilities = np.dot(K, alpha)

            # Compute probabilities
            probs = self.F(p - utilities)

            # Compute log likelihood
            log_probs = np.where(y == 1,
                                 np.log(1 - probs + 1e-10),
                                 np.log(probs + 1e-10))
            likelihood_loss = -np.mean(log_probs)

            # RKHS norm regularization
            rkhs_reg = 0.5 * self.R ** 2 * np.dot(alpha, np.dot(K, alpha))

            return likelihood_loss + rkhs_reg

        res = minimize(
            loss,
            x0=np.zeros(n),
            method='L-BFGS-B',
            bounds=[(-self.R, self.R)] * n
        )
        return res.x
