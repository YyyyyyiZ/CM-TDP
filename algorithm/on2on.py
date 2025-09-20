import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import logistic, norm
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import rbf_kernel
from functools import partial

class Online2Online:
    def __init__(self, F_dist='logistic', noise_B=1.0, W=1.0):
        """
        Online-to-online cross-market pricing transfer algorithm

        Parameters:
        F_dist: Noise distribution type ('logistic' or 'normal')
        noise_B: Support boundary for noise [-B, B]
        W: Bound for feature and parameter magnitudes (used for u_F calculation)
        """
        self.F_dist = F_dist
        self.noise_B = noise_B
        self.W = W
        self._init_F_functions()
        self.u_F = self._compute_uF()  # Precompute u_F value

    def _init_F_functions(self):
        """Initialize distribution and pricing functions"""
        if self.F_dist == 'logistic':
            self.F = logistic.cdf
            self.F_pdf = logistic.pdf
            self.F_deriv = lambda x: self.F(x) * (1 - self.F(x))  # Derivative of logistic
        elif self.F_dist == 'normal':
            self.F = norm.cdf
            self.F_pdf = norm.pdf
            self.F_deriv = norm.pdf
        else:
            raise ValueError("Unsupported distribution. Choose 'logistic' or 'normal'")

        # Predefine h-function solver
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
        x1 = -2 * self.W
        x2 = 2 * self.W

        if self.F_dist == 'logistic':
            # Simplified calculation for logistic derivatives
            term1 = 1 - self.F(x1)  # log'(F(x)) = F'(x)/F(x) = 1-F(x) for logistic
            term2 = self.F(x2)  # -log'(1-F(x)) = F'(x)/(1-F(x)) = F(x)
        else:
            # Normal distribution requires explicit calculation
            term1 = self.F_deriv(x1) / (self.F(x1) + 1e-10)
            term2 = self.F_deriv(x2) / (1 - self.F(x2) + 1e-10)

        return max(term1, term2)

    def _get_lambda_k(self, tau_prev: int, d: int) -> float:
        """Compute dynamic regularization parameter λ_k = 4u_F * sqrt(log(d)/τ_{k-1})"""
        return 4 * self.u_F * np.sqrt(np.log(d) / tau_prev)

    def fit(self,
            source_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
            target_X: np.ndarray,
            target_p: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute online transfer pricing algorithm

        Parameters:
        source_data: Function generating source market data for each episode
        target_X: Target market features [n_samples, n_features]
        max_episodes: Maximum number of episodes

        Returns:
        {
            'prices': Price decision sequence,
            'betas': Parameter estimates for each episode,
            'lambdas': Sequence of λ_k values used
        }
        """
        T = len(target_X)
        d = target_X.shape[1]
        prices = np.zeros(T)
        betas = []
        lambdas = []
        collected_data = []


        # Initial state (m=1)
        prices[0] = 0  # Initial price
        beta_m = 0
        betas.append(beta_m)
        collected_data.append((0, target_X[0], 1))

        # Episode-based processing
        m = 2
        while True:
            prev_start_t = 2 ** (m - 2)
            prev_end_t = 2 ** (m - 1) - 1
            start_t = 2 ** (m - 1)
            end_t = min(2 ** m - 1, T)

            # 1. Aggregate source market data
            p_all = np.concatenate([d[0][prev_start_t-1:prev_end_t] for d in source_data])
            X_all = np.concatenate([d[1][prev_start_t-1:prev_end_t] for d in source_data])
            y_all = np.concatenate([d[2][prev_start_t-1:prev_end_t] for d in source_data])

            # 2. Compute initial estimate β_m^(ag)
            beta_ag = self._aggregate_source_estimate(p_all, X_all, y_all)

            # 3. Compute dynamic λ_k (τ_{k-1} = 2^{m-2})
            tau_prev = 2 ** (m - 2)
            lambda_k = self._get_lambda_k(tau_prev, d)
            lambdas.append(lambda_k)

            # 4. Compute δ_m with L1 regularization
            prev_data = collected_data[:1] if m < 2 else collected_data[prev_start_t-1:prev_end_t]
            delta_m = self._compute_delta_m(beta_ag, prev_data, lambda_k)
            beta_m = beta_ag + delta_m
            betas.append(beta_m)

            # 5. Apply pricing for current episode
            for t in range(start_t-1, end_t):
                prices[t] = self.h(np.dot(target_X[t], beta_m))
                y_t =  (target_p[t] >= self.h(np.dot(target_X[t], beta_m))).astype(int)
                collected_data.append((prices[t], target_X[t], y_t))

            m += 1
            if end_t == T:
                break

        return {
            'prices': prices,
            'betas': betas,
            'lambdas': lambdas
        }

    def _aggregate_source_estimate(self, p_all: np.ndarray, X_all: np.ndarray, y_all: np.ndarray) -> np.ndarray:
        """Compute aggregated source market estimate (Eq.14), with fallback for single-class y by modifying p"""
        d = X_all.shape[1]
        beta_init = np.zeros(d)

        res = minimize(
            fun=self._negative_log_likelihood,
            x0=beta_init,
            args=(X_all, p_all, y_all),
            method='L-BFGS-B',
            bounds=[(-self.W, self.W)] * d,
            options={'maxiter': 1000}
        )

        if not res.success:
            raise ValueError(f"Optimization failed: {res.message}")
        return res.x

    def _compute_delta_m(self,
                         beta_ag: np.ndarray,
                         episode_data: List[Tuple[float, np.ndarray, int]],
                         lambda_k: float) -> np.ndarray:
        """Compute δ_m with L1 regularization (Eq.17)"""

        def loss(delta):
            beta = beta_ag + delta
            loss_val = 0.0
            for p, x, y in episode_data:
                u = p - np.dot(x, beta)
                prob = self.F(u)
                epsilon = 1e-10
                if y == 1:
                    loss_val -= np.log(max(1 - prob, epsilon))
                else:
                    loss_val -= np.log(max(prob, epsilon))
            return loss_val / len(episode_data) + lambda_k * np.linalg.norm(delta, 1)

        res = minimize(
            loss,
            x0=np.zeros_like(beta_ag),
            method='L-BFGS-B',
            bounds=[(-self.W, self.W)] * len(beta_ag)  # Add parameter constraints
        )
        return res.x


    def _negative_log_likelihood(self, beta, X, p, y):
        """Negative loglikelihood (Eq.10)"""
        utility = p - X @ beta
        prob1 = 1 - self.F(utility)  # P(y=1)
        prob0 = self.F(utility)  # P(y=0)

        # 添加小常数避免log(0)
        epsilon = 1e-10
        loss = -np.sum(
            (y == 1) * np.log(np.maximum(prob1, epsilon)) +
            (y == 0) * np.log(np.maximum(prob0, epsilon))
        )
        return loss / len(y)  # 平均损失



class Online2OnlineRKHS:
    def __init__(self, F_dist='logistic', noise_B=1.0, W=1.0,
                 kernel='rbf', gamma=1.0, alpha=1.0, beta=0.5):
        """
        Online-to-online RKHS transfer pricing algorithm

        Parameters:
        F_dist: Noise distribution type ('logistic' or 'normal')
        noise_B: Support boundary for noise [-B, B]
        W: Bound for utility magnitudes
        kernel: Kernel type ('rbf' or 'linear')
        gamma: RBF kernel bandwidth
        alpha: Effective dimension parameter (λ^{-1/(2α)} eigenvalue decay)
        beta: Smoothness parameter (g = Σ^β h)
        """
        self.F_dist = F_dist
        self.noise_B = noise_B
        self.W = W
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self._init_F_functions()
        self.u_F = self._compute_uF()

        # RKHS specific initialization
        self.kernel_func = partial(rbf_kernel, gamma=self.gamma)
        self.current_target_X = None  # Track target market features

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
            raise ValueError("Unsupported distribution")

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
        """Compute u_F for RKHS setting"""
        x1 = -2 * self.W
        x2 = 2 * self.W

        if self.F_dist == 'logistic':
            term1 = 1 - self.F(x1)
            term2 = self.F(x2)
        else:
            term1 = self.F_deriv(x1) / (self.F(x1) + 1e-10)
            term2 = self.F_deriv(x2) / (1 - self.F(x2) + 1e-10)

        return max(term1, term2)

    def _get_lambda_ag(self, n_K: int) -> float:
        """Compute aggregation regularization λ_ag ~ n_K^{-2α/(2αβ+1)}"""
        return (n_K) ** (-2 * self.alpha / (2 * self.alpha * self.beta + 1))

    def _get_lambda_tf(self, n_0: int, H: float) -> float:
        """Compute transfer regularization λ_tf ~ (n_0 H^2)^{-2α/(2α+1)}"""
        return (n_0 * H ** 2) ** (-2 * self.alpha / (2 * self.alpha + 1))

    def fit(self, source_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
            target_X: np.ndarray, target_p: np.ndarray, H: float = 0.3):
        """
        Execute RKHS online transfer pricing

        Parameters:
        source_data: List of (prices, features, responses) for each source market
        target_X: Target market features [n_samples, n_features]
        target_p: True valuations (for simulation)
        H: Task similarity parameter (∥ω∥_H ≤ H)

        Returns:
        {
            'prices': Price decisions,
            'alphas': RKHS coefficients for each episode,
            'kernel_X': Kernel matrix anchors
        }
        """
        T = len(target_X)
        prices = np.zeros(T)
        alphas = []
        collected_data = []
        kernel_X = None  # Will store the kernel matrix anchors

        # Initial state (m=1)
        prices[0] = 0
        alpha_m = np.zeros(1)
        alphas.append(alpha_m)
        collected_data.append((0, target_X[0], 1))

        m = 2
        while True:
            prev_start_t = 2 ** (m - 2)
            prev_end_t = 2 ** (m - 1) - 1
            start_t = 2 ** (m - 1)
            end_t = min(2 ** m - 1, T)

            # 1. Aggregate source market data
            p_all = np.concatenate([d[0][prev_start_t - 1:prev_end_t] for d in source_data])
            X_all = np.concatenate([d[1][prev_start_t - 1:prev_end_t] for d in source_data])
            y_all = np.concatenate([d[2][prev_start_t - 1:prev_end_t] for d in source_data])

            # 2. Compute RKHS aggregated estimate g^(ag)
            g_ag, K_ag = self._aggregate_rkhs_estimate(p_all, X_all, y_all)

            # 3. Compute debiasing term δ_m
            prev_data = collected_data[:1] if m < 2 else collected_data[prev_start_t - 1:prev_end_t]
            delta_m, kernel_X = self._compute_rkhs_delta(
                g_ag, prev_data, H, X_all if kernel_X is None else kernel_X
            )

            # 4. Combine estimates: g_m = g_ag + δ_m
            alpha_m = g_ag + delta_m
            # alpha_m = np.concatenate([g_ag, delta_m])
            alphas.append(alpha_m)

            # 5. Apply pricing
            for t in range(start_t - 1, end_t):
                # Compute g(x_t) = Σ α_i K(x_i, x_t)
                if kernel_X is None:
                    k_vec = self.kernel_func(target_X[t].reshape(1, -1), X_all)
                else:
                    k_vec = self.kernel_func(target_X[t].reshape(1, -1), kernel_X)

                g_xt = np.dot(k_vec, alpha_m)
                prices[t] = self.h(g_xt)
                y_t = (target_p[t] >= prices[t]).astype(int)
                collected_data.append((prices[t], target_X[t], y_t))

            m += 1
            if end_t == T:
                break

        return {
            'prices': prices,
            'alphas': alphas,
            'kernel_X': kernel_X
        }

    def _aggregate_rkhs_estimate(self, p: np.ndarray, X: np.ndarray, y: np.ndarray):
        """Compute g^(ag) via kernel ridge regression"""
        n_K = len(X)
        lambda_ag = self._get_lambda_ag(n_K)

        # Compute kernel matrix
        K = self.kernel_func(X)

        # Solve (K + λI)α = y
        alpha = np.linalg.solve(K + lambda_ag * np.eye(n_K), y)

        def g_func(x_new):
            k_vec = self.kernel_func(x_new.reshape(1, -1), X)
            return np.dot(k_vec, alpha)

        return g_func, X  # Return function and kernel anchors

    def _compute_rkhs_delta(self, g_ag, episode_data, H, kernel_X):
        """Compute δ_m in RKHS with regularization"""
        p_ep, X_ep, y_ep = zip(*episode_data)
        p_ep = np.array(p_ep)
        X_ep = np.array(X_ep)
        y_ep = np.array(y_ep)
        n_0 = len(X_ep)

        # Compute kernel matrices
        K_ep = self.kernel_func(X_ep, kernel_X)
        lambda_tf = self._get_lambda_tf(n_0, H)

        # Define loss function
        def loss(alpha_delta):
            g_total = g_ag(X_ep) + K_ep @ alpha_delta
            utility = p_ep - g_total
            prob = self.F(utility)
            epsilon = 1e-10
            loglik = np.sum(
                (y_ep == 1) * np.log(np.maximum(1 - prob, epsilon)) + (y_ep == 0) * np.log(np.maximum(prob, epsilon)))
            reg = lambda_tf * alpha_delta.T @ K_ep @ alpha_delta
            return -loglik / n_0 + reg

        # Optimize
        alpha_init = np.zeros(len(kernel_X))
        res = minimize(loss, alpha_init, method='L-BFGS-B')

        def delta_func(x_new):
            k_vec = self.kernel_func(x_new.reshape(1, -1), kernel_X)
            return np.dot(k_vec, res.x)

        return delta_func, kernel_X