import numpy as np
from scipy.stats import logistic
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import pairwise_kernels


class MarketSimulator:
    @staticmethod
    def generate_beta(d: int, W: float, sparsity: float) -> np.ndarray:
        """Generate sparse beta vector with ||beta||_1 <= W"""
        beta = np.zeros(d)
        non_zero = max(1, int(d * sparsity))
        beta[:non_zero] = np.random.randn(non_zero)
        beta = np.abs(beta * W / (np.sum(np.abs(beta)) + 1e-10))
        return beta

    @staticmethod
    def generate_market_data(config: dict) -> Tuple:
        """Generate data for a single market with guaranteed class balance"""
        np.random.seed(config['seed'])
        d = config['d']
        T = config['T']

        # Generate features with bounded support (||x||_∞ ≤ 1)
        X = np.random.uniform(0.3, 1, size=(T, d))

        # Generate beta based on configuration
        if config['scenario'] == 'identical':
            beta = config['base_beta']
        elif config['scenario'] == 'sparse_difference':
            delta = MarketSimulator.generate_beta(d, config['delta_W'], config['delta_sparsity'])
            beta = config['base_beta'] + delta

        if config.get('F_dist', 'logistic') == 'logistic':
            z = np.random.logistic(scale=1, size=T)
        else:
            z = np.random.normal(scale=1, size=T)

        X_beta = X @ beta
        z_min, z_max = max(-np.min(X_beta), -1), min(np.max(X_beta), 1)
        z = np.clip(z, z_min, z_max)
        v = X_beta + z

        perturbation = v * (np.random.rand(*v.shape) - 0.5)
        p = v + perturbation

        # Generate responses with guaranteed class balance
        max_attempts = 10
        for _ in range(max_attempts):
            y = (v >= p).astype(int)
            if len(np.unique(y)) >= 2:  # Ensure we have both classes
                return X, p, y, v, beta
            p = p * 0.8  # Adjust prices if we get all zeros or ones

        return X, p, y, v, beta

class RKHSMarketSimulator:
    @staticmethod
    def generate_rkhs_function(X: np.ndarray, kernel_type: str = 'rbf',
                               kernel_params: dict = None, sparsity: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a random function in RKHS by sampling from the kernel's feature space

        Args:
            X: Input features (n_samples, n_features)
            kernel_type: Type of kernel ('rbf', 'linear')
            kernel_params: Parameters for the kernel
            sparsity: Fraction of samples to use as basis points (induces sparsity)

        Returns:
            weights: Coefficients for kernel expansion
            basis_points: Selected basis points
        """
        if kernel_params is None:
            kernel_params = {'gamma': 1.0} if kernel_type == 'rbf' else {}

        n_samples = X.shape[0]
        n_basis = max(1, int(n_samples * sparsity))
        basis_idx = np.random.choice(n_samples, size=n_basis, replace=False)
        basis_points = X[basis_idx]

        # Generate random weights with decaying magnitudes
        weights = np.random.randn(n_basis) * np.exp(-0.1 * np.arange(n_basis))
        weights /= (np.linalg.norm(weights) + 1e-8)  # Normalize

        return weights, basis_points

    @staticmethod
    def evaluate_rkhs_function(x: np.ndarray, weights: np.ndarray, basis_points: np.ndarray,
                               kernel_type: str = 'rbf', kernel_params: dict = None) -> float:
        """
        Evaluate RKHS function at point x using kernel expansion

        Args:
            x: Input point (n_features,)
            weights: Coefficients for kernel expansion (must match basis_points)
            basis_points: Basis points for kernel expansion (n_basis, n_features)
            kernel_type: Type of kernel
            kernel_params: Parameters for the kernel

        Returns:
            Function value at x
        """
        if kernel_params is None:
            kernel_params = {'gamma': 1.0} if kernel_type == 'rbf' else {}

        # Ensure x is 2D and basis_points is 2D
        x = x.reshape(1, -1) if len(x.shape) == 1 else x
        basis_points = basis_points.reshape(-1, x.shape[1]) if len(basis_points.shape) == 1 else basis_points

        # Verify dimensions
        if weights.shape[0] != basis_points.shape[0]:
            raise ValueError(
                f"Weights dimension {weights.shape[0]} doesn't match basis points dimension {basis_points.shape[0]}")

        K = pairwise_kernels(basis_points, x, metric=kernel_type, **kernel_params)
        return np.dot(weights, K.ravel())

    @staticmethod
    def generate_market_data(config: dict) -> Tuple:
        """Generate data for a single market in RKHS setting"""
        np.random.seed(config['seed'])
        d = config['d']
        T = config['T']
        kernel_type = config.get('kernel_type', 'rbf')
        kernel_params = config.get('kernel_params', {'gamma': 1.0})

        # Generate base function
        base_X = np.random.uniform(0.3, 1, size=(100, d))  # Generate some initial points
        base_weights, base_basis = RKHSMarketSimulator.generate_rkhs_function(
            base_X, kernel_type=kernel_type, kernel_params=kernel_params, sparsity=0.7)

        # Generate market-specific function
        if config['scenario'] == 'identical':
            weights = base_weights
            basis_points = base_basis
        elif config['scenario'] == 'smooth_difference':
            # Generate difference function with same basis points
            diff_weights, _ = RKHSMarketSimulator.generate_rkhs_function(
                base_basis, kernel_type, kernel_params, sparsity=1.0)  # Use same basis

            # Combine with base function (with controlled difference magnitude)
            weights = base_weights + config['delta_scale'] * diff_weights
            basis_points = base_basis  # Keep same basis points

        # Generate features with bounded support (||x||_∞ ≤ 1)
        X = np.random.uniform(0.3, 1, size=(T, d))

        # Evaluate utility function
        v = np.array([
            RKHSMarketSimulator.evaluate_rkhs_function(
                x, weights, basis_points, kernel_type, kernel_params)
            for x in X
        ])

        # Generate noise
        if config.get('F_dist', 'logistic') == 'logistic':
            z = np.random.logistic(scale=1, size=T)
        else:
            z = np.random.normal(scale=1, size=T)

        # Clip noise to ensure reasonable price range
        z_min, z_max = max(-np.min(v), -1), min(np.max(v), 1)
        z = np.clip(z, z_min, z_max)
        v += z

        # Generate prices with perturbation
        perturbation = v * (np.random.rand(*v.shape) - 0.5)
        p = v + perturbation

        # Generate responses with guaranteed class balance
        max_attempts = 10
        for _ in range(max_attempts):
            y = (v >= p).astype(int)
            if len(np.unique(y)) >= 2:  # Ensure we have both classes
                return X, p, y, v, weights, basis_points
            p = p * 0.8  # Adjust prices if we get all zeros or ones

        return X, p, y, v, weights, basis_points

def compute_regret(algo_p: np.ndarray, target_X: np.ndarray,
                   true_beta: np.ndarray, model) -> np.ndarray:
    """Calculate cumulative regret with proper linear utility calculation"""
    # Calculate optimal prices using true beta
    utility = target_X @ true_beta  # Linear utility calculation
    opt_p = np.array([model.h(u) for u in utility])  # Apply pricing function to utility

    # Calculate revenue probabilities (clipped for numerical stability)
    eps = 1e-10
    opt_prob = np.clip(1 - model.F(opt_p - utility), eps, 1 - eps)
    algo_prob = np.clip(1 - model.F(algo_p - utility), eps, 1 - eps)

    # Calculate revenues
    opt_rev = opt_p * opt_prob
    algo_rev = algo_p * algo_prob

    regret = np.minimum(opt_rev - algo_rev, 2)

    return np.cumsum(regret)


def compute_regret_rkhs(algo_p: np.ndarray, target_X: np.ndarray,
                        true_weights: np.ndarray, basis_points: np.ndarray,
                        model, kernel_type='rbf', kernel_params=None) -> np.ndarray:
    if kernel_params is None:
        kernel_params = {'gamma': 1.0} if kernel_type == 'rbf' else {}

    # Calculate true utility using RKHS expansion
    K = pairwise_kernels(basis_points, target_X, metric=kernel_type, **kernel_params)
    utility = np.dot(true_weights, K)  # RKHS utility calculation

    # Calculate optimal prices
    opt_p = np.array([model.h_func(u) for u in utility])

    # Calculate revenue probabilities (clipped for numerical stability)
    eps = 1e-10
    opt_prob = np.clip(1 - model.F(opt_p - utility), eps, 1 - eps)
    algo_prob = np.clip(1 - model.F(algo_p - utility), eps, 1 - eps)

    # Calculate revenues
    opt_rev = opt_p * opt_prob
    algo_rev = algo_p * algo_prob

    # Clip regret to avoid extreme values
    regret = np.minimum(opt_rev - algo_rev, 2)

    return np.cumsum(regret)