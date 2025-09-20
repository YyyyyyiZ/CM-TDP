from utils.utils import MarketSimulator

F_dist = 'logistic'  # 'normal'
W = 2
# Configuration Setup

CONFIGURATIONS_idt_10 = {
    'identical_1': {
        'd': 10, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.8),
        'description': 'd=10, K=1, off_T=50, All markets share identical beta'
    },
    'identical_2': {
        'd': 10, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.8),
        'description': 'd=10, K=3, off_T=100, All markets share identical beta'
    },
    'identical_3': {
        'd': 10, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.8),
        'description': 'd=10, K=5, off_T=200, All markets share identical beta'
    },
    'identical_4': {
        'd': 10, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.8),
        'description': 'd=10, K=10, off_T=500, All markets share identical beta'
    },
}

CONFIGURATIONS_idt_15 = {
    'identical_1': {
        'd': 15, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.8),
        'description': 'd=15, K=1, off_T=50, All markets share identical beta'
    },
    'identical_2': {
        'd': 15, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.8),
        'description': 'd=15, K=3, off_T=100, All markets share identical beta'
    },
    'identical_3': {
        'd': 15, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.8),
        'description': 'd=15, K=5, off_T=200, All markets share identical beta'
    },
    'identical_4': {
        'd': 15, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.8),
        'description': 'd=15, K=10, off_T=500, All markets share identical beta'
    },
}

CONFIGURATIONS_idt_20 = {
    'identical_1': {
        'd': 20, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.8),
        'description': 'd=20, K=1, off_T=50, All markets share identical beta'
    },
    'identical_2': {
        'd': 20, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.8),
        'description': 'd=20, K=3, off_T=100, All markets share identical beta'
    },
    'identical_3': {
        'd': 20, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.8),
        'description': 'd=20, K=5, off_T=200, All markets share identical beta'
    },
    'identical_4': {
        'd': 20, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.8),
        'description': 'd=20, K=10, off_T=500, All markets share identical beta'
    },
}

CONFIGURATIONS_idt_100 = {
    'identical_1': {
        'd': 100, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(100, W=W, sparsity=0.8),
        'description': 'd=100, K=1, off_T=50, All markets share identical beta'
    },
    'identical_2': {
        'd': 100, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(100, W=W, sparsity=0.8),
        'description': 'd=100, K=3, off_T=100, All markets share identical beta'
    },
    'identical_3': {
        'd': 100, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(100, W=W, sparsity=0.8),
        'description': 'd=100, K=5, off_T=200, All markets share identical beta'
    },
    'identical_4': {
        'd': 100, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(100, W=W, sparsity=0.8),
        'description': 'd=100, K=10, off_T=500, All markets share identical beta'
    },
}

CONFIGURATIONS_sparse_10 = {
    'sparse_1': {
        'd': 10, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=10, K=1, off_T=50, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_2': {
        'd': 10, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=10, K=5, off_T=100, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_3': {
        'd': 10, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=10, K=5, off_T=200, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_4': {
        'd': 10, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=10, K=10, off_T=500, Markets have sparse differences in beta (s0-sparse)'
    },
}

CONFIGURATIONS_sparse_15 = {
    'sparse_1': {
        'd': 15, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=15, K=1, off_T=50, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_2': {
        'd': 15, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=15, K=3, off_T=100, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_3': {
        'd': 15, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=15, K=5, off_T=200, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_4': {
        'd': 15, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=15, K=10, off_T=500, Markets have sparse differences in beta (s0-sparse)'
    },
}

CONFIGURATIONS_sparse_20 = {
    'sparse_1': {
        'd': 20, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=20, K=1, off_T=50, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_2': {
        'd': 20, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=20, K=3, off_T=100, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_3': {
        'd': 20, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=20, K=5, off_T=200, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_4': {
        'd': 20, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=20, K=10, off_T=500, Markets have sparse differences in beta (s0-sparse)'
    },
}

CONFIGURATIONS_sparse_100 = {
    'sparse_1': {
        'd': 100, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(100, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=100, K=1, off_T=50, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_2': {
        'd': 100, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(100, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=100, K=3, off_T=100, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_3': {
        'd': 100, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(100, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=100, K=5, off_T=200, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_4': {
        'd': 100, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(100, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.3,  # At most 30% non-zero differences
        'description': 'd=100, K=10, off_T=500, Markets have sparse differences in beta (s0-sparse)'
    },
}

CONFIGURATIONS_dense_10 = {
    'sparse_1': {
        'd': 10, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=10, K=1, off_T=50, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_2': {
        'd': 10, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=10, K=3, off_T=100, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_3': {
        'd': 10, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=10, K=5, off_T=200, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_4': {
        'd': 10, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=10, K=10, off_T=500, Markets have sparse differences in beta (s0-sparse)'
    },
}

CONFIGURATIONS_dense_15 = {
    'sparse_1': {
        'd': 15, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=15, K=1, off_T=50, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_2': {
        'd': 15, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=15, K=3, off_T=100, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_3': {
        'd': 15, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=15, K=5, off_T=200, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_4': {
        'd': 15, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=15, K=10, off_T=500, Markets have sparse differences in beta (s0-sparse)'
    },
}

CONFIGURATIONS_dense_20 = {
    'sparse_1': {
        'd': 20, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=20, K=1, off_T=50, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_2': {
        'd': 20, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=20, K=3, off_T=100, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_3': {
        'd': 20, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=20, K=5, off_T=200, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_4': {
        'd': 20, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=20, K=10, off_T=500, Markets have sparse differences in beta (s0-sparse)'
    },
}


CONFIGURATIONS_dense_100 = {
    'sparse_1': {
        'd': 100, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(100, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=100, K=1, off_T=50, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_2': {
        'd': 100, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(100, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=100, K=3, off_T=100, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_3': {
        'd': 100, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(100, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=100, K=5, off_T=200, Markets have sparse differences in beta (s0-sparse)'
    },
    'sparse_4': {
        'd': 100, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'dense_difference',
        'base_beta': MarketSimulator.generate_beta(100, W=W, sparsity=0.8),
        'delta_W': 0.3,
        'delta_sparsity': 0.5,  # At most 50% non-zero differences
        'description': 'd=100, K=10, off_T=500, Markets have sparse differences in beta (s0-sparse)'
    },
}

config_dict = {'cfg_sparse_10': CONFIGURATIONS_sparse_10,
               'cfg_sparse_15': CONFIGURATIONS_sparse_15,
               'cfg_sparse_20': CONFIGURATIONS_sparse_20,
               'cfg_sparse_100': CONFIGURATIONS_sparse_20}
