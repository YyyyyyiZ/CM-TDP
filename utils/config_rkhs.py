from utils.utils import MarketSimulator

F_dist = 'logistic'  # 'normal'
h = 0.3
kernel = 'rbf'
kernel_params = {'gamma': 0.5}
# Configuration Setup

CONFIGURATIONS_idt_10 = {
    'identical_1': {
        'd': 10, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=10, K=1, off_T=50'
    },
    'identical_2': {
        'd': 10, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=10, K=3, off_T=100'
    },
    'identical_3': {
        'd': 10, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=10, K=5, off_T=200'
    },
    'identical_4': {
        'd': 10, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=10, K=10, off_T=500'
    },
}

CONFIGURATIONS_idt_15 = {
    'identical_1': {
        'd': 15, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=15, K=1, off_T=50'
    },
    'identical_2': {
        'd': 15, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=15, K=3, off_T=100'
    },
    'identical_3': {
        'd': 15, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=15, K=5, off_T=200'
    },
    'identical_4': {
        'd': 15, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=15, K=10, off_T=500'
    },
}

CONFIGURATIONS_idt_20 = {
    'identical_1': {
        'd': 20, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=20, K=1, off_T=50'
    },
    'identical_2': {
        'd': 20, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=20, K=3, off_T=100'
    },
    'identical_3': {
        'd': 20, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=20, K=5, off_T=200'
    },
    'identical_4': {
        'd': 20, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=20, K=10, off_T=500'
    },
}

CONFIGURATIONS_idt_100 = {
    'identical_1': {
        'd': 100, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=100, K=1, off_T=50'
    },
    'identical_2': {
        'd': 100, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=100, K=3, off_T=100'
    },
    'identical_3': {
        'd': 100, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=100, K=5, off_T=200'
    },
    'identical_4': {
        'd': 100, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'description': 'd=100, K=10, off_T=500'
    },
}

CONFIGURATIONS_sparse_10 = {
    'sparse_1': {
        'd': 10, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=10, K=1, off_T=50'
    },
    'sparse_2': {
        'd': 10, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=10, K=3, off_T=100'
    },
    'sparse_3': {
        'd': 10, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=10, K=5, off_T=200'
    },
    'sparse_4': {
        'd': 10, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=10, K=10, off_T=500'
    },
}

CONFIGURATIONS_sparse_15 = {
    'sparse_1': {
        'd': 15, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=15, K=1, off_T=50'
    },
    'sparse_2': {
        'd': 15, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=15, K=3, off_T=100'
    },
    'sparse_3': {
        'd': 15, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=15, K=5, off_T=200'
    },
    'sparse_4': {
        'd': 15, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=15, K=10, off_T=500'
    },
}

CONFIGURATIONS_sparse_20 = {
    'sparse_1': {
        'd': 20, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=20, K=1, off_T=50'
    },
    'sparse_2': {
        'd': 20, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=20, K=3, off_T=100'
    },
    'sparse_3': {
        'd': 20, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=20, K=5, off_T=200'
    },
    'sparse_4': {
        'd': 20, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=20, K=10, off_T=500'
    },
}

CONFIGURATIONS_sparse_100 = {
    'sparse_1': {
        'd': 20, 'K': 1, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=100, K=1, off_T=50'
    },
    'sparse_2': {
        'd': 100, 'K': 3, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 100,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=100, K=3, off_T=100'
    },
    'sparse_3': {
        'd': 100, 'K': 5, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=100, K=5, off_T=200'
    },
    'sparse_4': {
        'd': 100, 'K': 10, 'T': 2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'smooth_difference',
        'kernel_type': kernel,
        'kernel_params': kernel_params,
        'delta_scale': h,  
        'description': 'd=100, K=10, off_T=500'
    },
}

config_dict = {'cfg_sparse_10': CONFIGURATIONS_sparse_10,
               'cfg_sparse_15': CONFIGURATIONS_sparse_15, 'cfg_sparse_20': CONFIGURATIONS_sparse_20,
               'cfg_idt_10': CONFIGURATIONS_idt_10,
               'cfg_idt_15': CONFIGURATIONS_idt_15, 'cfg_idt_20': CONFIGURATIONS_idt_20}
