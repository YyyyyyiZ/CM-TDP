import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils import *
from utils.config import *
from algorithm.on2on import Online2Online
from algorithm.off2on import Offline2Online
from algorithm.RMLP import RMLP


def run_comparison(config_list: dict, name: str, n_sim: int = 50):
    """Compare algorithms across different market scenarios"""
    algorithms = ['rmlp', 'offline', 'online']

    results = {}
    for key, config in config_list.items():
        one_result = {alg: {'regret': [], 'beta_error': []} for alg in algorithms}
        for j in tqdm(range(n_sim), desc=config['description']):
            config['seed'] = j

            # Generate target market (always uses current time period)
            target_X, _, _, target_v, true_beta = MarketSimulator.generate_market_data(config)

            # Generate source markets
            offline_source_data = []
            online_source_data = []
            off_config = config.copy()
            if 'off_T' in config:
                off_config['T'] = config['off_T']
            X_k, p_k, y_k, _, _ = MarketSimulator.generate_market_data(off_config)
            offline_source_data.append((p_k, X_k, y_k))
            for k in range(config['K']):
                config['seed'] = config['seed']+k
                X_k, p_k, y_k, _ , _ = MarketSimulator.generate_market_data(config)
                online_source_data.append((p_k, X_k, y_k))

            # Initialize all models
            models = {
                'offline': Offline2Online(F_dist=F_dist, W=W),
                'online': Online2Online(F_dist=F_dist, W=W),
                'rmlp': RMLP(F_dist=F_dist, W=W),
            }

            # Run all algorithms
            for alg_name, model in models.items():
                if alg_name == 'offline':
                    alg_results = model.fit(offline_source_data, target_X, target_v)
                elif alg_name == 'online':
                    alg_results = model.fit(online_source_data, target_X, target_v)
                else:  # rmlp and basic linucb
                    alg_results = model.fit(target_X, target_v)

                regret = compute_regret(alg_results['prices'], target_X, true_beta, model)
                one_result[alg_name]['regret'].append(regret)
        results[key] = one_result

    # Compute statistics
    stats = {}
    for key in config_list.keys():
        stats[key] = {}
        for alg in algorithms:
            regret_data = results[key][alg]['regret']  # shape: (n_runs, n_periods)

            mean_regret = np.mean(regret_data, axis=0)
            std_regret = np.std(regret_data, axis=0)

            n_periods = config['T']
            periods = np.arange(1, n_periods + 1)  # 1, 2, ..., n_periods
            avg_regret = mean_regret / periods  # 累积平均 regret
            avg_std = std_regret / periods

            stats[key][alg] = {
                'mean_regret': mean_regret,  # cumulative regret
                'std_regret': std_regret,  # std error for cumulative regret
                'avg_regret': avg_regret,  # cumulative average regret per period
                'avg_std': avg_std,  # std error for cumulative average regret
            }

    plot_and_save_results(stats, name=name)

def plot_and_save_results(results: dict, name: str, kk:int=50):
    """Visualize and save comparison results"""
    plt.figure(figsize=(8, 6))

    alg_styles = {
        'offline': ('blue', '-', 'Offline to Online'),
        'online': ('green', '-', 'Online to Online'),
        'rmlp': ('#38ACEC', '--', 'RMLP')
    }


    pair_list = [('rmlp', 'online'), ('rmlp', 'offline')]
    key_list = list(results.keys())

    for pair in pair_list:
        base, trans = pair

        mean_base = results[key_list[0]][base]['mean_regret']
        std_base = results[key_list[0]][base]['std_regret']
        plt.plot(mean_base, label=f'{alg_styles[base][2]}',
                 color=alg_styles[base][0], linestyle=alg_styles[base][1])
        plt.fill_between(range(len(mean_base)), mean_base - std_base, mean_base + std_base,
                         alpha=0.1, color=alg_styles[base][0])
        plt.text(len(mean_base), mean_base[-1], f'{base}')
        market_li = [1, 3, 5, 10]    # online
        source_li = [50, 100, 200, 500]     # offline

        sorted_key_list = sorted(
            key_list,
            key=lambda k: results[k][trans]['mean_regret'][-1],
            reverse=True
        )

        for i, one_key in enumerate(sorted_key_list):
            mean = results[one_key][trans]['mean_regret']
            std = results[one_key][trans]['std_regret']
            plt.plot(mean, label=f'{alg_styles[trans][2]}',
                     color=alg_styles[trans][0], linestyle=alg_styles[trans][1])
            plt.fill_between(range(len(mean)), mean - std, mean + std,
                             alpha=0.1, color=alg_styles[trans][0])
            if trans in ['offline']:
                plt.text(len(mean), mean[-1], f'$n_\mathcal{{K}} = {source_li[i]}$')
            if trans in ['online']:
                plt.text(len(mean), mean[-1], f'$K={market_li[i]}$')

        plt.xlabel('Time Period')
        plt.ylabel('Cumulative Regret')
        plt.xlim(-50, len(mean) * 1.2)
        plt.grid(True)

        # Save plot
        filename = f"fig/dense/cumulative_linear_{name}_{base}_{trans}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {filename}")



if __name__ == "__main__":
    all_results = {}
    for key, config in config_dict.items():
        print(f"\nRunning {key} ...")
        run_comparison(config, name=key)
