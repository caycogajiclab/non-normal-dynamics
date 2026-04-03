
import argparse
import json
import time
from pathlib import Path
import sys
from hyperparam_search import main as hp_main
from skopt import gp_minimize
from skopt.space import Real, Integer


def load_results(filename):
    results = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                results.append(json.loads(line))
    except FileNotFoundError:
        pass
    return results


def log_result(filename, hyperparams, metrics):
    entry = {
        'timestamp': time.time(),
        'hyperparameters': hyperparams,
        'metrics': metrics,
    }
    with open(filename, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def objective(x):
    # x is [starting_lr, batch_size, masked_reg, anti_masked_reg, eta_min, l2_reg]
    hyperparams = {
        'starting_lr': float(x[0]),
        'batch_size': int(x[1]),
        'masked_reg': float(x[2]),
        'anti_masked_reg': float(x[3]),
        'eta_min': float(x[4]),
        'l2_reg': float(x[5]),
    }

    args = argparse.Namespace(
        sweep_id=int(time.time() * 1000) % 100000,
        dim=25,
        time_steps=50,
        activation_function='linear',
        with_rotation=True,
        n_epochs=20,
        noise=0.0,
        n_restarts=1,
    )

    run_record = hp_main(args, hps=hyperparams)  # returns run_record with metrics

    metrics = {
        'avg_mse_loss': run_record['Average MSE Loss'],
        'eig_imag': run_record['Eigen Imag Score'],
        'composite': run_record['Composite Score'],
    }

    log_result('hpo_log.jsonl', hyperparams, metrics)
    return metrics['composite']


space = [
    Real(1e-5, 1e-1, prior='log-uniform', name='starting_lr'),
    Integer(16, 128, name='batch_size'),
    Real(0.0, 7.0, name='masked_reg'),
    Real(0.0, 3.0, name='anti_masked_reg'),
    Real(0.0, 1e-5, prior='log-uniform', name='eta_min'),
    Real(0.005, 0.5, prior='log-uniform', name='l2_reg'),
]


def main(args):
    results = load_results('hpo_log.jsonl')
    x0 = [[r['hyperparameters']['starting_lr'], r['hyperparameters']['batch_size'], r['hyperparameters']['masked_reg'], r['hyperparameters']['anti_masked_reg'], r['hyperparameters']['eta_min'], r['hyperparameters']['l2_reg']] for r in results]
    y0 = [r['metrics']['composite'] for r in results]

    result = gp_minimize(objective, space, x0=x0 if x0 else None, y0=y0 if y0 else None, n_calls=30, random_state=42)

    best = {
        'best_params': {
            'starting_lr': result.x[0],
            'batch_size': result.x[1],
            'masked_reg': result.x[2],
            'anti_masked_reg': result.x[3],
            'eta_min': result.x[4],
            'l2_reg': result.x[5],
        },
        'best_score': result.fun,
    }
    Path('hpo_best.json').write_text(json.dumps(best, indent=4))
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    print(main(args))