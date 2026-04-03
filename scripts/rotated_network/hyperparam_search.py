import torch as torch
import torch.nn.functional as F
import sys
from pathlib import Path
REPO_ROOT = Path.cwd().parent.parent
sys.path.append(str(REPO_ROOT))
from src.data.generation import GaussianPulseDataset
from src.utils.plotting import plot_interm_fig, plot_ground_truth
from src.models.rnn import SequentialRNN
import numpy as np
import argparse
import wandb
import plotly.graph_objects as go
from datetime import datetime
import plotly.io as pio
import random
import math
import fcntl
import json


def loss_function(y, y_hat):
    return F.mse_loss(y,y_hat)


def eigen_imag_score(eigvals):
    """Lower is better: mean absolute imaginary part of eigenvalues."""
    imag_parts = np.abs(np.imag(eigvals))
    return float(np.mean(imag_parts))


def combined_score(avg_mse_loss, eig_imag, lambda_im=1.0):
    """Composite loss used for selection of best hyperparameters."""
    return float(avg_mse_loss + lambda_im * eig_imag)


def sample_hyperparameters(seed):
    random.seed(seed)

    return {
        "anti_masked_reg": random.uniform(0.0, 3.0),
        "masked_reg": random.uniform(0.5, 7.0),
        "eta_min": random.uniform(0.0, 1e-5),
        "batch_size": random.choice([16, 32, 64]),
        "l2_reg": math.exp(random.uniform(math.log(0.005), math.log(0.5))),
        "starting_lr": math.exp(random.uniform(math.log(1e-5), math.log(0.1))),
    }

def train(model, loader, optimizer, scheduler, config, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        for (X, y) in loader:
            optimizer.zero_grad()
            y_pred = model(X)
            mask = y > 0
            anti_mask = y == 0
            l2_regularization = config.l2_reg*torch.linalg.norm(model.W_hh.weight, ord=2)
            masked_loss = config.masked_reg*loss_function(y[mask],y_pred[mask])
            anti_masked_loss = config.anti_masked_reg*loss_function(y[anti_mask],y_pred[anti_mask])
            loss = masked_loss + anti_masked_loss + l2_regularization
            loss.backward()
            optimizer.step()
        scheduler.step() # scheduler updates per epoch

        #_________________WandB_Logging every 10 epochs________________

        if epoch%10==0:
            wandb.log({'epoch':epoch})
            wandb.log({'weight_regularization(l2)': l2_regularization.item()})
            wandb.log({'masked_mse_loss': masked_loss.item()})
            wandb.log({'antimasked_mse_loss': anti_masked_loss.item()})
            wandb.log({'total_loss':loss.item()})


def main(args, hps=None):

    #________________________WANDB Initialization______________________

    if hps is None:
        hps = sample_hyperparameters(args.sweep_id)

    experiment_name = f"rnn-dynamics-rotated-{args.activation_function}-hyperparam-search"

    wandb.init(
        project=experiment_name,
        mode="offline",
        config=hps,
        group="offline_sweep",
        name=f"sweep_{args.sweep_id}"
    )

    config = wandb.config
    print(config)



  #__________________________Data Generation___________________________


    n_neurons = args.dim
    seq_length = args.time_steps

    gps = GaussianPulseDataset(seq_length=seq_length, n_pulse=n_neurons, n_samples = 500, sequence_type='rnn', activation_function=args.activation_function)
    sample = gps[6]


  #__________________________Model Setup______________________________

    # set seed for reproducibility
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1234)


    input_size = 1

    # pack dataset and initialize model and optimizer
    loader = torch.utils.data.DataLoader(gps, batch_size=config.batch_size, shuffle=True)
    model = SequentialRNN(input_size, n_neurons, activation_function=args.activation_function, rotation=args.with_rotation, noise=args.noise)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.starting_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=config.eta_min)


  #__________________________Plotting Setup and Ground Truth______________________________


    custom_template = go.layout.Template(pio.templates['plotly_white'])
    custom_template.layout.xaxis.showgrid = False
    custom_template.layout.yaxis.showgrid = False
    pio.templates['custom_template'] = custom_template
    pio.templates.default = 'custom_template'


    rotated_W = (model.D.T @ gps.sequence.T @ model.D).detach().numpy()
    fig = plot_ground_truth(sample=sample,T=gps.sequence.T,D=model.D,rotated_W=rotated_W)

    wandb.log({'Ground Truth Data':fig})



  #__________________________RNN Training______________________________

    train(model, loader, optimizer, scheduler, config, args.n_epochs)


#_________________Logging to WandB________________

    # log sample prediction

    with torch.no_grad():
        test_outp = model(torch.reshape(sample[0], (1,50,1)).float()).squeeze(0).detach().numpy()
    M_in = model.W_hh.weight.detach().numpy()
    interm_fig = plot_interm_fig(M_in, test_outp)
    wandb.log({'Intermediate Result':interm_fig})
    wandb.log({'Network Size':args.dim})
    wandb.log({'Noise Level':args.noise})

    # log the model state dictionary

    torch.save(model.state_dict(), f'/lustre/fsn1/projects/rech/pbx/utg98xt/model_weights/model_{args.sweep_id}.pth')
    artifact = wandb.Artifact("torch-model", type="model")
    artifact.add_file(f'/lustre/fsn1/projects/rech/pbx/utg98xt/model_weights/model_{args.sweep_id}.pth')  # or add_dir(...)
    wandb.run.log_artifact(artifact)


#_________________Logging Hyperparameters, MSE Loss and eigenvalues to local file________________

    mse_losses = []
    for (X, y) in loader:
        y_pred = model(X)
        mse_loss = loss_function(y,y_pred)
        mse_losses.append(mse_loss.item())
    avg_mse_loss = sum(mse_losses) / len(mse_losses)
    W_hh_eigvals = np.linalg.eigvals(model.W_hh.weight.detach().numpy())
    eig_imag = eigen_imag_score(W_hh_eigvals)
    composite = combined_score(avg_mse_loss, eig_imag, lambda_im=1.0)

    # Ensure paths exist
    log_dir = Path(f'/lustre/fsn1/projects/rech/pbx/utg98xt/{experiment_name}/hyperparam_logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%y%m%d")
    run_log_path = log_dir / f"{timestamp}_{args.sweep_id}.json"

    config_dict = dict(config)
    run_record = {
        "sweep_id": args.sweep_id,
        "timestamp": datetime.now().isoformat(),
        "Hyperparameters": config_dict,
        "Average MSE Loss": avg_mse_loss,
        "W_hh Eigenvalues": W_hh_eigvals.tolist(),
        "Eigen Imag Score": eig_imag,
        "Composite Score": composite,
    }

    # Write per-run log
    with open(run_log_path, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(run_record, f, indent=4)
        fcntl.flock(f, fcntl.LOCK_UN)

    # Append full history (optional but useful)
    history_path = log_dir / 'hyperparam_search_history.json'
    if history_path.exists():
        with open(history_path, 'r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                history_data = json.load(f)
            except json.JSONDecodeError:
                history_data = []
            history_data.append(run_record)
            f.seek(0)
            json.dump(history_data, f, indent=4)
            f.truncate()
            fcntl.flock(f, fcntl.LOCK_UN)
    else:
        with open(history_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump([run_record], f, indent=4)
            fcntl.flock(f, fcntl.LOCK_UN)

    # Maintain best hyperparameters across runs using composite score (mse + eigimag penalty)
    best_path = log_dir / 'best_hyperparameters.json'
    if best_path.exists():
        with open(best_path, 'r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                existing_best = json.load(f)
            except json.JSONDecodeError:
                existing_best = None
            if existing_best is None or run_record['Composite Score'] < existing_best.get('Composite Score', float('inf')):
                f.seek(0)
                json.dump(run_record, f, indent=4)
                f.truncate()
            fcntl.flock(f, fcntl.LOCK_UN)
    else:
        with open(best_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(run_record, f, indent=4)
            fcntl.flock(f, fcntl.LOCK_UN)

    return run_record


#__________________________End of Main______________________________



if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=int, required=True)
    parser.add_argument('--dim', type=int, default=25)
    parser.add_argument('--time_steps', type=int, default=50)
    parser.add_argument('--activation_function', type=str, default='linear')
    parser.add_argument('--with_rotation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--n_restarts', type=int, default=20)
    args=parser.parse_args()
    print(args)
    main(args)