#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu_p1
#SBATCH --time=6:00:0
#SBATCH --array=1-20
#SBATCH --output=/lustre/fsn1/projects/rech/pbx/utg98xt/slurm-%j.log
#SBATCH --chdir=/lustre/fshomisc/home/rech/genolx01/utg98xt/unitary-optimization-manopt/dynamics/
date;hostname;id;pwd

echo 'activating virtual environment'
module unload gcc
module load pytorch-gpu/py3/2.7.0
which python

export WANDB_MODE=offline

export WANDB_DIR=/lustre/fsn1/projects/rech/pbx/utg98xt/wandb_runs

mkdir -p $WANDB_DIR


echo "Running sweep task ${SLURM_ARRAY_TASK_ID}"

python hyperparam-search.py \
	--sweep_id $((SLURM_ARRAY_TASK_ID)) --dim 8 --n_epochs 10000 --activation_function tanh