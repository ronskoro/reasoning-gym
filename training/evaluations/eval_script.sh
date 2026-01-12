#!/bin/bash
#SBATCH --job-name="reasoning_gym"
#SBATCH --partition=a100-galvani
#SBATCH --time=9:00:00
#SBATCH --gpus=1
#SBATCH --output=/home/bethge/bkr261/.logs/%u-%x-%j.out
#SBATCH --error=/home/bethge/bkr261/.logs/%u-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ronald.skorobogat@bethgelab.org
#SBATCH --nodes=1

~/.conda/envs/reasoning-gym/bin/python -u evaluate_model.py --config inter_generalisation/algorithmic.yaml