#!/bin/bash
#SBATCH --time=0-15:00:00
#SBATCH --nodes=1 
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --account=def-mjshafie
#SBATCH --gres=gpu:v100:3
#SBATCH --cpus-per-task=15

bash scripts/hparams/critic/critic_hparam.sh kfac