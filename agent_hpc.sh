#!/bin/bash
#
#SBATCH --job-name=ensemble
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#
#SBATCH --mail-type=END
#SBATCH --mail-user=cc6858@nyu.edu

source ~/.bashrc
#cd /scratch/gilbreth/"$USER"/AdaptiveEnsemble || exit
#conda activate d2l

source ~/chengc.sh
cd /scratch/gilbreth/"$USER"/cheng/EdgeFlex || exit
conda activate ../AdaptiveEnsemble/penv

python train_agent.py