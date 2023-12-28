#!/bin/bash
#
#SBATCH --job-name=ensemble
#SBATCH --output=ensemble_12.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -A schaterj-g

source ~/.bashrc
#cd /scratch/gilbreth/"$USER"/AdaptiveEnsemble || exit
#conda activate d2l

source ~/chengc.sh
cd /scratch/gilbreth/"$USER"/cheng/EdgeFlex || exit
conda activate ../AdaptiveEnsemble/penv

for PP in 5 3 8
do
  for NB in 2048 1024 4096
  do
    for NP in 4 2 1 8
    do
      python test_imagenet.py --pp $PP --n_parts $NP --n_embed $NB --lr 1e-4 --ep 12
    done
  done
done

# Single Non-Quantization Model
#python train_tiny.py --ep 100 --id -1 --skip_quant

# Accuracy Profiling
#PP=5
#NU=5
#EB=2048
#for NP in 1 2 4 8
#do
#  python train_100.py --id -1 --nu $NU --pp $PP --ep 100 --n_parts $NP --n_embed $EB
#  for i in {0..4}
#  do
#    echo "$i"
#    python train_100.py --id "$i" --nu $NU --pp $PP --resume --ep 100 --n_parts $NP --n_embed $EB
#  done
#  python test_100.py --pp $PP --n_parts $NP --n_embed $EB
#done

#python train_100.py --nu 5 --pp 5 --ep 100 --id -1 --quant BQ --n_embed 1024