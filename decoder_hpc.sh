#!/bin/bash
#
#SBATCH --job-name=decoder
#SBATCH --output=decoder_540968.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
#cd /scratch/gilbreth/"$USER"/AdaptiveEnsemble || exit
#conda activate d2l

source ~/chengc.sh
cd /scratch/gilbreth/"$USER"/cheng/EdgeFlex || exit
conda activate ../AdaptiveEnsemble/penv


python train_100.py --pp 5 --n_embed 4096 --n_parts 8 --id 0 --resume
python train_100.py --pp 5 --n_embed 4096 --n_parts 8 --id 1 --resume
python train_100.py --pp 5 --n_embed 4096 --n_parts 8 --id 2 --resume
python train_100.py --pp 5 --n_embed 4096 --n_parts 8 --id 3 --resume
python train_100.py --pp 5 --n_embed 4096 --n_parts 8 --id 4 --resume

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