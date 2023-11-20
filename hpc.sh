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
cd /scratch/gilbreth/"$USER"/AdaptiveEnsemble || exit
conda activate d2l

#source ~/chengc.sh
#cd /scratch/gilbreth/"$USER"/cheng/AdaptiveEnsemble || exit
#conda activate ./penv

PP=5
NU=5
NP=1
EB=4096
#python train_100.py --id -1 --nu $NU --pp $PP --ep 100 --n_parts $NP --n_embed $EB
for i in {0..4}
do
  echo "$i"
  python train_100.py --id "$i" --nu $NU --pp $PP --resume --ep 100 --n_parts $NP --n_embed $EB
done
#python train_100.py --id 0 --resume --ep 100
python test_100.py --pp $PP --n_parts $NP