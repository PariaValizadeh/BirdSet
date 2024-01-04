#!/bin/bash
#SBATCH --job-name=embed_birdnet
#SBATCH --output=embed_birdnet.log
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@90

date;hostname;pwd
source /mnt/stud/home/mrichert/.zshrc
conda activate gadme
cd /mnt/stud/home/mrichert/Projects/GADME-BaselineResults-BA/src
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
srun python -u main.py experiment=s_high_sierras_embed_birdnet_multilabel trainer=single_gpu paths=server
