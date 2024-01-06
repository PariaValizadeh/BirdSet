#!/bin/bash
#SBATCH --job-name=ml_e_bn
#SBATCH --output=embed_birdnet_ml.log
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
srun python -u main.py experiment=high_sierras_local_embed_birdnet_multilabel trainer=single_gpu paths=server
