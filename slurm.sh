#!/bin/bash
#SBATCH --job-name=high_sierras_embed_birdnet
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=dev

date;hostname;pwd
source /mnt/stud/home/mrichert/.zshrc
conda activate gadme
cd /mnt/stud/home/mrichert/Projects/GADME-BaselineResults-BA/src
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
srun python -u main.py \
experiment=high_sierras_embed_birdnet_base \
trainer=single_gpu \
paths=server
