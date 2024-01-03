#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=high_sierras_embed_birdnet
#SBATCH --array=1-5%5
date;hostname;pwd
source /mnt/stud/home/mrichert/.zshrc
conda activate 5gadme
cd /mnt/stud/home/mrichert/Projects/GADME-BaselineResults-BA/src
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
srun python -u main.py \
experiment= high_sierras_embed_birdnet_base \
trainer= single_gpu \
paths=server
