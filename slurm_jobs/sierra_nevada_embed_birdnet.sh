#!/bin/bash
#SBATCH --job-name=sierra_nevada
#SBATCH --output=dl_embed_birdnet_sn.log
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --signal=SIGUSR1@90

date;hostname;pwd
source /mnt/stud/home/mrichert/.zshrc
conda activate gadme
cd /mnt/stud/home/mrichert/Projects/GADME-BaselineResults-BA/src
export HYDRA_FULL_ERROR=1
srun python -u create_embedding_ds.py experiment=high_sierras_dl_embed_birdnet datamodule=sierra_nevada trainer=default paths=server
