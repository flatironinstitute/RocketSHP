#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 
#SBATCH --gpus=1
#SBATCH --constraint=a100  # if you want a particular type of GPU
#SBATCH --time=2-00:00
#SBATCH --output=output_%j.out
#SBATCH --error=output_%j.err

export MPLCONFIGDIR=/tmp/matplotlib-cache
module load cuda
source $HOME/venvs/openmm-env/bin/activate
cd $HOME/rocketshp

python scripts/04_downstream/protein_in_water.py \
    --pdb-path $1 \
    --output-dir data/processed/relaxdb_sims/$2 \
    --run-id $2 \
    --time 100
    -d CUDA