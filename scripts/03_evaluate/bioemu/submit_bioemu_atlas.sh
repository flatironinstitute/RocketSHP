#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --constraint=a100-40gb
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=bioemu_atlas
#SBATCH --output=bioemu_atlas_%j.out
#SBATCH --error=bioemu_atlas_%j.err

# Load required modules
module load python/3.11.7
module load cudnn/8.9.7.29-12
module load nccl
module load gcc

# Activate virtual environment
PLANET_MD_DIR=${PLANET_MD_DIR:-$HOME/PLANET-MD}
cd "$PLANET_MD_DIR"
export UV_CACHE_DIR=/tmp/${SLURM_JOB_ID}/uv-cache
export UV_PROJECT_ENVIRONMENT=/tmp/${SLURM_JOB_ID}/venv
uv sync

# Set HuggingFace cache to temporary directory
export HF_HOME=/tmp/$USER/huggingface_cache_${SLURM_JOB_ID}
mkdir -p $HF_HOME

# Set Colabfold cache to temporary directory
export BIOEMU_COLABFOLD_DIR=/tmp/$USER/colabfold_cache_${SLURM_JOB_ID}
mkdir -p $BIOEMU_COLABFOLD_DIR

# Set XDG cache to temporary directory
export XDG_CACHE_HOME=/tmp/$USER/cache_${SLURM_JOB_ID}
mkdir -p $XDG_CACHE_HOME/colabfold/params

# Run bioemu inference with cluster-specific paths
uv run python scripts/03_evaluate/bioemu/bioemu_inference_atlas_cluster.py
