#!/usr/bin/bash
#SBATCH --job-name=clinvar_04_features
#SBATCH -p ccb
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/clinvar_%x.%j.out
#SBATCH --error=slurm_logs/clinvar_%x.%j.err

# This script extracts features from WT and variant predictions
# - Loads predictions from HDF5 files
# - Extracts per-position and global features
# - Computes deltas
# - Adds covariates

module load uv

cd $HOME/Projects/rocketshp

# Set cache directories to accessible locations
export PYENSEMBL_CACHE_DIR=/mnt/home/ssledzieski/.cache/pyensembl
export MPLCONFIGDIR=/mnt/home/ssledzieski/.cache/matplotlib

# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Run feature extraction
uv run python scripts/04_downstream/clinvar/04_extract_features.py \
    configs/clinvar_analysis_config.yml

echo "Step 4 complete: Features extracted"
