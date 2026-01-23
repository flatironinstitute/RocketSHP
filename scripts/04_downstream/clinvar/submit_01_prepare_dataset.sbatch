#!/usr/bin/bash
#SBATCH --job-name=clinvar_01_prepare
#SBATCH -p ccb
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/clinvar_%x.%j.out
#SBATCH --error=slurm_logs/clinvar_%x.%j.err

# This script prepares the ClinVar dataset:
# - Parses VCF
# - Maps variants to proteins
# - Downloads AlphaFold structures
# - Applies quality filters

module load uv

cd $HOME/Projects/rocketshp

# Set cache directories to accessible locations
export PYENSEMBL_CACHE_DIR=/mnt/home/ssledzieski/.cache/pyensembl
export MPLCONFIGDIR=/mnt/home/ssledzieski/.cache/matplotlib

# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Run data preparation
uv run python scripts/04_downstream/clinvar/01_prepare_clinvar_dataset.py \
    configs/clinvar_analysis_config.yml

echo "Step 1 complete: Dataset prepared"
