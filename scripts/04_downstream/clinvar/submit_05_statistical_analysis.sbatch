#!/usr/bin/bash
#SBATCH --job-name=clinvar_05_stats
#SBATCH -p ccb
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/clinvar_%x.%j.out
#SBATCH --error=slurm_logs/clinvar_%x.%j.err

# This script performs statistical analysis
# - Mann-Whitney U tests
# - Logistic regression
# - ROC/AUC analysis

module load uv

cd $HOME/Projects/rocketshp

# Set cache directories to accessible locations
export PYENSEMBL_CACHE_DIR=/mnt/home/ssledzieski/.cache/pyensembl
export MPLCONFIGDIR=/mnt/home/ssledzieski/.cache/matplotlib

# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Run statistical analysis
uv run python scripts/04_downstream/clinvar/05_statistical_analysis.py \
    configs/clinvar_analysis_config.yml

echo "Step 5 complete: Statistical analysis done"
