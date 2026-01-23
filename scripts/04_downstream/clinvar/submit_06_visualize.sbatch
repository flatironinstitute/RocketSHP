#!/usr/bin/bash
#SBATCH --job-name=clinvar_06_visualize
#SBATCH -p ccb
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/clinvar_%x.%j.out
#SBATCH --error=slurm_logs/clinvar_%x.%j.err

# This script creates visualizations
# - Distribution plots
# - ROC curves
# - Feature importance
# - Network examples

module load uv

cd $HOME/Projects/rocketshp

# Set cache directories to accessible locations
export PYENSEMBL_CACHE_DIR=/mnt/home/ssledzieski/.cache/pyensembl
export MPLCONFIGDIR=/mnt/home/ssledzieski/.cache/matplotlib

# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Run visualization
uv run python scripts/04_downstream/clinvar/06_visualize_results.py \
    configs/clinvar_analysis_config.yml

echo "Step 6 complete: Visualizations created"
