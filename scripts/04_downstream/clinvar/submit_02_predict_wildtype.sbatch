#!/usr/bin/bash
#SBATCH --job-name=clinvar_02_wt_predict
#SBATCH -p gpu
#SBATCH --constraint=a100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/clinvar_%x.%j.out
#SBATCH --error=slurm_logs/clinvar_%x.%j.err

# This script generates RocketSHP predictions for wild-type proteins
# - Loads AlphaFold structures
# - Runs RocketSHP inference
# - Builds allosteric networks
# - Calculates centrality metrics

module load uv
module load cuda

cd $HOME/Projects/rocketshp

# Set cache directories to accessible locations
export PYENSEMBL_CACHE_DIR=/mnt/home/ssledzieski/.cache/pyensembl
export MPLCONFIGDIR=/mnt/home/ssledzieski/.cache/matplotlib

# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Run wild-type predictions
uv run python scripts/04_downstream/clinvar/02_predict_wildtype.py \
    configs/clinvar_analysis_config.yml

echo "Step 2 complete: Wild-type predictions generated"
