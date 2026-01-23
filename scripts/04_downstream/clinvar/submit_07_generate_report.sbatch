#!/usr/bin/bash
#SBATCH --job-name=clinvar_07_report
#SBATCH -p ccb
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/clinvar_%x.%j.out
#SBATCH --error=slurm_logs/clinvar_%x.%j.err

# This script generates the final HTML report
# - Compiles all results
# - Creates summary tables
# - Generates HTML output

module load uv

cd $HOME/Projects/rocketshp

# Set cache directories to accessible locations
export PYENSEMBL_CACHE_DIR=/mnt/home/ssledzieski/.cache/pyensembl
export MPLCONFIGDIR=/mnt/home/ssledzieski/.cache/matplotlib

# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Run report generation
uv run python scripts/04_downstream/clinvar/07_generate_report.py \
    configs/clinvar_analysis_config.yml

echo "Step 7 complete: Report generated"
echo "View report at: reports/figures/clinvar/clinvar_pathogenicity_report.html"
