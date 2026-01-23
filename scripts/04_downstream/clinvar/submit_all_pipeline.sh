#!/bin/bash
# Master submission script for ClinVar pathogenicity analysis pipeline
# This script submits all 7 steps with job dependencies

set -e  # Exit on error

cd $HOME/Projects/rocketshp/scripts/04_downstream/clinvar

echo "=========================================="
echo "ClinVar Pathogenicity Analysis Pipeline"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p ../../../slurm_logs

# Submit Step 1: Data Preparation
echo "Submitting Step 1: Data Preparation..."
JOB1=$(sbatch --parsable submit_01_prepare_dataset.sbatch)
echo "  Job ID: $JOB1"

# Submit Step 2: WT Predictions (depends on Step 1)
echo "Submitting Step 2: Wild-type Predictions..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 submit_02_predict_wildtype.sbatch)
echo "  Job ID: $JOB2 (depends on $JOB1)"

# Submit Step 3: Variant Predictions (depends on Step 1)
echo "Submitting Step 3: Variant Predictions..."
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 submit_03_predict_variants.sbatch)
echo "  Job ID: $JOB3 (depends on $JOB1)"

# Submit Step 4: Feature Extraction (depends on Steps 2 & 3)
echo "Submitting Step 4: Feature Extraction..."
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2:$JOB3 submit_04_extract_features.sbatch)
echo "  Job ID: $JOB4 (depends on $JOB2 and $JOB3)"

# Submit Step 5: Statistical Analysis (depends on Step 4)
echo "Submitting Step 5: Statistical Analysis..."
JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 submit_05_statistical_analysis.sbatch)
echo "  Job ID: $JOB5 (depends on $JOB4)"

# Submit Step 6: Visualization (depends on Step 5)
echo "Submitting Step 6: Visualization..."
JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 submit_06_visualize.sbatch)
echo "  Job ID: $JOB6 (depends on $JOB5)"

# Submit Step 7: Report Generation (depends on Step 6)
echo "Submitting Step 7: Report Generation..."
JOB7=$(sbatch --parsable --dependency=afterok:$JOB6 submit_07_generate_report.sbatch)
echo "  Job ID: $JOB7 (depends on $JOB6)"

echo ""
echo "=========================================="
echo "All jobs submitted successfully!"
echo "=========================================="
echo ""
echo "Job chain:"
echo "  $JOB1 → [$JOB2, $JOB3] → $JOB4 → $JOB5 → $JOB6 → $JOB7"
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo ""
echo "View logs in:"
echo "  $HOME/Projects/rocketshp/slurm_logs/"
echo ""
echo "Final report will be at:"
echo "  $HOME/Projects/rocketshp/reports/figures/clinvar/clinvar_pathogenicity_report.html"
echo ""
