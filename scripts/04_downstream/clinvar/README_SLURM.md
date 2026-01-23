# ClinVar Pathogenicity Analysis - SLURM Submission Guide

This directory contains scripts for running the ClinVar pathogenicity analysis pipeline on a SLURM cluster.

## Quick Start: Run Full Pipeline

To submit all 7 steps with automatic dependencies:

```bash
cd scripts/04_downstream/clinvar
./submit_all_pipeline.sh
```

This will submit all jobs in sequence with proper dependencies, so each step waits for its prerequisites to complete.

## Individual Step Submission

You can also submit individual steps manually:

### Step 1: Data Preparation (CPU, ~12 hours)

```bash
sbatch submit_01_prepare_dataset.sbatch
```
- Parses ClinVar VCF
- Maps variants to proteins
- Downloads AlphaFold structures
- Applies quality filters
- **Output:** `data/processed/clinvar/variant_dataset.csv`

### Step 2: Wild-type Predictions (GPU, ~2 days)

```bash
sbatch submit_02_predict_wildtype.sbatch
```
- Generates RocketSHP predictions for WT proteins
- Builds allosteric networks
- Calculates centrality metrics
- **Output:** `data/processed/clinvar/wildtype_predictions.h5`
- **Requires:** Step 1 complete

### Step 3: Variant Predictions (GPU, ~2 days)

```bash
sbatch submit_03_predict_variants.sbatch
```
- Generates RocketSHP predictions for variants
- Uses variant sequences with WT structures
- **Output:** `data/processed/clinvar/variant_predictions.h5`
- **Requires:** Step 1 complete

### Step 4: Feature Extraction (CPU, ~4 hours)

```bash
sbatch submit_04_extract_features.sbatch
```
- Extracts per-position and global features
- Computes WT vs variant deltas
- Adds covariates
- **Output:** `data/processed/clinvar/variant_features.csv`
- **Requires:** Steps 2 & 3 complete

### Step 5: Statistical Analysis (CPU, ~2 hours)

```bash
sbatch submit_05_statistical_analysis.sbatch
```
- Mann-Whitney U tests
- Logistic regression
- ROC/AUC analysis
- **Output:** `data/processed/clinvar/statistical_results.json`
- **Requires:** Step 4 complete

### Step 6: Visualization (CPU, ~2 hours)

```bash
sbatch submit_06_visualize.sbatch
```
- Distribution plots
- ROC curves
- Feature importance plots
- Network topology examples
- **Output:** `reports/figures/clinvar/*.png`
- **Requires:** Step 5 complete

### Step 7: Report Generation (CPU, ~1 hour)

```bash
sbatch submit_07_generate_report.sbatch
```
- Generates comprehensive HTML report
- **Output:** `reports/figures/clinvar/clinvar_pathogenicity_report.html`
- **Requires:** Step 6 complete

## Job Monitoring

### Check job status

```bash
squeue -u $USER
```

### Check specific job

```bash
squeue -j <JOB_ID>
```

### View logs

```bash
# Logs are in slurm_logs/ directory
tail -f slurm_logs/clinvar_*.out
```

### Cancel jobs

```bash
# Cancel single job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Cancel specific job name pattern
scancel -n clinvar_*
```

## Resource Requirements

| Step | Partition | GPUs | CPUs | Memory | Time |
|------|-----------|------|------|--------|------|
| 1. Data Prep | ccb | 0 | 8 | 64G | 12h |
| 2. WT Predict | gpu | 1 | 8 | 128G | 2d |
| 3. Var Predict | gpu | 1 | 8 | 128G | 2d |
| 4. Features | ccb | 0 | 4 | 32G | 4h |
| 5. Stats | ccb | 0 | 4 | 32G | 2h |
| 6. Visualize | ccb | 0 | 4 | 32G | 2h |
| 7. Report | ccb | 0 | 2 | 16G | 1h |

**Total estimated time:** ~4 days (with Steps 2 & 3 running in parallel)

## Restarting from Checkpoint

Steps 2 and 3 support checkpointing. If a job fails or times out:

1. Simply resubmit the same sbatch script
2. The script will automatically resume from where it left off
3. Checkpoints are stored in `data/processed/clinvar/*_checkpoint.txt`

## Troubleshooting

### Job fails immediately

- Check logs in `slurm_logs/clinvar_*.err`
- Ensure `uv` module is available on your cluster
- Ensure `configs/clinvar_analysis_config.yml` exists
- Verify `.venv` exists in project root (run `uv sync` if needed)

### Out of memory

- Increase `--mem` in the sbatch script
- For Steps 2 & 3, try reducing batch size in config

### Job timeout

- Increase `--time` in the sbatch script
- For Steps 2 & 3, resubmit to resume from checkpoint

### GPU not available

- Check available GPU types: `sinfo -p gpu -o "%n %G %C %m"`
- Modify `--constraint` in sbatch script if needed

## Configuration

All parameters are controlled by `configs/clinvar_analysis_config.yml`. Key settings:

- `min_plddt`: Minimum structure quality (default: 70)
- `distance_cutoff`: Network edge cutoff in Å (default: 8.0)
- `device`: GPU device (default: cuda:0)
- `checkpoint_interval`: Save progress every N items (default: 100)

Edit the config file to adjust these parameters.

## Directory Structure

```
scripts/04_downstream/clinvar/
├── submit_01_prepare_dataset.sbatch
├── submit_02_predict_wildtype.sbatch
├── submit_03_predict_variants.sbatch
├── submit_04_extract_features.sbatch
├── submit_05_statistical_analysis.sbatch
├── submit_06_visualize.sbatch
├── submit_07_generate_report.sbatch
├── submit_all_pipeline.sh              # Master submission script
├── 01_prepare_clinvar_dataset.py
├── 02_predict_wildtype.py
├── 03_predict_variants.py
├── 04_extract_features.py
├── 05_statistical_analysis.py
├── 06_visualize_results.py
├── 07_generate_report.py
└── README_SLURM.md                      # This file
```

## Expected Output Files

After successful completion:

```
data/processed/clinvar/
├── variant_dataset.csv                  # Step 1
├── wildtype_predictions.h5              # Step 2
├── variant_predictions.h5               # Step 3
├── variant_features.csv                 # Step 4
├── statistical_results.json             # Step 5
├── wt_predictions_checkpoint.txt        # Checkpoint (Step 2)
└── variant_predictions_checkpoint.txt   # Checkpoint (Step 3)

reports/figures/clinvar/
├── distributions.png                    # Step 6
├── roc_curves.png                       # Step 6
├── feature_importance.png               # Step 6
├── summary_table.png                    # Step 6
├── network_examples/                    # Step 6
│   └── *.png
└── clinvar_pathogenicity_report.html    # Step 7
```

## Additional Notes

- Steps 2 and 3 can run in parallel (they both depend only on Step 1)
- GPU jobs use A100 GPUs by default; modify `--constraint` if needed
- All paths are relative to `$HOME/Projects/rocketshp`
- Scripts use `uv run` for dependency management (`.venv` in project root)
- Ensure dependencies are installed with `uv sync` before running
