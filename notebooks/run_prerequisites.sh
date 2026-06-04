#!/usr/bin/env bash
# Slurm submission commands for all prerequisites of notebooks/paper_figures.ipynb
# Run from the root of the rocketshp repository.
# Estimated total wall time (parallel): ~2–6 hours (bottleneck: script 3)
#
# Usage: bash notebooks/run_prerequisites.sh
# Or submit individually by copying the sbatch calls below.

set -euo pipefail

REPO_ROOT=/mnt/home/ssledzieski/Projects/rocketshp

mkdir -p logs

# ── 1. atlas_validation.py ────────────────────────────────────────────────
# Runs model inference on the full ATLAS val+test set (~1,400 proteins).
# Outputs: data/evaluation/evaluations/large_model_20250427/*_metrics.csv
# Estimated time: 1–2 hours (GPU)
sbatch \
  --job-name=atlas_val \
  --partition=gpu \
  --gpus=1 \
  --constraint=a100 \
  --mem=64G \
  --cpus-per-task=8 \
  --time=4:00:00 \
  --output=logs/plotting_data/atlas_validation_%j.log \
  --wrap="module load cuda && \
    cd $REPO_ROOT && \
    .venv/bin/python scripts/03_evaluate/atlas_validation.py \
    large_model_20250427 \
    $REPO_ROOT/models/big_model/model-epoch=50-val_loss=1.00.pt.ckpt \
    $REPO_ROOT/configs/20250427_large.yml \
    --split test --device cuda:0"

# ── 2. plot_runtime.py ────────────────────────────────────────────────────
# Parses logged runtime .txt files and saves runtime_times.json.
# Outputs: data/evaluation/evaluations/large_model_20250427/precomputed/runtime_times.json
# Estimated time: <5 minutes (CPU, I/O only)
sbatch \
  --job-name=plot_runtime \
  --partition=ccb \
  --mem=8G \
  --cpus-per-task=2 \
  --time=0:10:00 \
  --output=logs/plotting_data/plot_runtime_%j.log \
  --wrap="cd $REPO_ROOT && \
    .venv/bin/python scripts/03_evaluate/plot_runtime.py"

# ── 3. compare_rshp_bioemu_dyna.py ───────────────────────────────────────
# Loads ATLAS reference trajectories and computes RMSF/GCC/IMSD metrics
# against PLANET-MD, Dyna-1, and BioEmu. Trajectory I/O is the bottleneck.
# Outputs: precomputed/{test_rmse_by_size,test_spearman_by_size,test_imsd_by_size}.csv
# Estimated time: 2–6 hours (CPU + high memory)
sbatch \
  --job-name=compare_methods \
  --partition=ccb \
  --mem=128G \
  --cpus-per-task=16 \
  --time=8:00:00 \
  --output=logs/plotting_data/compare_methods_%j.log \
  --wrap="cd $REPO_ROOT && \
    .venv/bin/python scripts/03_evaluate/compare_rshp_bioemu_dyna.py \
    large_model_20250427 --split test"

# ── 4. rshp_relaxdb_infer.py ─────────────────────────────────────────────
# Runs PLANET-MD inference on RelaxDB (~100 proteins) and saves NPZ files.
# Outputs: <relaxdb_root>/rxdb_rocketshp_npz/*.npz
# Estimated time: 15–30 minutes (GPU)
sbatch \
  --job-name=relaxdb_infer \
  --partition=gpu \
  --gpus=1 \
  --constraint=a100 \
  --mem=32G \
  --cpus-per-task=4 \
  --time=1:00:00 \
  --output=logs/plotting_data/relaxdb_infer_%j.log \
  --wrap="module load cuda && \
    cd $REPO_ROOT && \
    .venv/bin/python scripts/04_downstream/relaxDB/rshp_relaxdb_infer.py"

# ── 5. mdcath_temperature.py ──────────────────────────────────────────────
# Runs inference with the MDCath model then evaluates temperature-dependent RMSF.
# Outputs: data/evaluation/evaluations/mdcath_large_ep10/precomputed/
#          temperature_joint_results.csv, temperature_case_study.npz
# Estimated time: 1–2 hours (GPU inference + disk I/O)
sbatch \
  --job-name=mdcath_temp \
  --partition=gpu \
  --gpus=1 \
  --constraint=a100 \
  --mem=64G \
  --cpus-per-task=8 \
  --time=4:00:00 \
  --output=logs/plotting_data/mdcath_temp_%j.log \
  --wrap="module load cuda && \
    cd $REPO_ROOT && \
    .venv/bin/python scripts/03_evaluate/mdcath_temperature.py \
    mdcath_large_ep10 \
    $REPO_ROOT/models/full_model_mdcath_5/model-epoch=10-val_loss=0.80151.pt.ckpt \
    $REPO_ROOT/configs/20250519_mdcath_large.yml \
    --split test --device cuda:0"

# ── 6. weng_KRAS/03_eval_and_plot.py ─────────────────────────────────────
# Builds allosteric networks and computes betweenness centrality for all
# KRAS DMS mutants (~3,200 networks via networkx).
# Outputs: reports/large_model_20250427/precomputed/
#          kras_res_df.csv, kras_folding_ddg.csv, kras_wild_type_centrality.json
# Estimated time: 30 minutes – 2 hours (CPU)
sbatch \
  --job-name=kras_eval \
  --partition=ccb \
  --mem=32G \
  --cpus-per-task=4 \
  --time=2:00:00 \
  --output=logs/plotting_data/kras_eval_%j.log \
  --wrap="cd $REPO_ROOT && \
    .venv/bin/python scripts/04_downstream/weng_KRAS/03_eval_and_plot.py"

# ── 7. weng_KRAS/04_visualize_kras_networks.py ───────────────────────────
# Builds the wild-type masked GCC-LMI network, runs Girvan-Newman clustering
# (k=5), saves portable NPZ + JSON outputs. Also writes TXT and PML for PyMOL.
# Outputs: reports/large_model_20250427/precomputed/
#          kras_wt_masked_network.npz, kras_gcc_lmi_clusters.json
# Requires: mutant_results.pkl (produced by the KRAS inference job)
# Estimated time: 5–15 minutes (CPU, single-threaded networkx)
sbatch \
  --job-name=kras_network \
  --partition=ccb \
  --mem=16G \
  --cpus-per-task=2 \
  --time=0:30:00 \
  --output=logs/plotting_data/kras_network_%j.log \
  --wrap="cd $REPO_ROOT && \
    .venv/bin/python scripts/04_downstream/weng_KRAS/04_visualize_kras_networks.py \
    large_model_20250427"

# ── 8. human_proteome_analyze.py ─────────────────────────────────────────
# Parses uniprot_annotations.csv and aggregates RMSF by region type and
# subcellular location across ~25k human proteome proteins.
# Outputs: data/processed/rocketshp_afdb_human/precomputed/
#          disordered_rmsf.csv, subcellular_location_counts.csv,
#          subcellular_location_rmsf.csv
# Estimated time: 5–15 minutes (CPU, pandas only)
sbatch \
  --job-name=proteome_analyze \
  --partition=ccb \
  --mem=32G \
  --cpus-per-task=4 \
  --time=0:30:00 \
  --output=logs/plotting_data/proteome_analyze_%j.log \
  --wrap="cd $REPO_ROOT && \
    .venv/bin/python scripts/04_downstream/proteome_scale/human_proteome_analyze.py"

echo "All 8 jobs submitted. Monitor with: squeue -u \$USER"
