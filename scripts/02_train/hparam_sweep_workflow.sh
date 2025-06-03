#!/bin/bash
# filepath: /mnt/home/ssledzieski/Projects/rocketshp/scripts/02_train/hparam_sweep_workflow.sh

# Complete hyperparameter sweep workflow
cd ~/Projects/rocketshp

# SWEEP_NAME="architecture_sweep_$(date +%Y%m%d_%H%M%S)"
SWEEP_NAME="loss_alpha_sweep_$(date +%Y%m%d_%H%M%S)"
SWEEP_FILE="sweeps/${SWEEP_NAME}.json"

echo "=== RocketSHP Hyperparameter Sweep ==="
echo "Sweep name: $SWEEP_NAME"

# 1. Generate sweep configurations
echo "Generating sweep configurations..."
mkdir -p sweeps
python -m rocketshp.interface.train_sweep generate-sweep \
    configs/sweep_loss_weights.yml \
    $SWEEP_FILE \
    --count 50 \
    --name $SWEEP_NAME

# 2. Test locally (optional)
# echo "Running 3 jobs locally for testing..."
# python -m rocketshp.interface.train_sweep run-local-sweep \
#     $SWEEP_FILE \
#     --max-jobs 3 \
#     --dataset atlas \
#     --config configs/20250427_large.yml

# 3. Submit to SLURM
echo "Submitting to SLURM..."
export SWEEP_FILE=$SWEEP_FILE
sbatch scripts/02_train/run_hparam_sweep.sbatch

echo "Sweep submitted"
echo "Monitor jobs with: squeue -u $USER"
echo "Analyze results with: python -m rocketshp.interface.train_sweep analyze-sweep $SWEEP_NAME configs/sweep_loss_weights.yml --new-config configs/loss_weights_optimal.yml" --output-table $SWEEP_NAME"_results.csv"