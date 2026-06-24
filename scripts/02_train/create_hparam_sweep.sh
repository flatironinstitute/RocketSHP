#!/bin/bash

# Create and run a hyperparameter sweep
PLANET_MD_DIR=${PLANET_MD_DIR:-$HOME/PLANET-MD}
cd "$PLANET_MD_DIR"

# Create the sweep
echo "Creating sweep..."
SWEEP_ID=$(python -m planet_md.interface.train_sweep create-sweep \
    --sweep-config configs/sweep_basic.yml \
    --count 50 \
    --name "basic_arch_sweep_$(date +%Y%m%d_%H%M%S)")

echo "Created sweep: $SWEEP_ID"

# Submit SLURM jobs to run the sweep
echo "Submitting SLURM jobs..."
export SWEEP_ID=$SWEEP_ID
sbatch scripts/sweep/submit_sweep_jobs.sbatch

echo "Sweep jobs submitted!"

echo "Analyze results with: python -m planet_md.interface.train_sweep analyze-sweep $SWEEP_ID"
