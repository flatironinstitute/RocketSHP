#!/bin/bash
# filepath: /mnt/home/ssledzieski/Projects/rocketshp/scripts/02_train/create_hparam_sweep.sh
#!/bin/bash

# Create and run a hyperparameter sweep
cd ~/Projects/rocketshp

# Create the sweep
echo "Creating sweep..."
SWEEP_ID=$(python -m rocketshp.interface.train_sweep create-sweep \
    --sweep-config configs/sweep_basic.yml \
    --count 50 \
    --name "basic_arch_sweep_$(date +%Y%m%d_%H%M%S)")

echo "Created sweep: $SWEEP_ID"

# Submit SLURM jobs to run the sweep
echo "Submitting SLURM jobs..."
export SWEEP_ID=$SWEEP_ID
sbatch scripts/sweep/submit_sweep_jobs.sbatch

echo "Sweep jobs submitted!"
echo "Monitor progress at: https://app.neptune.ai"
echo "Analyze results with: python -m rocketshp.interface.train_sweep analyze-sweep $SWEEP_ID"