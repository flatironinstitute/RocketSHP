#!/usr/bin/bash
#SBATCH --job-name=planet_md_sweep
#SBATCH -p gpu
#SBATCH --constraint=a100-40gb
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --array=0-49  # Adjust based on number of configs
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

module load python/3.11.7
module load cudnn/8.9.7.29-12
module load nccl/2.20.3-1
module load gcc
PLANET_MD_DIR=${PLANET_MD_DIR:-$HOME/PLANET-MD}
PLANET_MD_VENV=${PLANET_MD_VENV:-$PLANET_MD_DIR/.venv}
cd "$PLANET_MD_DIR"
source "$PLANET_MD_VENV/bin/activate"

# Each array job runs one configuration
python -m planet_md.interface.train_sweep run-sweep-from-file \
    $SWEEP_FILE \
    --job-index $SLURM_ARRAY_TASK_ID \
    --dataset atlas \
    --config configs/architecture_sweep_20250528_135738_best.yaml \
    # --config configs/20250427_large.yml