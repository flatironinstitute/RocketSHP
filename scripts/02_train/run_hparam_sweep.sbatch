#!/usr/bin/bash
# filepath: /mnt/home/ssledzieski/Projects/rocketshp/scripts/02_train/run_hparam_sweep.sbatch
#SBATCH --job-name=rocketshp_sweep
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
source ~/venvs/rocketshp/bin/activate

cd $HOME/Projects/rocketshp

# Each array job runs one configuration
python -m rocketshp.interface.train_sweep run-sweep-from-file \
    $SWEEP_FILE \
    --job-index $SLURM_ARRAY_TASK_ID \
    --dataset atlas \
    --config configs/architecture_sweep_20250528_135738_best.yaml \
    # --config configs/20250427_large.yml