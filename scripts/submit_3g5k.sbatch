#!/usr/bin/bash
#SBATCH --job-name=submit_3g5k_slurm_rep1
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --constraint=a100-40gb  # if you want a particular type of GPU
#SBATCH --time=2-00:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

module load python
module load cudnn
module load nccl
module load cuda
module load openmm
source ~/venvs/scc-rocketshp/bin/activate

cd $HOME/Projects/rocketshp
REPLICATE=1
python scripts/simulate_3g5k.py $REPLICATE
