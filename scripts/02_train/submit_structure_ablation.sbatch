#!/usr/bin/bash
#SBATCH --job-name=struct_ramachandran
#SBATCH -p gpu
#SBATCH --constraint=a100-40gb  # if you want a particular type of GPU
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

module load python/3.11.7
module load cudnn/8.9.7.29-12
module load nccl/2.20.3-1
module load gcc
source ~/venvs/scc-rocketshp/bin/activate

cd $HOME/Projects/rocketshp
rocketshp_train $SLURM_JOB_NAME --config configs/ablate_structure_encodings/${SLURM_JOB_NAME}.yml
