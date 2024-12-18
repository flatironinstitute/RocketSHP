#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --constraint=a100-80gb  # if you want a particular type of GPU
#SBATCH --time=0-12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

module load python/3.11.7
module load cudnn/8.9.7.29-12
module load nccl/2.20.3-1
source ~/venvs/torch/bin/activate

cd $HOME/Projects/rocketshp
python rocketshp/modeling/train.py $RSHP_NAME --config $RSHP_CONFIG
