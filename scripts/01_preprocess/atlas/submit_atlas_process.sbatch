#!/usr/bin/bash
#SBATCH --job-name=process_atlas_huggingface
#SBATCH -p gen
#SBATCH --cpus-per-task=72
#SBATCH --ntasks=1
#SBATCH --mem=900G
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
python scripts/process_atlas_cpu.py
