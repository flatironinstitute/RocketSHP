#!/usr/bin/bash
#SBATCH --job-name=atlas_hfds_3000_4000
#SBATCH -p genx
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=512GB
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
python scripts/01_preprocess/atlas/process_atlas_cpu.py 3000 4000
# python scripts/01_preprocess/atlas/process_atlas_shp.py 2000 5000
