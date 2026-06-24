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
PLANET_MD_DIR=${PLANET_MD_DIR:-$HOME/PLANET-MD}
PLANET_MD_VENV=${PLANET_MD_VENV:-$PLANET_MD_DIR/.venv}
cd "$PLANET_MD_DIR"
source "$PLANET_MD_VENV/bin/activate"
python scripts/01_preprocess/atlas/process_atlas_cpu.py 3000 4000
# python scripts/01_preprocess/atlas/process_atlas_shp.py 2000 5000
