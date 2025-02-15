#!/usr/bin/bash
#SBATCH --job-name=mdcath_hfds_j7
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
# module load openmm
source ~/venvs/scc-rocketshp/bin/activate

cd $HOME/Projects/rocketshp
# python scripts/01_preprocess/mdcath/process_mdcath_cpu.py 0 16200
# python scripts/01_preprocess/mdcath/process_mdcath_cpu.py 16200 32400
# python scripts/01_preprocess/mdcath/process_mdcath_cpu.py 32400 48600
# python scripts/01_preprocess/mdcath/process_mdcath_cpu.py 48600 64800
# python scripts/01_preprocess/mdcath/process_mdcath_cpu.py 64800 81000
# python scripts/01_preprocess/mdcath/process_mdcath_cpu.py 81000 97200
# python scripts/01_preprocess/mdcath/process_mdcath_cpu.py 97200 113400
# python scripts/01_preprocess/mdcath/process_mdcath_cpu.py 113400 129600
