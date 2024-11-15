#!/usr/bin/bash
id=$1
module purge
module load python cuda cudnn nccl
source ~/venvs/scc-rocketshp/bin/activate
module load foldseek

cd ~/Projects/rocketshp/
python rocketshp/mdcath.py convert-foldseek ${id} > data/interim/${id}_mdc_fs.out 2> data/interim/${id}_mdc_fs.err
