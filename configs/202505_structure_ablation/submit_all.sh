cd ~/Projects/rocketshp
RSHP_CONFIG=configs/202505_structure_ablation/seq_only.yml sbatch --job-name="202505_struct_ablation_seq_only" scripts/02_train/submit_training.sbatch
RSHP_CONFIG=configs/202505_structure_ablation/struct_encoded.yml sbatch --job-name="202505_struct_ablation_struct_encoded" scripts/02_train/submit_training.sbatch
RSHP_CONFIG=configs/202505_structure_ablation/struct_prequantized.yml sbatch --job-name="202505_struct_ablation_struct_prequantized" scripts/02_train/submit_training.sbatch
RSHP_CONFIG=configs/202505_structure_ablation/struct_quantized.yml sbatch --job-name="202505_struct_ablation_struct_quantized" scripts/02_train/submit_training.sbatch
RSHP_CONFIG=configs/202505_structure_ablation/struct_ramachandran.yml sbatch --job-name="202505_struct_ablation_struct_ramachandran" scripts/02_train/submit_training.sbatch


