module load gcc
source /mnt/home/ssledzieski/venvs/openmm-env/bin/activate
cd /mnt/home/ssledzieski/Projects/rocketshp
sbatch scripts/04_downstream/piw_slurm.sh ~/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/11080/11080_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb 11080
sbatch scripts/04_downstream/piw_slurm.sh ~/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/BLOT5/BLOT5_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb BLOT5
sbatch scripts/04_downstream/piw_slurm.sh ~/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/CRABP2/CRABP2_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb CRABP2
sbatch scripts/04_downstream/piw_slurm.sh ~/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/RNASE/RNASE_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb RNASE
sbatch scripts/04_downstream/piw_slurm.sh ~/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/CHEA/CHEA_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb CHEA
sbatch scripts/04_downstream/piw_slurm.sh ~/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/SPOOF/SPOOF_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb SPOOF
