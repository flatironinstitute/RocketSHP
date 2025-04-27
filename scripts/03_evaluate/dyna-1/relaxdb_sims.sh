module load gcc
source /mnt/home/ssledzieski/venvs/openmm-env/bin/activate
cd /mnt/home/ssledzieski/Projects/rocketshp
sbatch scripts/04_downstream/simulate_protein.sh ~/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/11080/11080_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb data/processed/relaxdb_sims/11080 100
sbatch scripts/04_downstream/simulate_protein.sh ~/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/BLOT5/BLOT5_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb data/processed/relaxdb_sims/BLOT5 100
sbatch scripts/04_downstream/simulate_protein.sh ~/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/CRABP2/CRABP2_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb data/processed/relaxdb_sims/CRABP2 100
sbatch scripts/04_downstream/simulate_protein.sh ~/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/RNASE/RNASE_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb data/processed/relaxdb_sims/RNASE 100
sbatch scripts/04_downstream/simulate_protein.sh ~/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/CHEA/CHEA_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb data/processed/relaxdb_sims/CHEA 100
sbatch scripts/04_downstream/simulate_protein.sh ~/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/SPOOF/SPOOF_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb data/processed/relaxdb_sims/SPOOF 100
sbatch scripts/04_downstream/simulate_protein.sh ~/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/27888/27888_0_unrelaxed_rank_004_alphafold2_ptm_model_1_seed_000.pdb data/processed/relaxdb_sims/27888 100 # beta lactamase
sbatch scripts/04_downstream/simulate_protein.sh ~/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/ACRIIA4/ACRIIA4_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb data/processed/relaxdb_sims/ACRIIA4 100 # cas 9 inhibitor acriia4

# longer simulations of cas 9 inhibitor
sbatch scripts/04_downstream/simulate_protein.sh ~/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/ACRIIA4/ACRIIA4_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb ACRIIA4_2us 2000 # cas 9 inhibitor acriia4, 2us