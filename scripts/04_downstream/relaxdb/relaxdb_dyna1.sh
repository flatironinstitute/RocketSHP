module load gcc
cd ~/GitHub/Dyna-1
mamba activate dyna1
python dyna1.py --pdb data/RelaxDB_datasets/output_structures/11080/11080_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb --use_pdb_seq --save_dir data/RelaxDB_datasets/output_structures/11080
python dyna1.py --pdb data/RelaxDB_datasets/output_structures/BLOT5/BLOT5_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb --use_pdb_seq --save_dir data/RelaxDB_datasets/output_structures/BLOT5
python dyna1.py --pdb data/RelaxDB_datasets/output_structures/CHEA/CHEA_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb --use_pdb_seq --save_dir data/RelaxDB_datasets/output_structures/CHEA
python dyna1.py --pdb data/RelaxDB_datasets/output_structures/RNASE/RNASE_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb --use_pdb_seq --save_dir data/RelaxDB_datasets/output_structures/RNASE
python dyna1.py --pdb data/RelaxDB_datasets/output_structures/CRABP2/CRABP2_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb --use_pdb_seq --save_dir data/RelaxDB_datasets/output_structures/CRABP2
python dyna1.py --pdb data/RelaxDB_datasets/output_structures/SPOOF/SPOOF_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb --use_pdb_seq --save_dir data/RelaxDB_datasets/output_structures/SPOOF
