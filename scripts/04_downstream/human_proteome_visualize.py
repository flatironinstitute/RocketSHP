# %% Imports

from rocketshp import config

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# %%
result_dir = config.PROCESSED_DATA_DIR / "rocketshp_afdb_human"
subdirectories = list(result_dir.iterdir())

# %%

# rocketshp_results = {}
# for subdir in subdirectories:
#     logger.info(f"Loading {subdir.name}")
#     proteins = list(subdir.glob("*.rshp.pt"))
#     for protein_file in tqdm(proteins):
#         pid = protein_file.stem.split("-")[1]
#         rocketshp_results[pid] = torch.load(protein_file, map_location="cpu")

# %%
