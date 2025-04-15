#%% Imports
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
import mdtraj as md
from rocketshp.trajectory import compute_rmsf
from rocketshp import config
from tqdm import tqdm

plt.rcParams.update({
    "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])",
    # "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#1D2954', '#FFD53E'])",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 16,
    "figure.autolayout": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "svg.fonttype": "none",
    })
#%% Paths

reference_traj_root = Path("/mnt/home/ssledzieski/Projects/rocketshp/data/raw/atlas")
dyna_results_root = Path("/mnt/home/ssledzieski/GitHub/Dyna-1/rshp_results/")
bioemu_results_root = Path("/mnt/home/ssledzieski/GitHub/bioemu/rshp_results/")
rshp_results_pickle = config.EVALUATION_DATA_DIR / "evaluations/fs_shp_pred_kl_reweighted2/inference_results.pkl"

assert reference_traj_root.exists(), f"Reference trajectory root not found: {reference_traj_root}"
assert rshp_results_pickle.exists(), f"RocketSHP results pickle not found: {rshp_results_pickle}"
assert dyna_results_root.exists(), f"Dyna results root not found: {dyna_results_root}"
assert bioemu_results_root.exists(), f"BioEMU results root not found: {bioemu_results_root}"

#%% Load RocketSHP results
with open(rshp_results_pickle, "rb") as f:
    rshp_results = pk.load(f)
rshp_results = {k.split("/")[0]: v for k, v in rshp_results.items() if k.endswith("R1")}
rshp_rmsf = {k: v["rmsf"].numpy() for k, v in rshp_results.items()}

#%% Load Dyna results
dyna_results = {}
for k in tqdm(rshp_results.keys()):
    dyna_path = dyna_results_root / k / f"{k}-Dyna1.csv"
    assert dyna_path.exists(), f"Dyna results not found: {dyna_path}"
    dyna_df = pd.read_csv(dyna_path)
    dyna_probability = dyna_df.iloc[:, 2].values
    dyna_results[k] = dyna_probability 

# %% Load BioEMU results
bioemu_results = {}
bioemu_100_results = {}
for k in tqdm(rshp_results.keys()):
        
    bioemu_path = bioemu_results_root / f"{k}_10"
    assert bioemu_path.exists(), f"BioEMU results not found: {bioemu_path}"
    bioemu_traj = md.load(f"{bioemu_path}/samples.xtc", top=f"{bioemu_path}/topology.pdb")
    bioemu_rmsf = compute_rmsf(bioemu_traj)
    bioemu_results[k] = bioemu_rmsf

    bioemu_100_path = bioemu_results_root.parent / "rshp_results_100" / f"{k}_100"
    assert bioemu_100_path.exists(), f"BioEMU 100 results not found: {bioemu_100_path}"
    bioemu_100_traj = md.load(f"{bioemu_100_path}/samples.xtc", top=f"{bioemu_100_path}/topology.pdb")
    bioemu_100_rmsf = compute_rmsf(bioemu_100_traj)
    bioemu_100_results[k] = bioemu_100_rmsf

# %% Load reference trajectory
reference_rmsf = {}
system_sizes = {}

for k in tqdm(rshp_results.keys()):
    reference_xtc = reference_traj_root / k[:2] / f"{k}_prod_R1_fit.xtc"
    reference_top = reference_traj_root / k[:2] / f"{k}.pdb"
    reference_traj = md.load(reference_xtc, top=reference_top)
    system_sizes[k] = reference_traj.n_residues
    reference_rmsf[k] = compute_rmsf(reference_traj)

# %% Rescale dyna from probability to RMSF (not really how it works, but for comparison)
all_predicted = []
all_calibration = []
for system in ["11080", "BLOT5", "CHEA", "RNASE", "CRABP2", "SPOOF"]:
    dyna_calibration_sim = f"/mnt/home/ssledzieski/Projects/rocketshp/data/processed/relaxdb_sims/{system}"
    dyna_calibration_pred = pd.read_csv(f"/mnt/home/ssledzieski/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/{system}/{system}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000-Dyna1.csv").p_exchange
    dyna_calibration_traj = md.load(f"{dyna_calibration_sim}/{system}_traj.xtc", top=f"{dyna_calibration_sim}/{system}_top.pdb")

    dyna_calibration_rmsf = compute_rmsf(dyna_calibration_traj)

    all_predicted.append(dyna_calibration_pred)
    all_calibration.append(dyna_calibration_rmsf)
all_predicted = np.concatenate(all_predicted)
all_calibration = np.concatenate(all_calibration)


# plt.plot(dyna_calibration_rmsf, label="Dyna RMSF", c="grey")
# Reshape for sklearn (needs 2D array)
X = all_predicted.reshape(-1, 1)
y = all_calibration

# Fit linear model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# Get parameters
calibration_scale = model.coef_[0]  # Scaling factor
calibration_offset = model.intercept_  # Offset
print(f"Scaling factor: {calibration_scale}, Offset: {calibration_offset}")

#%% Plot calibrated data
system = "CRABP2"
dyna_calibration_sim = f"/mnt/home/ssledzieski/Projects/rocketshp/data/processed/relaxdb_sims/{system}"
dyna_calibration_pred = pd.read_csv(f"/mnt/home/ssledzieski/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/{system}/{system}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000-Dyna1.csv").p_exchange
calibration_scaled = calibration_scale * dyna_calibration_pred + calibration_offset

dyna_calibration_traj = md.load(f"{dyna_calibration_sim}/{system}_traj.xtc", top=f"{dyna_calibration_sim}/{system}_top.pdb")
dyna_calibration_rmsf = compute_rmsf(dyna_calibration_traj)

rxdb = pd.read_json("~/GitHub/Dyna-1/data/RelaxDB_datasets/RelaxDB_with_other_metrics_22jan2025.json")
rxdb_label = rxdb.loc[f"{system}"]["label"]

rxdb_map = {
    "p": ".",
    ".": ".",
    "x": "-",
    "v": "v",
    "^": "^",
    "b": "^",
    "A": "-"
}

plt.figure(figsize=(10, 6))
plt.plot(dyna_calibration_pred, label="Dyna Pred")
plt.plot(calibration_scaled, label="Dyna Pred (scaled)")
for i, c in enumerate(rxdb_label):
    c = rxdb_map.get(c, c)
    plt.annotate(c, xy=(i, dyna_calibration_rmsf[i]), xytext=(i, dyna_calibration_rmsf[i]), ha="center", fontsize=10)
plt.ylim(0, 1)
plt.legend()
plt.show()

# %% Calibrate RMSF
dyna_rmsf = {}
for k in tqdm(rshp_results.keys()):
    dyna_probability = dyna_results[k]
    # min max scale probability based on observed calibration rmsf
    dyna_scaled = (calibration_scale * dyna_probability) + calibration_offset
    dyna_rmsf[k] = dyna_scaled

# %% Compare RMSF for all methods
methods = {
    "RocketSHP": rshp_rmsf,
    "Dyna-1": dyna_results,
    "Dyna-1 (Calibrated)": dyna_rmsf,
    "BioEMU (10)": bioemu_results,
    "BioEMU (100)": bioemu_100_results,
    "Reference": reference_rmsf
}

rmsf_results = {}
for k in tqdm(rshp_results.keys()):
    rmsf_results[k] = {m: methods[m][k] for m in methods.keys()}
rmsf_df = pd.DataFrame(rmsf_results).T
rmsf_df = rmsf_df.rename_axis("System").reset_index()
rmsf_df = rmsf_df.melt(id_vars=["System"], var_name="Method", value_name="RMSF")

# %% Plot a particular system
# system = "7fd1_A"
system = "4ayg_B"
# system = "1ab1_A"
# system = "1tzw_A"
fig, ax = plt.subplots(figsize=(8, 6))

for method in methods:
    ax.plot(rmsf_df[rmsf_df["System"] == system].loc[rmsf_df["Method"] == method, "RMSF"].values[0], label=method)

ax.set_title(f"{system} ({system_sizes[system]} residues)")
ax.set_xlabel("Residue Index")
ax.set_ylabel("RMSF (Angstroms)")
ax.legend(title="Method", loc="upper right")
plt.tight_layout()
# plt.savefig(config.FIGURES_DIR / f"rmsf_comparison_{system}.svg")
plt.show()

# %% Compute MSE for all systems
mean_sq_error = {}
for method in methods:
    if method == "Reference":
        continue
    mean_sq_error[method] = []
    for k in tqdm(rmsf_results.keys()):
        mse = np.mean((rmsf_results[k][method] - rmsf_results[k]["Reference"]) ** 2)
        mean_sq_error[method].append(mse)
mean_sq_error_df = pd.DataFrame(mean_sq_error)
mean_sq_error_df = mean_sq_error_df.rename_axis("System").reset_index()
mean_sq_error_df = mean_sq_error_df.melt(id_vars=["System"], var_name="Method", value_name="MSE")
# mean_sq_error_df = mean_sq_error_df.groupby("Method").mean().reset_index()
# mean_sq_error_df = mean_sq_error_df.sort_values("MSE", ascending=False)

# %% Plot MSE
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=mean_sq_error_df, x="Method", y="MSE", ax=ax,
            order=["RocketSHP", "Dyna-1 (Calibrated)", "Dyna-1", "BioEMU (100)", "BioEMU (10)"]
            )
ax.set_title("Mean Squared Error (MSE) of RMSF")
ax.set_xlabel("Method")
ax.set_ylabel("MSE")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
# plt.savefig(config.FIGURES_DIR / "rmsf_mse_comparison.svg")
plt.show()

# %% Plot MSE in several bins by system size
mean_sq_error_by_size = []
for method in methods:
    if method == "Reference":
        continue
    for k in tqdm(rmsf_results.keys()):
        mse = np.mean((rmsf_results[k][method] - rmsf_results[k]["Reference"]) ** 2)
        mean_sq_error_by_size.append([k, system_sizes[k], method, mse])

mean_sq_error_by_size_df = pd.DataFrame(mean_sq_error_by_size)
mean_sq_error_by_size_df.columns = ["System", "Size", "Method", "MSE"]

# assign size group by quartiles
mean_sq_error_by_size_df["Size Group"] = pd.qcut(mean_sq_error_by_size_df["Size"], q=4, labels=["Small", "Medium", "Large", "Extra Large"])
# # update labels with exact quantile definitions -- use qcut to get quantile cutoffs then assign them as labels
quartile_boundaries = mean_sq_error_by_size_df["Size"].quantile([0, 0.25, 0.5, 0.75, 1]).values
# # size group labels should be the min and max of each bin
size_group_labels = []
for i in range(len(quartile_boundaries) - 1):
    size_group_labels.append(f"{int(quartile_boundaries[i])} - {int(quartile_boundaries[i + 1])}")
mean_sq_error_by_size_df["Size Group"] = mean_sq_error_by_size_df["Size Group"].cat.rename_categories(size_group_labels)

# %% Plot MSE by Method and Size Group
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(
    data=mean_sq_error_by_size_df,
    x="Method",
    y="MSE",
    hue="Size Group",
    ax=ax,
    order=["RocketSHP", "Dyna-1 (Calibrated)", "Dyna-1", "BioEMU (100)", "BioEMU (10)"]
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.show()

# mean_sq_error_by_size_df = mean_sq_error_by_size_df.drop(columns=["System"]).groupby(["Method", "Size Group"]).mean().reset_index()
# mean_sq_error_by_size_df = mean_sq_error_by_size_df.sort_values("MSE", ascending=False)

#%% Plot MSE by size group then method
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=mean_sq_error_by_size_df, x="Size Group", y="MSE", hue="Method", ax=ax)
ax.set_title("Mean Squared Error (MSE) of RMSF by System Size")
ax.set_xlabel("Number of Amino Acids")
ax.set_ylabel("MSE")
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "rmsf_mse_comparison_by_size.svg")
plt.show()


# %%
