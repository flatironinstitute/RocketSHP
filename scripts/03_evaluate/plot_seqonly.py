#%%
from rocketshp import config
import pandas as pd

struct_results_dir = config.EVALUATION_DATA_DIR / "evaluations" / "fs_shp_pred_kl_reweighted2"
seq_results_dir = config.EVALUATION_DATA_DIR / "evaluations" / "fs_shp_pred_kl_seqonly"

struct_results = pd.read_csv(struct_results_dir / "results.csv").set_index("pdb_code")
seq_results = pd.read_csv(seq_results_dir / "results.csv").set_index("pdb_code")

seq_results.columns = [f"seq_{i}" for i in seq_results.columns]
all_results = pd.merge(struct_results, seq_results, left_index=True, right_index=True)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(3,1, figsize=(5, 10))

ax[0].scatter(all_results["spearman"], all_results["seq_spearman"], s= 4)
ax[0].set_xlabel("Structure+Sequence RMSF Spearman Correlation")
ax[0].set_ylabel("Sequence Only\nRMSF Spearman Correlation")
ax[0].plot([0,1],[0,1], color="gray", linestyle="--")

ax[1].scatter(all_results["autocorr_mse"], all_results["seq_autocorr_mse"], s= 4)
ax[1].set_xlabel("Structure+Sequence Autocorrelation MSE")
ax[1].set_ylabel("Sequence Only\nAutocorrelation MSE")
xmax = all_results["autocorr_mse"].max()
ymax = all_results["seq_autocorr_mse"].max()
ax[1].plot([0,xmax],[0,xmax], color="gray", linestyle="--")

ax[2].scatter(all_results["shp_kl"], all_results["seq_shp_kl"], s= 4)
ax[2].set_xlabel("Structure+Sequence SHP KL Divergence")
ax[2].set_ylabel("Sequence Only\nSHP KL Divergence")
xmax = all_results["shp_kl"].max()
ax[2].plot([0,xmax],[0,xmax], color="gray", linestyle="--")

sns.despine()
plt.savefig(config.PROJ_ROOT / "img" / "seqonly_vs_structseq.svg", bbox_inches="tight")
plt.show()
# %%
