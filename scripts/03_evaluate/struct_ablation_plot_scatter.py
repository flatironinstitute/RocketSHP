# %%
import pickle as pk

import matplotlib.pyplot as plt
import pandas as pd

from rocketshp import config

plt.rcParams.update(
    {
        # "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#537eba', '#56B4E9'])",
        "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#1D2954', '#FFD53E'])",  # simons foundation
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 16,
        # "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    }
)

struct_model_key = "struct_encoded"
seq_model_key = "seq_only"

struct_results_dir = config.EVALUATION_DATA_DIR / "evaluations" / struct_model_key
seq_results_dir = config.EVALUATION_DATA_DIR / "evaluations" / seq_model_key

# %%
with open(struct_results_dir / f"{struct_model_key}_test_metrics.pkl", "rb") as f:
    struct_results = pk.load(f)


with open(seq_results_dir / f"{seq_model_key}_test_metrics.pkl", "rb") as f:
    seq_results = pk.load(f)

seq_results.columns = [f"seq_{i}" for i in seq_results.columns]
all_results = pd.merge(struct_results, seq_results, left_index=True, right_index=True)
# %%

SIZE = 8
ALPHA = 0.5

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(all_results["rmsf_mse"], all_results["seq_rmsf_mse"], s=SIZE, alpha=ALPHA)
ax.set_xlabel("Structure+Sequence\nRMSF Root Mean Squared Error")
ax.set_ylabel("Sequence Only\nRMSF Root Mean Squared Error")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.tight_layout()
plt.savefig(
    config.REPORTS_DIR / "figures" / "struct_ablation_rmsf_rmse_scatter.svg",
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    all_results["rmsf_spearman_r"],
    all_results["seq_rmsf_spearman_r"],
    s=SIZE,
    alpha=ALPHA,
)
ax.set_xlabel("Structure+Sequence\nRMSF Spearman Correlation")
ax.set_ylabel("Sequence Only\nRMSF Spearman Correlation")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.tight_layout()
plt.savefig(
    config.REPORTS_DIR / "figures" / "struct_ablation_rmsf_scatter.svg",
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    all_results["gcc_im_dist"], all_results["seq_gcc_im_dist"], s=SIZE, alpha=ALPHA
)
ax.set_xlabel("Structure+Sequence\nGCC-LMI Ipsen-Mikhailov Distance")
ax.set_ylabel("Sequence Only\nGCC-LMI Ipsen-Mikhailov Distance")
xmax = all_results["gcc_im_dist"].max()
ymax = all_results["seq_gcc_im_dist"].max()
ax.plot([0, xmax], [0, xmax], color="gray", linestyle="--")
plt.tight_layout()
plt.savefig(
    config.REPORTS_DIR / "figures" / "struct_ablation_gcc_scatter.svg",
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    all_results["shp_kl_div"], all_results["seq_shp_kl_div"], s=SIZE, alpha=ALPHA
)
ax.set_xlabel("Structure+Sequence\nSHP KL Divergence")
ax.set_ylabel("Sequence Only\nSHP KL Divergence")
xmax = all_results["shp_kl_div"].max()
ax.plot([0, xmax], [0, xmax], color="gray", linestyle="--")
plt.tight_layout()
plt.savefig(
    config.REPORTS_DIR / "figures" / "struct_ablation_shp_scatter.svg",
)

# %%
