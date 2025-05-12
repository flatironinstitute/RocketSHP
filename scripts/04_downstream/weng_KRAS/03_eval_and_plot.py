# %% Imports
import pickle as pk

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from rocketshp import config

plt.rcParams.update(
    {
        # "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])",
        "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#1D2954', '#FFD53E'])",  # simons foundation    "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "font.size": 30,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    }
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# parser = argparse.ArgumentParser(description="Evaluate RocketSHP model on DMS data")
# parser.add_argument("eval_key", type=str, help="Evaluation key")
# parser.add_argument("--dist-thresh", type=float, help="Distance threshold for centrality calculation in angstrom [6.0]", default=6.0)
# args = parser.parse_args()
# EVAL_KEY = args.eval_key

# EVAL_KEY = "full_seq_model"
# EVAL_KEY = "mini_seq_model"
EVAL_KEY = "large_model_20250427"

DIST_THRESH_ANGSTROM = 6.0
DIST_THRESH_NM = DIST_THRESH_ANGSTROM / 10.0

# %% Load pickle

with open(config.REPORTS_DIR / EVAL_KEY / "mutant_results.pkl", "rb") as f:
    mutant_results = pk.load(f)
    logger.info(f"Loaded {len(mutant_results)} mutant results")
if mutant_results[0][0] == 0:
    wild_type_result = mutant_results[0]
    mutant_results = mutant_results[1:]  # remove the first element which is a dummy
# %% Process networks
SEQ_LENGTH = len(mutant_results[0][2])
logger.info(f"Sequence length: {SEQ_LENGTH}")

# for each prediction, get the list of RMSFs predicted at that position for all variants
res_i_results = {}
for pos_i in range(SEQ_LENGTH):
    res_i_results[pos_i] = []
    for mutant_result in mutant_results:
        if mutant_result[0] == pos_i:
            res_i_results[pos_i].append(mutant_result)

res_i_computed = []
for pos_i, mutants in tqdm(
    res_i_results.items(), total=len(res_i_results), desc="Processing mutants"
):
    for m in mutants:
        m_pos = m[0]
        assert pos_i == m_pos, f"Position mismatch: {pos_i} != {m_pos}"
        m_alt = m[1]
        m_dict = m[3]
        pos_rmsf = m_dict["rmsf"][pos_i].item()

        mask = m_dict["ca_dist"].numpy() < 0.6
        masked_gcc = m_dict["gcc_lmi"].numpy() * mask
        prot_graph = nx.from_numpy_array(masked_gcc)
        centrality = nx.betweenness_centrality(prot_graph, normalized=True)
        pos_centrality = centrality[pos_i]

        res_i_computed.append((int(pos_i), m_alt, pos_rmsf, pos_centrality))

# %% Process wild type
wild_type_graph = wild_type_result[3]["gcc_lmi"].numpy()
wild_type_mask = wild_type_graph < 0.6
wild_type_graph = wild_type_graph * wild_type_mask
wild_type_graph = nx.from_numpy_array(wild_type_graph)
wild_type_centrality = nx.betweenness_centrality(wild_type_graph, normalized=True)

# %% Create DataFrame
res_df = pd.DataFrame(res_i_computed, columns=["pos_i", "aa_alt", "rmsf", "centrality"])
variance_df = res_df.groupby("pos_i")[["rmsf", "centrality"]].var().reset_index()
variance_df.rename(
    columns={"rmsf": "variance_rmsf", "centrality": "variance_centrality"}, inplace=True
)
res_df = pd.merge(res_df, variance_df, on="pos_i", how="left")

# %% Recompupte variance test
res_df_test = res_df.copy()
variance_by_group = res_df_test.groupby("pos_i")["centrality"].var()
res_df_test["variance_centrality"] = res_df_test["pos_i"].map(variance_by_group)

# %% Load true DMS data

supp5 = pd.read_excel(
    config.RAW_DATA_DIR / "KRAS_DMS" / "KRAS_Folding_Binding_DDG_DMS.xlsx",
    sheet_name="TableS5",
    header=0,
).dropna()

supp5_readme = pd.read_excel(
    config.RAW_DATA_DIR / "KRAS_DMS" / "KRAS_Folding_Binding_DDG_DMS.xlsx",
    sheet_name="README",
    header=0,
)

supp5["variance_kcal/mol"] = supp5["std_kcal/mol"] ** 2

# Compute absolute values and variance
raf1_ddg = supp5[supp5["assay"] == "RAF1"]
raf1_ddg["abs_mean_kcal/mol"] = abs(raf1_ddg["mean_kcal/mol"])
raf1_variance = raf1_ddg.groupby("Pos_real")["abs_mean_kcal/mol"].var()
raf1_ddg["variance_kcal/mol"] = raf1_ddg["Pos_real"].map(raf1_variance)

folding_ddg = supp5[supp5["assay"] == "folding"]
folding_ddg["abs_mean_kcal/mol"] = abs(folding_ddg["mean_kcal/mol"])
folding_variance = folding_ddg.groupby("Pos_real")["abs_mean_kcal/mol"].var()
folding_ddg["variance_kcal/mol"] = folding_ddg["Pos_real"].map(folding_variance)

fold_raf = pd.merge(
    folding_ddg,
    raf1_ddg,
    left_on=["Pos_real", "wt_codon", "mt_codon"],
    right_on=["Pos_real", "wt_codon", "mt_codon"],
)
fold_raf["mean_kcal/mol"] = fold_raf["mean_kcal/mol_x"] + fold_raf["mean_kcal/mol_y"]
fold_raf["abs_mean_kcal/mol"] = (
    fold_raf["abs_mean_kcal/mol_x"] + fold_raf["abs_mean_kcal/mol_y"]
)
fold_raf_variance = fold_raf.groupby("Pos_real")["abs_mean_kcal/mol"].var()
fold_raf["variance_kcal/mol"] = fold_raf["Pos_real"].map(fold_raf_variance)

# %% Compute correlation

merged_df = pd.merge(
    res_df_test,
    folding_ddg,
    left_on=["pos_i", "aa_alt"],
    right_on=["Pos_real", "mt_codon"],
    how="inner",
)

# CORR_A = "variance_centrality"
CORR_A = "centrality"

# CORR_B = "variance_kcal/mol"
CORR_B = "abs_mean_kcal/mol"

# Compute correlation between betweenness centrality and folding DDG
corr, p_value = pearsonr(merged_df[CORR_A], merged_df[CORR_B])
logger.info(
    f"Pearson correlation between betweenness centrality and folding DDG: {corr:.2f} (p-value: {p_value:.2e})"
)
corr, p_value = spearmanr(merged_df[CORR_A], merged_df[CORR_B])
logger.info(
    f"Spearman correlation between betweenness centrality and folding DDG: {corr:.2f} (p-value: {p_value:.2e})"
)

fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(x=merged_df[CORR_A], y=merged_df[CORR_B], ax=ax)
plt.xlabel("Betweenness Centrality")
plt.ylabel("Folding |DDG|\n(kcal/mol)")
plt.title(
    f"Pearson: {corr:.2f} (p-value: {p_value:.2e})\nSpearman: {corr:.2f} (p-value: {p_value:.2e})"
)

# %% Binding


def add_cbar(ax, label, cmap, vmin, vmax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=-0.5)
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(label)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)  # Force scientific notation
    formatter.set_powerlimits((-4, 4))  # Adjust these values based on your needs
    cbar.formatter = formatter
    cbar.update_ticks()
    return cax


LINEWIDTH = 0.5
ALPHA = 1.0
POINT_SIZE = 80

# fig, ax = plt.subplots(4, 1, figsize=(30, 24), sharex=True)
fig, ax = plt.subplots(2, 1, figsize=(24, 12), sharex=True)

sns.scatterplot(
    data=res_df_test,
    x="pos_i",
    y="centrality",
    hue="variance_centrality",
    palette="magma",
    hue_norm=(0, 1e-3),
    edgecolor="black",
    linewidth=LINEWIDTH,
    alpha=ALPHA,
    ax=ax[0],
    legend=False,
    s=POINT_SIZE,
)
add_cbar(ax[0], "Variance(Centrality)", "magma", 0, 1e-3)
ax[0].set_xlabel("")
ax[0].set_ylabel("Betweenness\nCentrality")

sns.lineplot(
    x=wild_type_centrality.keys(),
    y=wild_type_centrality.values(),
    ax=ax[0],
    label="Wild Type Centrality",
    color="green",
    linewidth=6,
    alpha=0.8,
)
ax[0].legend(loc="upper left", fontsize=25)

sns.scatterplot(
    data=folding_ddg,
    x="Pos_real",
    y="abs_mean_kcal/mol",
    hue="variance_kcal/mol",
    palette="viridis",
    hue_norm=(0, 1),
    edgecolor="black",
    linewidth=LINEWIDTH,
    alpha=ALPHA,
    ax=ax[1],
    legend=False,
    s=POINT_SIZE,
)
add_cbar(ax[1], "Variance (Fold)", "viridis", 0, 1)
ax[1].set_xlabel("")
ax[1].set_ylabel("Folding |DDG|\n(kcal/mol)")

# sns.scatterplot(data=raf1_ddg, x="Pos_real", y="abs_mean_kcal/mol",
#                 hue="variance_kcal/mol", palette="viridis", hue_norm=(0, 1),
#                 # edgecolor='white', linewidth=LINEWIDTH, alpha=ALPHA,
#                 ax=ax[2], legend=False, s=POINT_SIZE
#                 )
# add_cbar(ax[2], "Variance (Raf)", "viridis", 0, 1)
# ax[2].set_xlabel("")
# ax[2].set_ylabel("RAF Binding |DDG|\n(kcal/mol)")

# sns.scatterplot(data=fold_raf, x="Pos_real", y="abs_mean_kcal/mol",
#                 hue="variance_kcal/mol", palette="plasma", hue_norm=(0, 1.5),
#                 # edgecolor='white', linewidth=LINEWIDTH, alpha=ALPHA,
#                 ax=ax[3], legend=False, s=POINT_SIZE
#                 )
# add_cbar(ax[3], "Variance (F+B)", "plasma", 0, 1.5)
# ax[3].set_xlabel("Amino Acid")
# ax[3].set_ylabel("Fold + Bind |DDG|\n(kcal/mol)")

plt.tight_layout()
plt.savefig(config.REPORTS_DIR / EVAL_KEY / "figures" / "KRAS_centrality_vs_ddg.svg")
plt.savefig(
    config.REPORTS_DIR / EVAL_KEY / "figures" / "KRAS_centrality_vs_ddg.png", dpi=300
)
# %%

# %%
