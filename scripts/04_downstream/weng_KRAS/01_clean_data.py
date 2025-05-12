# %% Imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rocketshp.config import RAW_DATA_DIR

plt.rcParams.update(
    {
        # "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])",
        "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#1D2954', '#FFD53E'])",  # simons foundation    "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "font.size": 16,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    }
)

# %% Load data

# supp4 = pd.read_excel(
#     RAW_DATA_DIR / "KRAS_DMS" / "KRAS_Fitness_Binding_DMS.xlsx",
#     sheet_name="TableS4",
#     header=0,
# )
# supp4_readme = pd.read_excel(
#     RAW_DATA_DIR / "KRAS_DMS" / "KRAS_Fitness_Binding_DMS.xlsx",
#     sheet_name="README",
#     header=0,
# )

supp5 = pd.read_excel(
    RAW_DATA_DIR / "KRAS_DMS" / "KRAS_Folding_Binding_DDG_DMS.xlsx",
    sheet_name="TableS5",
    header=0,
)

supp5_readme = pd.read_excel(
    RAW_DATA_DIR / "KRAS_DMS" / "KRAS_Folding_Binding_DDG_DMS.xlsx",
    sheet_name="README",
    header=0,
)

supp5["variance_kcal/mol"] = supp5["std_kcal/mol"] ** 2

# %% Plots
raf1_ddg = supp5[supp5["assay"] == "RAF1"]
raf1_ddg["abs_mean_kcal/mol"] = abs(raf1_ddg["mean_kcal/mol"])

fig, ax = plt.subplots(figsize=(18, 6))
sns.scatterplot(
    data=raf1_ddg,
    x="Pos_real",
    y="abs_mean_kcal/mol",
    hue="variance_kcal/mol",
    palette="magma",
    ax=ax,
)
plt.show()

# %%
folding_ddg = supp5[supp5["assay"] == "folding"]
folding_ddg["abs_mean_kcal/mol"] = abs(folding_ddg["mean_kcal/mol"])

fig, ax = plt.subplots(figsize=(18, 6))
sns.scatterplot(
    data=folding_ddg,
    x="Pos_real",
    y="abs_mean_kcal/mol",
    hue="variance_kcal/mol",
    palette="magma",
    ax=ax,
)
plt.show()

# %%
