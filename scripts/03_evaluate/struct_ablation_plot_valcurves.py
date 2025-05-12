# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rocketshp import config

plt.rcParams.update({
    # "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])",
    "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#FFD53E', '#1D2954'])", # simons foundation
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 16,
    "figure.autolayout": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "svg.fonttype": "none",
    })

# %% Load data
root_dir = f"{config.PROCESSED_DATA_DIR}/202505_struct_ablation"

id_map = {
    "RSHP-189": "Encoded",
    "RSHP-188": "Pre-quantized",
    "RSHP-193": "Quantized",
    "RSHP-190": "Sequence Only",
    "RSHP-191": "Ramachandran Angles",
}

data = []

for k in id_map.keys():
    df = pd.read_csv(f"{root_dir}/{k}__training_val_loss.csv", header=None)
    df.columns = ["Step", "?", "Validation Loss"]
    df["Struct. Encoding"] = id_map[k]
    data.append(df)

data = pd.concat(data)
# %%
fig,ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data, x="Step", y="Validation Loss",
            hue="Struct. Encoding",
            hue_order=["Sequence Only", "Ramachandran Angles", "Pre-quantized", "Quantized", "Encoded"],
)
plt.tight_layout()
plt.savefig(
    config.REPORTS_DIR / "figures" / "struct_ablation_validation_loss.svg",
)
# %%
