#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rocketshp.config import PROJ_ROOT

#%% Load data
root_dir = "/mnt/home/ssledzieski/Projects/rocketshp/data/processed/neptune_logs"

id_map = {
    "RSHP-144": "Encoded",
    "RSHP-145": "Pre-quantized",
    "RSHP-147": "Quantized",
    "RSHP-148": "Sequence Only",
    "RSHP-149": "Ramachandran Angles",
}

data = []

for k in id_map.keys():
    df = pd.read_csv(f"{root_dir}/{k}__training_val_loss.csv",header=None)
    df.columns = ["Step","?","Validation Loss"]
    df["Struct. Encoding"] = id_map[k]
    data.append(df)

data = pd.concat(data)
# %%
sns.lineplot(data,x="Step",y="Validation Loss",hue="Struct. Encoding")
sns.despine()
plt.savefig(PROJ_ROOT / "img" / "struct_ablation_validation_loss.png",bbox_inches="tight",dpi=300)
# %%
