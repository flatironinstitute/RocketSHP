# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

files_root = "/mnt/home/ssledzieski/Projects/rocketshp/data/evaluation/evaluations"
tasks = ["rmsf", "ca_dist", "dyncorr"]
tweights = {}

for t in tasks:
    tweights[t] = pd.read_csv(
        f"{files_root}/training_weights_{t}.csv", header=None, sep=","
    )

fig, ax = plt.subplots(figsize=(10, 6))
for t, tw in tweights.items():
    plt.plot(tw[0], tw[2], label=t, alpha=0.75)
plt.xlabel("Step")
plt.ylabel("Task Weight")
plt.legend()
sns.despine()
plt.show()
# %%
