#%% Imports
import torch
from torch.nn.functional import softmax
from rocketshp import config
from rocketshp import RocketSHP, load_sequence
from rocketshp.network import build_allosteric_network, calculate_centrality, display_network,pairwise_correlation_to_network
from rocketshp.plot import plot_predictions

from biotite.sequence.io import fasta
from loguru import logger
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

#%% Load FASTA file
fasta_path = config.RAW_DATA_DIR / "na_channels/all-constructs.fasta"
fasta_file = fasta.FastaFile.read(fasta_path)
fasta_records = fasta.get_sequences(fasta_file)
records_tuple = tuple(fasta_records.items())

#%% Set up model

# Set compute device
device = torch.device("cuda:0")

# Load the model (will download if not present)
model = RocketSHP.load_from_checkpoint("v1_seq", strict=False).to(device)

#%%

predictions = {}
for key, sequence in records_tuple:
    logger.info(f"Processing sequence: {key}")
    sequence = str(sequence)

    # Load sequence features
    seq_features = load_sequence(sequence, device=device)

    # Predict dynamics from sequence only
    with torch.no_grad():
        dynamics_pred = model({"seq_feats": seq_features, "struct_feats": None})

    # Access prediction results
    rmsf = dynamics_pred["rmsf"].squeeze().cpu().numpy()
    gcc_lmi = dynamics_pred["gcc_lmi"].squeeze().cpu().numpy()
    ca_dist = dynamics_pred["ca_dist"].squeeze().cpu().numpy()
    shp = softmax(dynamics_pred["shp"].squeeze(), dim=1).cpu().numpy()

    # Store predictions
    predictions[key] = {
        "rmsf": rmsf,
        "gcc_lmi": gcc_lmi,
        "ca_dist": ca_dist,
        "shp": shp,
    }

# %% Align sequences
from biotite.sequence import align

matrix = align.SubstitutionMatrix.std_protein_matrix()
alignment, order, tree, distances = align.align_multiple([i[1] for i in records_tuple], matrix)
trace = alignment.trace.T

trace_map = {}
for i, (name, seq) in enumerate(records_tuple):
    trace_map[name] = trace[i]

def align_with_trace(array, trace, missing_id = -1):
    """
    Aligns an array with a 1D trace, inserting np.nan for gaps.
    """
    dim1 = array.shape[1] if array.ndim > 1 else 1
    aligned_array = np.full((len(trace), dim1), np.nan)
    for trace_index, array_index in enumerate(trace):
        if array_index != missing_id:
            aligned_array[trace_index] = array[array_index]

    return aligned_array.squeeze()

# %% Using sequence alignment, align the predictions with np.nan for gaps

for i in range(len(records_tuple)):
    a = len(records_tuple[i][1])
    b= len(trace[i][trace[i] != -1])
    assert a == b
    print(records_tuple[i][0], a, b)


# %% Compare RMSF

# generate colors distinct for the number of proteins
num_proteins = len(predictions)
colors = plt.cm.viridis(torch.linspace(0, 1, num_proteins))

fig, ax = plt.subplots(figsize=(12, 8))
for i, k in enumerate(predictions):
    y = align_with_trace(predictions[k]["rmsf"], trace_map[k])
    x = np.arange(len(y))
    plt.plot(x, y, label=k[:18], color=colors[i])
plt.xlabel("Residue Index")
plt.ylabel("RMSF")
plt.title("RMSF Predictions for NA Channels")
# move legend outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

#  %% Compute Betweenness
DIST_THRESH_ANGSTROM = 8.0
DIST_THRESH_NM = DIST_THRESH_ANGSTROM / 10.0

centrality_results = {}

for k in tqdm(predictions):
    allo_net = build_allosteric_network(
        predictions[k]["gcc_lmi"],
        predictions[k]["ca_dist"],
        distance_cutoff=DIST_THRESH_NM,
    )
    centrality_results_k = calculate_centrality(allo_net)
    centrality_results[k] = {
        "betweenness": centrality_results_k['betweenness'],
        "closeness": centrality_results_k['closeness'],
        "degree": centrality_results_k['degree'],
        "network": allo_net,
    }

# %% Plot Betweenness Centrality for each protein on its own subplot
fig, axs = plt.subplots(len(predictions), 1, figsize=(10, 2 * len(predictions)))
fig.tight_layout(pad=3.0)
for i, (k, result) in enumerate(centrality_results.items()):
    ax = axs[i]
    betweenness = align_with_trace(result["betweenness"], trace_map[k])
    ax.bar(range(len(betweenness)), list(betweenness), color='blue')
    ax.set_title(f"Betweenness Centrality for {k}")
    ax.set_xlabel("Residue Index")
    ax.set_ylabel("Betweenness Centrality")
    # ax.set_xticks(range(len(betweenness)))
    # ax.set_xticklabels(list(betweenness.keys()), rotation=90)
plt.show()

# %% Plot SHP for each protein on its own subplot
fig, axs = plt.subplots(len(predictions), 1, figsize=(10, 2 * len(predictions)))
fig.tight_layout(pad=3.0)
for i, (k, result) in enumerate(predictions.items()):
    ax = axs[i]
    shp = align_with_trace(result["shp"], trace_map[k])
    ax.imshow(shp.T, cmap='binary', interpolation='none', aspect='auto',
              vmin=0, vmax=1)
    ax.set_title(f"SHP for {k}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Residue Index")
    ax.set_yticks(range(20))
    ax.set_yticklabels(range(1, 21))
plt.show()

#%%

# fig,ax = plt.subplots(figsize=(10, 8))
# plt.imshow(predictions[k]["gcc_lmi"],
#            cmap='viridis',interpolation='none',aspect='auto',
#            vmin=0, vmax=1)
# plt.colorbar(label='GCC LMI')
# plt.show()
# pairwise_correlation_to_network(predictions[k]["gcc_lmi"], thresh=0.75, title=k)

# fig, ax = plt.subplots(figsize=(10, 6))
# plt.imshow(predictions[k]["shp"].T,
#            cmap='binary',interpolation='none',aspect='auto',
#            vmin=0, vmax=1)
# plt.ylim(20, -1)
# plt.yticks(ticks=range(20), labels=range(1, 21))
# plt.show()

# %% Cluster SHPs

for k, v in fasta_records.items():
    print(k, len(v))
# %%
