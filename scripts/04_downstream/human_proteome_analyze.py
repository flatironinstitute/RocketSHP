# %% Imports
import os
import torch
import pickle as pk
import pandas as pd
import numpy as np
from loguru import logger
from rocketshp import config
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from biotite.database import uniprot

plt.rcParams.update(
    {
        # "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])",
        "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#FFD53E', '#1D2954'])",  # simons foundation
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 16,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    }
)

def parse_uniprot_xml_biotite(up_results):
    """
    Extract protein annotations from biotite UniProt XML results.
    
    Args:
        up_results: Dictionary from biotite.database.uniprot.fetch with format="xml"
        
    Returns:
        DataFrame with annotations
    """
    results = []
    
    # Define UniProt XML namespace
    ns = {'up': 'http://uniprot.org/uniprot'}
    
    for uniprot_id, xml_text in tqdm(up_results.items(), desc="Parsing UniProt XML"):
        # Parse XML string
        try:
            root = ET.fromstring(xml_text)
            entry = root.find('.//up:entry', ns)
            
            if entry is None:
                logger.warning(f"Warning: No entry found for {uniprot_id}")
                continue
                
            protein_info = {
                "uniprot_id": uniprot_id,
                "protein_name": "",
                "gene_names": [],
                "protein_class": [],
                "subcellular_location": [],
                "domains": [],
                "regions": [],
                "functions": []
            }
            
            # Extract protein name
            protein_elem = entry.find('./up:protein', ns)
            if protein_elem is not None:
                rec_name = protein_elem.find('./up:recommendedName', ns)
                if rec_name is not None:
                    full_name = rec_name.find('./up:fullName', ns)
                    if full_name is not None and full_name.text:
                        protein_info["protein_name"] = full_name.text
            
            # Extract gene names
            gene_elems = entry.findall('./up:gene/up:name', ns)
            for gene_elem in gene_elems:
                if gene_elem.text:
                    protein_info["gene_names"].append(gene_elem.text)
            
            # Extract keywords (protein class)
            keyword_elems = entry.findall('./up:keyword', ns)
            for kw_elem in keyword_elems:
                if kw_elem.text:
                    protein_info["protein_class"].append(kw_elem.text)
            
            # Extract subcellular location
            comment_elems = entry.findall('./up:comment', ns)
            for comment in comment_elems:
                comment_type = comment.get('type')
                
                if comment_type == 'subcellular location':
                    loc_elems = comment.findall('.//up:location', ns)
                    for loc in loc_elems:
                        if loc.text and loc.text not in protein_info["subcellular_location"]:
                            protein_info["subcellular_location"].append(loc.text)
                
                elif comment_type == 'function':
                    text_elems = comment.findall('./up:text', ns)
                    for text in text_elems:
                        if text.text:
                            protein_info["functions"].append(text.text)
            
            # Extract regions of interest
            feature_elems = entry.findall('./up:feature', ns)
            for feature in feature_elems:
                feature_type = feature.get('type')
                
                if feature_type == 'region of interest':
                    location = feature.find('./up:location', ns)
                    if location is not None:
                        begin = location.find('./up:begin', ns)
                        end = location.find('./up:end', ns)
                        
                        try:
                            if begin is not None and end is not None:
                                start_pos = int(begin.get('position'))
                                end_pos = int(end.get('position'))
                                description = feature.get('description', '')
                                
                                feature_info = {
                                    "name": description,
                                    "start": start_pos,
                                    "end": end_pos
                                }

                                protein_info["regions"].append(feature_info)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid position for {uniprot_id}: {begin}, {end}")

            # Extract pfam domains
            pfam_elems = entry.findall('./up:dbReference[@type="Pfam"]', ns)
            for pfam in pfam_elems:
                pfam_id = pfam.get('id')
                if pfam_id:
                    protein_info["domains"].append(pfam_id)
                    protein_info["domains"] = list(set(protein_info["domains"]))  # Remove duplicates
            
            results.append(protein_info)
            
        except ET.ParseError as e:
            logger.error(f"Error parsing XML for {uniprot_id}: {e}")
    
    # Convert lists to comma-separated strings for better DataFrame handling
    result_df = pd.DataFrame(results)
    
    # Convert gene names list to string
    result_df['gene_names'] = result_df['gene_names'].apply(lambda x: ', '.join(x) if x else '')
    
    return result_df

# %%
result_dir = config.PROCESSED_DATA_DIR / "rocketshp_afdb_human"
img_dir = config.REPORTS_DIR / "proteome_scale"
# subdirectories = list(result_dir.iterdir())
subdirectories = [i for i in result_dir.iterdir() if i.is_dir()]
os.makedirs(img_dir, exist_ok=True)

# %%

# rocketshp_results = {}
# for subdir in subdirectories:
#     logger.info(f"Loading {subdir.name}")
#     proteins = list(subdir.glob("*.rshp.pt"))
#     for protein_file in tqdm(proteins):
#         pid = protein_file.stem.split("-")[1]
#         rocketshp_results[pid] = torch.load(protein_file, map_location="cpu")

# # save to single pickle file
# with open(result_dir / "rocketshp_results.pkl", "wb") as f:
#     pk.dump(rocketshp_results, f)

# load from pickle
# with open(result_dir / "rocketshp_results.pkl", "rb") as f:
#     rocketshp_results = pk.load(f)

#%% write to csv

# with open(result_dir / "rocketshp_results.csv", "w") as f:
#     for k,v in tqdm(rocketshp_results.items()):
#         f.write(f"{k}")
#         for flex in v["rmsf"].numpy():
#             f.write(f",{flex}")
#         f.write("\n")

rocketshp_rmsf = {}
with open(result_dir / "rocketshp_results.csv", "r") as f:
    for line in tqdm(f):
        line = line.strip().split(",")
        pid = line[0]
        rmsf = np.array([float(i) for i in line[1:]])
        if len(rmsf) == 1400:
            continue
        rocketshp_rmsf[pid] = rmsf
# %% Get annotations for a sub-sample

subsample_size = len(rocketshp_rmsf)
random_state = 42
np.random.seed(random_state)
subsample_prots = np.random.choice(
    list(rocketshp_rmsf.keys()), size=subsample_size, replace=False
)
subsample_prots = list(subsample_prots)

# up_results_paths = uniprot.fetch(subsample_prots, format="xml", target_path = result_dir.parent / "uniprot_xml", verbose=True)

#%% Parse XML results
# up_results = {pid: open(path, "r").read() for pid, path in zip(subsample_prots, up_results_paths)}
# annotations_df = parse_uniprot_xml_biotite(up_results)
# annotations_df.to_csv(result_dir / "uniprot_annotations.csv", index=False)

#%%
# with open(result_dir / "uniprot_annotations.pk", "wb") as f:
#     pk.dump(annotations_df, f)

with open(result_dir / "uniprot_annotations.pk", "rb") as f:
    annotations_df = pk.load(f)    

# %% Types of regions
types_of_regions = set()
for r in annotations_df["regions"]:
    for region in r:
        types_of_regions.add(region["name"])


def boxplot_regions(region_type, search_query):
    type_results = {}
    for k in tqdm(subsample_prots):
        pred_rmsf = rocketshp_rmsf[k]
        if len(pred_rmsf) == 1400:
            continue

        # get regions from annotations_df
        regions = annotations_df.loc[annotations_df["uniprot_id"] == k, "regions"].values
        try:
            regions = regions[0]
        except:
            continue
        type_residues = []
        for r in regions:
            if search_query.lower() in r["name"].lower():
                start = r["start"]
                end = r["end"]
                if end > len(pred_rmsf):
                    continue
                type_residues.extend(np.arange(start - 1, end))
        ordered_residues = np.setdiff1d(np.arange(0, len(pred_rmsf)), type_residues)
        pred_rmsf_interactions = np.mean(pred_rmsf[type_residues])
        pred_rmsf_order = np.mean(pred_rmsf[ordered_residues])

        # print(f"interactionsed RMSF: {pred_rmsf_interactions}")
        # print(f"Ordered RMSF: {pred_rmsf_order}")
        type_results[k] = {
            "type_rmsf": pred_rmsf_interactions,
            "other_rmsf": pred_rmsf_order,
        }

    type_df = pd.DataFrame.from_dict(type_results, orient="index")
    type_df = type_df.reset_index()
    type_df = type_df.rename(columns={"index": "uniprot_id"})

    type_melt = type_df.melt(id_vars="uniprot_id", value_vars=["type_rmsf","other_rmsf"])

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.boxplot(data=type_melt, x="variable", y="value", ax=ax)
    ax.set_title(f"Predicted RMSF of {region_type} regions")
    ax.set_xlabel("Region")
    ax.set_xticks([0, 1], [region_type, "Other"])
    ax.set_ylabel("RMSF")

    annotator = Annotator(
        ax, pairs=[("type_rmsf", "other_rmsf")], data=type_melt, x="variable", y="value"
    )
    annotator.configure(test="t-test_ind", loc="inside", verbose=2)
    _, test_results = annotator.apply_and_annotate()

    sns.despine()
    plt.tight_layout()
    plt.savefig(img_dir / f"{region_type}_rmsf_boxplot.svg")

#%% Disordere regions
boxplot_regions("Disordered", "Disordered")

#%% Interaction regions
boxplot_regions("Interaction", "Interaction with")

# %% Create subcellular location pie chart
subcell_by_prot = {}
subcellular_locations = []

allowed_regions = [
    "Nucleus",
    "Cytoplasm",
    "Cell membrane",
    "Endosome",
    "Secreted",
    "Golgi apparatus",
    "Flagellum",
    "Cytosekeleton",
    "Extracellular matrix",
    "Endroplasmic reticulum",
    "Centrosome",
    "Cytosol",
    "Mitochondrion",
]

for k in tqdm(subsample_prots):
    # get regions from annotations_df
    subcellular_location = annotations_df.loc[annotations_df["uniprot_id"] == k, "subcellular_location"].values
    try:
        subcellular_location = subcellular_location[0]
    except:
        continue

    for loc in subcellular_location:
        if loc in allowed_regions: 
            subcell_by_prot[k] = loc 
            continue

subcellular_location_counts_raw = pd.Series(subcell_by_prot.values()).value_counts()
subcellular_location_counts = subcellular_location_counts_raw.reset_index()
subcellular_location_counts.columns = ["subcellular_location", "count"]
subcellular_location_counts = subcellular_location_counts.sort_values(by="count", ascending=False)

# Group anything less 2% into the "Other" category
subcellular_location_counts["subcellular_location"] = np.where(
    subcellular_location_counts["count"] / len(subsample_prots) < 0.02,
    "Other",
    subcellular_location_counts["subcellular_location"],
)
subcellular_location_counts = subcellular_location_counts.groupby("subcellular_location").sum().reset_index()
subcellular_location_counts = subcellular_location_counts.sort_values(by="count", ascending=False)
subcellular_location_counts = subcellular_location_counts.reset_index(drop=True)
# %% Plot pie chart
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(
    subcellular_location_counts["count"],
    labels=subcellular_location_counts["subcellular_location"],
    autopct="%1.1f%%",
    startangle=90,
    colors=sns.color_palette("pastel"),
)
plt.savefig(img_dir / "subcellular_location_pie_chart.svg")

# %%

location_rmsfs = {}
for loc in subcellular_location_counts["subcellular_location"]:
    if loc == "Other":
        continue
    sub_df = annotations_df[annotations_df["subcellular_location"].apply(lambda x: loc in x)]
    sub_median_rmsf = []
    for p in sub_df["uniprot_id"]:
        sub_median_rmsf.append(np.median(rocketshp_rmsf[p]))
    location_rmsfs[loc] = sub_median_rmsf

location_rmsfs_df = pd.DataFrame.from_dict(location_rmsfs, orient="index").T
location_rmsfs_df = location_rmsfs_df.melt().dropna()

fig, ax = plt.subplots(figsize=(10, 5))
# sns.stripplot(location_rmsfs_df, x="variable", y="value", s=2, ax = ax)
sns.boxplot(location_rmsfs_df, x="variable", y="value", ax=ax)
plt.title("Median RMSF by subcellular location")
ax.set_xlabel("Subcellular location")
ax.set_ylabel("RMSF")
# plt.xticks(rotation=45, ha="right")

# annotator = Annotator(
#     ax,
#     pairs=[
#         ("Nucleus", "Cytoplasm"), ("Nucleus", "Cell membrane"), ("Nucleus", "Secreted"), ("Nucleus", "Mitochondrion"),
#         ("Cell membrane", "Cytoplasm"), ("Cell membrane", "Secreted"), ("Cell membrane", "Mitochondrion"),
#         ("Cytoplasm","Secreted"), ("Cytoplasm", "Mitochondrion"), ("Secreted", "Mitochondrion"),
#         ],
#     data=location_rmsfs_df,
#     x="variable",
#     y="value",
# )
# annotator.configure(test="t-test_ind", loc="inside", verbose=2)
# _, test_results = annotator.apply_and_annotate()


plt.tight_layout()
sns.despine()
plt.savefig(img_dir / "subcellular_location_rmsf_boxplot.svg")
plt.show()

# %%
