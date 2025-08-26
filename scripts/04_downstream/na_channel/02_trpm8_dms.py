# %% Imports
import pickle as pk
import h5py
import time

import matplotlib.pyplot as plt
import torch
from loguru import logger
from tqdm import tqdm

from rocketshp import config
from rocketshp import RocketSHP, load_sequence, load_structure
from biotite.structure.io import pdb
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# %% Parameters
log_file = "trpm8_dms_evaluation.log"
logger.add(log_file, level="INFO", format="{message}", encoding="utf-8")

# Batch size for inference (adjust based on your GPU memory)
BATCH_SIZE = 64  # Increase if you have more GPU memory, decrease if you get OOM errors

# %% Load model
logger.info("Loading RocketSHP model...")
model = RocketSHP.load_from_checkpoint("v1", strict=False).to(DEVICE)

# %% Run inference function
def run_inference(
    seq, struct_feats, model, device=DEVICE
):
    with torch.inference_mode():
        seq_feats = load_sequence(seq, device=device)
        temp_feats = torch.ones(seq_feats.shape[0]).to(device) * 300.0

        with torch.inference_mode():
            result = model({
                "seq_feats": seq_feats,
                "struct_feats": struct_feats,
                "temp_feats": temp_feats,
            })
        result = {k: v.squeeze().cpu() for k, v in result.items()}

    return result


def run_batched_inference_with_saving(
    mutant_data, struct_feats, model, h5_file, batch_size=32, device=DEVICE
):
    """
    Run inference on batches of sequences and save results directly to HDF5
    
    Args:
        mutant_data: List of tuples (pos, aa, mutant_sequence)
        struct_feats: Structure features (same for all sequences)
        model: The RocketSHP model
        h5_file: Open HDF5 file handle to write results
        batch_size: Number of sequences to process at once
        device: Device to run inference on
    
    Returns:
        Number of mutants processed
    """
    total_processed = 0
    
    with torch.inference_mode():
        for i in tqdm(range(0, len(mutant_data), batch_size), desc="Processing batches"):
            batch_data = mutant_data[i:i + batch_size]
            batch_sequences = [data[2] for data in batch_data]
            
            try:
                # Verify all sequences have the same length
                seq_lengths = [len(seq) for seq in batch_sequences]
                if len(set(seq_lengths)) > 1:
                    logger.error(f"Sequences in batch have different lengths: {seq_lengths}")
                    raise ValueError("All sequences must have the same length for batching")
                
                # Load sequence features for the batch
                batch_seq_feats = []
                for seq in batch_sequences:
                    seq_feats = load_sequence(seq, device=device)
                    batch_seq_feats.append(seq_feats)
                
                # Stack into a batch tensor and squeeze to match expected format
                batch_seq_feats = torch.stack(batch_seq_feats, dim=0)
                # Remove the extra dimension: [batch_size, 1, seq_length, features] -> [batch_size, seq_length, features]
                batch_seq_feats = batch_seq_feats.squeeze(1)
                
                # Create temperature features for the batch
                # batch_seq_feats shape: [batch_size, seq_length, features]
                # temp_feats should be: [batch_size, seq_length]
                batch_temp_feats = torch.ones(batch_seq_feats.shape[0], batch_seq_feats.shape[1]).to(device) * 300.0
                
                # Expand struct_feats to match batch size
                # struct_feats should be the same for all sequences since they're just mutations
                batch_struct_feats = struct_feats.unsqueeze(0).repeat(len(batch_sequences), 1, 1)
                
                # Run inference
                batch_result = model({
                    "seq_feats": batch_seq_feats,
                    "struct_feats": batch_struct_feats,
                    "temp": batch_temp_feats,
                })
                
                # Process and save results for each sequence in the batch
                for j, (pos, aa, mutant) in enumerate(batch_data):
                    result = {k: v[j].cpu() for k, v in batch_result.items()}
                    mutant_key = f"{pos}/{aa}"
                    
                    # Save to HDF5 immediately
                    for k, v in result.items():
                        h5_file.create_dataset(f"{mutant_key}/{k}", data=v.detach().cpu().numpy())
                    
                    total_processed += 1
                
                # Flush HDF5 file periodically to ensure data is written to disk
                if i % (batch_size * 5) == 0:
                    h5_file.flush()
                
                # Clear GPU cache periodically to avoid memory buildup
                if i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"GPU out of memory at batch {i//batch_size}. Try reducing BATCH_SIZE.")
                    logger.error(f"Current batch size: {batch_size}, sequences in batch: {len(batch_sequences)}")
                    raise
                else:
                    raise
    
    # Final flush to ensure all data is written
    h5_file.flush()
    
    return total_processed


def estimate_optimal_batch_size(model, seq_length, struct_feats, device=DEVICE, max_memory_fraction=0.8):
    """
    Estimate optimal batch size based on available GPU memory
    
    Args:
        model: The RocketSHP model
        seq_length: Length of the sequences
        struct_feats: Structure features
        device: Device to test on
        max_memory_fraction: Maximum fraction of GPU memory to use
    
    Returns:
        Estimated optimal batch size
    """
    if device == "cpu":
        return 32  # Default for CPU
    
    try:
        # Get GPU memory info
        total_memory = torch.cuda.get_device_properties(device).total_memory
        max_memory = int(total_memory * max_memory_fraction)
        
        # Test with a small batch to estimate memory usage per sequence
        test_batch_size = 4
        test_seq = "A" * seq_length  # Simple test sequence
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.inference_mode():
            # Test inference
            test_seqs = [test_seq] * test_batch_size
            seq_feats_list = []
            for seq in test_seqs:
                seq_feats = load_sequence(seq, device=device)
                seq_feats_list.append(seq_feats)
            
            batch_seq_feats = torch.stack(seq_feats_list, dim=0)
            batch_seq_feats = batch_seq_feats.squeeze(1)  # Remove extra dimension
            batch_temp_feats = torch.ones(batch_seq_feats.shape[0], batch_seq_feats.shape[1]).to(device) * 300.0
            batch_struct_feats = struct_feats.unsqueeze(0).repeat(test_batch_size, 1, 1)
            
            _ = model({
                "seq_feats": batch_seq_feats,
                "struct_feats": batch_struct_feats,
                "temp": batch_temp_feats,
            })
            
            # Get peak memory usage
            peak_memory = torch.cuda.max_memory_allocated()
            memory_per_seq = peak_memory / test_batch_size
            
            # Estimate optimal batch size
            optimal_batch_size = int(max_memory / memory_per_seq)
            optimal_batch_size = max(1, min(optimal_batch_size, 256))  # Clamp between 1 and 256
            
            logger.info(f"GPU memory: {total_memory / 1e9:.1f} GB")
            logger.info(f"Peak memory for test batch: {peak_memory / 1e6:.1f} MB")
            logger.info(f"Memory per sequence: {memory_per_seq / 1e6:.1f} MB")
            logger.info(f"Estimated optimal batch size: {optimal_batch_size}")
            
            torch.cuda.empty_cache()
            return optimal_batch_size
            
    except Exception as e:
        logger.warning(f"Could not estimate optimal batch size: {e}")
        return 32  # Default fallback


# %%
wt_fasta = config.RAW_DATA_DIR / "na_channels/trpm8_wt.fasta"
with open(wt_fasta, "r") as f:
    wt_sequence = f.read().strip().split("\n", 1)[1].replace("\n", "")
wt_pdb = config.RAW_DATA_DIR / "na_channels/trpm8_wt.pdb"
wt_struct = pdb.PDBFile.read(wt_pdb).get_structure()
struct_feats = load_structure(wt_struct, device=DEVICE).squeeze()

AMINO_ACIDS = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]


def generate_mutant(wt, pos, aa):
    """
    Generate a mutant sequence by replacing the amino acid at the given position with the given amino acid.
    """
    assert pos > 0 and pos <= len(wt), "Position out of range"
    assert aa in AMINO_ACIDS, "Invalid amino acid"
    mutant = list(wt)
    mutant[pos - 1] = aa
    return "".join(mutant)


def generate_mutants(wt, pos):
    """
    Generate all possible mutants by replacing the amino acid at the given position with all possible amino acids.
    """
    assert pos > 0 and pos <= len(wt), "Position out of range"
    mutants = []
    for aa in AMINO_ACIDS:
        if wt[pos - 1] != aa:
            mutants.append(generate_mutant(wt, pos, aa))
    return mutants


def iter_mutants(wt):
    """
    Iterate over all possible mutants sequences
    """
    # create the iterator of all mutants, get the length, and yield
    for pos in range(1, len(wt) + 1):
        for aa in AMINO_ACIDS:
            if wt[pos - 1] != aa:
                mutant = generate_mutant(wt, pos, aa)
                yield pos, aa, mutant


# %% Run inference on wildtype sequence
logger.info("Running inference on wildtype sequence")
wt_result = run_inference(
    wt_sequence,
    struct_feats,
    model,
)
for k, v in wt_result.items():
    logger.info(f"{k}: {v.shape}")

# Debug: Print shapes for wildtype
logger.info(f"Debug - wt_sequence length: {len(wt_sequence)}")
logger.info(f"Debug - struct_feats shape: {struct_feats.shape}")
wt_seq_feats = load_sequence(wt_sequence, device=DEVICE)
logger.info(f"Debug - wt_seq_feats shape: {wt_seq_feats.shape}")

# %% Run inference on all mutants
logger.info("Running inference on all mutants")

# Set batch size (adjust based on your GPU memory)
# BATCH_SIZE is defined at the top of the file

# Optionally estimate optimal batch size automatically
AUTO_ESTIMATE_BATCH_SIZE = True  # Set to False to use the manual BATCH_SIZE

if AUTO_ESTIMATE_BATCH_SIZE and DEVICE.startswith("cuda"):
    logger.info("Estimating optimal batch size...")
    estimated_batch_size = estimate_optimal_batch_size(
        model, len(wt_sequence), struct_feats, device=DEVICE
    )
    BATCH_SIZE = min(BATCH_SIZE, estimated_batch_size)  # Use the smaller of manual or estimated
    logger.info(f"Using batch size: {BATCH_SIZE}")
else:
    logger.info(f"Using manual batch size: {BATCH_SIZE}")

# Collect all mutants first
logger.info("Collecting all mutant sequences...")
mutant_data = []
for pos, aa, mutant in tqdm(
    iter_mutants(wt_sequence), total=len(wt_sequence) * (len(AMINO_ACIDS) - 1),
    desc="Collecting mutants"
):
    mutant_data.append((pos, aa, mutant))

logger.info(f"Total mutants to process: {len(mutant_data)}")

start = time.time()

# Create HDF5 file and save wildtype results
h5fi = h5py.File(config.RAW_DATA_DIR / "na_channels/trpm8_dms.h5", "w")
h5fi.create_dataset("wt/sequence", data=wt_sequence.encode("utf-8"))
for k, v in wt_result.items():
    h5fi.create_dataset(f"wt/{k}", data=v.detach().cpu().numpy())

# Run batched inference and save results directly to HDF5
BATCH_SIZE = 1
logger.info(f"Running batched inference with batch size {BATCH_SIZE}...")
logger.info("Results are being saved to HDF5 file as each batch completes...")


total_processed = run_batched_inference_with_saving(
    mutant_data, struct_feats, model, h5fi, batch_size=BATCH_SIZE, device=DEVICE
)

end = time.time()

h5fi.close()

logger.info(f"Time taken: {end - start:.2f} seconds")
logger.info(f"Time taken per mutant: {(end - start) / total_processed:.4f} seconds")
logger.info(f"Total mutants processed: {total_processed}")
logger.info(f"Speedup factor: ~{BATCH_SIZE}x (theoretical, actual speedup may vary)")
logger.info("All results have been saved to HDF5 file with minimal memory usage!")

# %%
