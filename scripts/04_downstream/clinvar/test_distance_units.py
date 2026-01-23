#!/usr/bin/env python3
"""
Quick test to verify RocketSHP ca_dist prediction units.

This script loads a known protein structure, predicts ca_dist,
and compares to actual distances to determine units.

Run before executing the full pipeline to verify distance_cutoff settings.
"""

import numpy as np
import torch
from biotite.structure import distance
from biotite.structure.io import pdb

from rocketshp import RocketSHP, load_sequence, load_structure


def test_distance_units(pdb_path: str = "kras_afdb.pdb", device: str = "cuda:0"):
    """
    Test whether RocketSHP predicts distances in Angstroms or nanometers.

    Args:
        pdb_path: Path to a test PDB file
        device: Device to run model on
    """
    print("="*60)
    print("RocketSHP Distance Unit Verification")
    print("="*60)

    # Load structure
    print(f"\nLoading structure from {pdb_path}...")
    structure_file = pdb.PDBFile.read(pdb_path)
    structure = structure_file.get_structure()

    # Get CA atoms only
    ca_mask = structure.atom_name == "CA"
    ca_atoms = structure[ca_mask]
    n_residues = len(ca_atoms)

    print(f"Found {n_residues} CA atoms")

    # Calculate actual distances
    print("\nCalculating actual CA-CA distances...")
    actual_distances = np.zeros((n_residues, n_residues))
    for i in range(n_residues):
        for j in range(i+1, n_residues):
            dist = distance(ca_atoms[i], ca_atoms[j])
            actual_distances[i, j] = dist
            actual_distances[j, i] = dist

    # Load model and predict
    print("\nLoading RocketSHP model...")
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model = RocketSHP.load_from_checkpoint("latest").to(device_obj)

    print("Generating predictions...")
    from biotite.structure import to_sequence
    sequence = str(to_sequence(structure)[0][0])

    seq_features = load_sequence(sequence, device=device_obj)
    struct_features = load_structure(structure, device=device_obj)

    with torch.inference_mode():
        predictions = model({
            "seq_feats": seq_features,
            "struct_feats": struct_features,
        })

    predicted_distances = predictions["ca_dist"].squeeze().cpu().numpy()

    print(f"Prediction shape: {predicted_distances.shape}")

    # Compare a sample of distances
    print("\n" + "="*60)
    print("Sample Distance Comparisons")
    print("="*60)
    print(f"{'Residues':<15} {'Actual (Å)':<12} {'Actual (nm)':<12} {'Predicted':<12} {'Match'}")
    print("-"*60)

    # Sample 10 random pairs
    pairs = []
    for _ in range(10):
        i = np.random.randint(0, min(n_residues-10, n_residues))
        j = i + np.random.randint(1, min(10, n_residues-i))
        pairs.append((i, j))

    angstrom_errors = []
    nm_errors = []

    for i, j in pairs:
        actual_angstrom = actual_distances[i, j]
        actual_nm = actual_angstrom / 10.0
        predicted = predicted_distances[i, j]

        error_angstrom = abs(predicted - actual_angstrom)
        error_nm = abs(predicted - actual_nm)

        angstrom_errors.append(error_angstrom)
        nm_errors.append(error_nm)

        # Determine which unit matches better
        if error_nm < error_angstrom:
            match = "nm ✓"
        else:
            match = "Å ✓"

        print(f"{i:>3}-{j:<3}  {actual_angstrom:>10.2f}  {actual_nm:>10.2f}  {predicted:>10.2f}  {match}")

    # Summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)

    mean_error_angstrom = np.mean(angstrom_errors)
    mean_error_nm = np.mean(nm_errors)

    print(f"Mean Absolute Error (assuming Å): {mean_error_angstrom:.3f}")
    print(f"Mean Absolute Error (assuming nm): {mean_error_nm:.3f}")

    if mean_error_nm < mean_error_angstrom:
        print("\n✓ RESULT: Predictions appear to be in NANOMETERS")
        print("  → distance_cutoff division by 10 is CORRECT")
        print("  → 8.0 Å cutoff → 0.8 nm threshold is appropriate")
    else:
        print("\n✗ RESULT: Predictions appear to be in ANGSTROMS")
        print("  → distance_cutoff division by 10 is INCORRECT")
        print("  → Should NOT divide by 10 in build_allosteric_network()")
        print("  → Need to fix rocketshp/network.py line 72")

    print("="*60)

    # Additional check: look at typical protein contact distances
    print("\nTypical Protein Contact Distances:")
    print("  - Adjacent residues (i, i+1): ~3.8 Å (0.38 nm)")
    print("  - Hydrogen bond distance: ~3-4 Å (0.3-0.4 nm)")
    print("  - Contact cutoff typically: 8-12 Å (0.8-1.2 nm)")

    # Check adjacent residues
    if n_residues > 5:
        adjacent_actual = actual_distances[0, 1]
        adjacent_pred = predicted_distances[0, 1]
        print(f"\nAdjacent residues (0-1):")
        print(f"  Actual: {adjacent_actual:.2f} Å ({adjacent_actual/10:.2f} nm)")
        print(f"  Predicted: {adjacent_pred:.2f}")

        if 0.3 < adjacent_pred < 0.5:
            print("  → Predicted value in nm range ✓")
        elif 3 < adjacent_pred < 5:
            print("  → Predicted value in Å range ✓")

    return predictions, actual_distances


if __name__ == "__main__":
    import sys

    pdb_path = sys.argv[1] if len(sys.argv) > 1 else "kras_afdb.pdb"
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"

    try:
        test_distance_units(pdb_path, device)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have a valid PDB file and the model is accessible.")
