from pathlib import Path

import biotite.structure as bs
import numpy as np
import torch
import typer
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.generation import _stack_protein_tensors
from loguru import logger
from tqdm import tqdm

from rocketshp.config import PROCESSED_DATA_DIR
from rocketshp.structure.affine3d import build_affine3d_from_coordinates
from rocketshp.structure.normalize_coordinates import normalize_coordinates
from rocketshp.structure.protein_chain import ProteinChain

app = typer.Typer()


def ramachandran_angles(chain: ProteinChain):
    """
    Compute Ramachandran angles for a protein chain.
    """
    atom_array = chain.atom_array
    # Extract first peptide chain
    peptide = atom_array[bs.filter_amino_acids(atom_array)]
    # Calculate backbone dihedral angles
    # from one of the two identical chains in the asymmetric unit
    phi, psi, _ = bs.dihedral_backbone(peptide)
    phi = np.rad2deg(phi)
    psi = np.rad2deg(psi)
    # Replace invalid values at first and last position with 0
    phi[0] = phi[-1] = 0.0
    psi[0] = psi[-1] = 0.0
    return torch.stack([torch.tensor(phi), torch.tensor(psi)], dim=1).unsqueeze(0)


def esm3_vqvae(chain: ProteinChain, esm_struct_encoder, stage="encoded"):
    """
    "stage" can be one of "encoded", "pre-quantized", or "quantized"
    """
    assert stage in ["encoded", "pre-quantized", "quantized"], "Invalid stage"
    device = next(esm_struct_encoder.parameters()).device

    coords = normalize_coordinates(
        torch.tensor(chain.atom37_positions, dtype=torch.float32)
    ).unsqueeze(0)
    coords = coords.to(device)
    coords = coords[..., :3, :]
    affine, affine_mask = build_affine3d_from_coordinates(coords=coords)

    attention_mask = torch.ones_like(affine_mask, dtype=torch.bool)
    attention_mask = attention_mask.bool()
    sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)

    z = esm_struct_encoder.encode_local_structure(
        coords=coords,
        affine=affine,
        attention_mask=attention_mask,
        affine_mask=affine_mask,
        sequence_id=sequence_id,
    )

    if stage in ["pre-quantized", "quantized"]:
        z = z.masked_fill(~affine_mask.unsqueeze(2), 0)
        z = esm_struct_encoder.pre_vq_proj(z)

    if stage == "quantized":
        _, z, _ = esm_struct_encoder.codebook(z)

    return z


def esm3_sequence_OLD(chain: ProteinChain, esm_model, esm_tokenizer):
    """
    Encode sequence using ESM-3 model.
    """
    device = esm_model.device

    sequence = chain.sequence
    tokens = esm_tokenizer(sequence)
    tokens = tokens.to(device)

    with torch.no_grad():
        result = esm_model(sequence)
    return result["representations"]["mean"].detach()[1:-1]


def esm3_chain_sequence(
    chain: ProteinChain,
    model,
    tokenizers,
) -> torch.Tensor:
   return esm3_sequence(
        sequence=chain.sequence,
        model=model,
        tokenizers=tokenizers,
    ) 

def esm3_sequence(
    sequence: str,
    model,
    tokenizers,
) -> torch.Tensor:
    """
    Embed sequences using the ESM-3 model.

    Parameters:
    seqs: list[str]
        The sequences to embed.
    model: ESM3InferenceClient
        The ESM-3 model.
    device: str
        The device to use.

    Returns:
    torch.Tensor
        The embeddings.
    """
    d = model.device

    esmprots = [ESMProtein(sequence=sequence)]
    tokens = [model.encode(i) for i in esmprots]
    lengths = [len(i) for i in tokens]
    batch = _stack_protein_tensors(tokens, lengths, tokenizers, d)
    cfg = LogitsConfig(
        sequence=True,
        structure=False,
        secondary_structure=False,
        sasa=False,
        function=False,
        residue_annotations=False,
        return_embeddings=True,
    )
    logits = model.logits(batch, cfg)
    return logits.embeddings


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
