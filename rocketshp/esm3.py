import torch
import tempfile
import numpy as np
from io import StringIO
from esm.models.esm3 import ESM3
from esm.pretrained import ESM3_structure_decoder_v0, ESM3_structure_encoder_v0
from esm.sdk.api import ESM3InferenceClient, ESMProtein, LogitsConfig
from esm.tokenization import get_model_tokenizers
from esm.utils.constants import models as M
from esm.utils.generation import _stack_protein_tensors
from esm.utils.encoding import tokenize_structure
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils import residue_constants as RC
from esm.utils.structure.affine3d import build_affine3d_from_coordinates
from esm.utils.structure.normalize_coordinates import normalize_coordinates

def _auth_huggingface(token):
    import os

    os.environ["HF_TOKEN"] = token


def _get_model(
    model: str = M.ESM3_OPEN_SMALL, device: str = "cuda:0"
) -> ESM3InferenceClient:
    """
    Load the ESM-3 model.

    Returns:
    ESM3
        The ESM-3 model.
    """
    d: torch.device = torch.device(device)
    model: ESM3InferenceClient = ESM3.from_pretrained(model).to(d)
    return model


def _get_tokenizers(model: str = M.ESM3_OPEN_SMALL) -> tuple:
    """
    Get the ESM-3 tokenizers.

    Returns:
    tuple
        The ESM-3 tokenizers.
    """
    return get_model_tokenizers(model)


def _get_structure_vae() -> tuple:
    """
    Get the ESM-3 structure encoder.

    Returns:
    ESM3InferenceClient
        The ESM-3 structure encoder.
    """
    encoder = ESM3_structure_encoder_v0()
    decoder = ESM3_structure_decoder_v0()

    return encoder, decoder

def _tokenize_chain(esmc) -> torch.Tensor:
    struct_encoder = _get_structure_vae()[0]
    struct_tokenizer = _get_tokenizers().structure

    _, _, struct_tokens = tokenize_structure(
        torch.from_numpy(esmc.atom37_positions),
        structure_encoder = struct_encoder,
        structure_tokenizer = struct_tokenizer,
        reference_sequence = esmc.sequence,
    )
    return struct_tokens[1:-1].cpu()


def sequence_encode(
    seqs: list[str],
    model: ESM3InferenceClient,
    tokenizers: tuple,
    device: str = "cuda:0",
) -> torch.Tensor:
    """
    Embed sequences using the ESM-3 model.

    Parameters:
    seqs: list[str]
        The sequences to embed.
    model: ESM3InferenceClient
        The ESM-3 model.
    tokenizers: tuple
        ESM-3 tokenizers.
    device: str
        The device to use.

    Returns:
    torch.Tensor
        The embeddings.
    """
    d: torch.device = torch.device(device)

    seqs: list[str] = [s for s in seqs]
    esmprots: list[ESMProtein] = [ESMProtein(sequence=s) for s in seqs]
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

def frame_to_chain(F):
    with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp:
        F.save_pdb(tmp.name)
        tmp.seek(0)
        pdb_content = tmp.read()
        with StringIO(pdb_content.decode()) as sio:
            esmc = ProteinChain.from_pdb(sio)
    return esmc

def frame_to_struct_encoder_inputs(F):

    num_res = F.n_residues

    atom_positions = np.full(
        [num_res, RC.atom_type_num, 3], np.nan, dtype=np.float32
    )
    residue_index = np.full([num_res], -1, dtype=np.int64)

    for i, res in enumerate(F.top.residues):

        residue_index[i] = res.index + 1

        for j, atom in enumerate(res.atoms):
            if atom.name == "SE" and res.name == "MSE":
                atom.name = "SD"

            if atom.name in RC.atom_order:
                atom_positions[i, RC.atom_order[atom.name]] = 10 * np.round(F.xyz[0, atom.index], 4)

    return atom_positions, residue_index
        
def struct_tokenize_frame(F, encoder=None, tokenizer=None, device=None):
    """
    Tokenize a trajectory frame using the ESM-3 structure tokenizer.
    """

    if device is None:
        device = torch.device("cuda:0")
    if encoder is None:
        encoder = _get_structure_vae()[0]
    if tokenizer is None:
        tokenizer = _get_tokenizers().structure
    encoder = encoder.to(device)

    atom37_positions, residue_index = frame_to_struct_encoder_inputs(F)
    coordinates = torch.from_numpy(atom37_positions).unsqueeze(0).to(device)
    residue_index = torch.from_numpy(residue_index).unsqueeze(0).to(device)
    
    _, structure_tokens = encoder.encode(coordinates, residue_index=residue_index)
    return torch.squeeze(structure_tokens, dim=0)

def struct_tokenize_chain(esmc, encoder=None, tokenizer=None, device=None):
    """
    Tokenize a trajectory frame using the ESM-3 structure tokenizer.
    """

    if device is None:
        device = torch.device("cuda:0")
    if encoder is None:
        encoder = _get_structure_vae()[0]
    if tokenizer is None:
        tokenizer = _get_tokenizers().structure
    encoder = encoder.to(device)

    esmc = esmc.normalize_coordinates()
    coordinates = torch.from_numpy(esmc.atom37_positions).unsqueeze(0).to(device)
    residue_index = torch.from_numpy(esmc.residue_index).unsqueeze(0).to(device)
    
    _, structure_tokens = encoder.encode(coordinates, residue_index=residue_index)
    return torch.squeeze(structure_tokens, dim=0)

def structure_encode(chain: ProteinChain, encoder, stage = "encoded"):
    """
    "stage" can be one of "encoded", "pre-quantized", or "quantized"
    """
    assert stage in ["encoded", "pre-quantized", "quantized"], "Invalid stage"

    coords = normalize_coordinates(torch.tensor(chain.atom37_positions, dtype=torch.float32)).unsqueeze(0)
    coords = coords.to("cuda")
    coords = coords[..., :3, :]
    affine, affine_mask = build_affine3d_from_coordinates(coords=coords)

    attention_mask = torch.ones_like(affine_mask, dtype=torch.bool)
    attention_mask = attention_mask.bool()
    sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)

    z = encoder.encode_local_structure(
        coords=coords,
        affine=affine,
        attention_mask=attention_mask,
        affine_mask=affine_mask,
        sequence_id=sequence_id,
    )

    if stage in ["pre-quantized", "quantized"]:
        z = z.masked_fill(~affine_mask.unsqueeze(2), 0)
        z = encoder.pre_vq_proj(z)

    if stage == "quantized":
        _, z, _ = encoder.codebook(z)

    return z