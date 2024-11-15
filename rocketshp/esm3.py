import torch
from esm.models.esm3 import ESM3
from esm.pretrained import ESM3_structure_decoder_v0, ESM3_structure_encoder_v0
from esm.sdk.api import ESM3InferenceClient, ESMProtein, LogitsConfig
from esm.tokenization import get_model_tokenizers
from esm.utils.constants import models as M
from esm.utils.generation import _stack_protein_tensors


def _auth_huggingface(token):
    import os

    os.environ["HF_TOKEN"] = token


def _get_esm3_model(
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


def _get_esm3_tokenizers(model: str = M.ESM3_OPEN_SMALL) -> tuple:
    """
    Get the ESM-3 tokenizers.

    Returns:
    tuple
        The ESM-3 tokenizers.
    """
    return get_model_tokenizers(model)


def _get_esm3_structure_vae() -> tuple:
    """
    Get the ESM-3 structure encoder.

    Returns:
    ESM3InferenceClient
        The ESM-3 structure encoder.
    """
    encoder = ESM3_structure_encoder_v0()
    decoder = ESM3_structure_decoder_v0()

    return encoder, decoder


def esm3_embed(
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
