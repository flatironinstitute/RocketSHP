import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import huggingface_hub
import torch
from esm.models.esm3 import ESM3
from esm.models.vqvae import (
    StructureTokenEncoder,
)
from esm.pretrained import (
    ESM3_function_decoder_v0,
    ESM3_structure_decoder_v0,
    ESM3_structure_encoder_v0,
)
from esm.tokenization import (
    EsmSequenceTokenizer,
    InterProQuantizedTokenizer,
    ResidueAnnotationsTokenizer,
    SASADiscretizingTokenizer,
    SecondaryStructureTokenizer,
    StructureTokenizer,
)
from esm.utils.constants import models as M
from huggingface_hub import snapshot_download


@dataclass
class ESM_CONSTANT_CLASS:
    SEQUENCE_BOS_TOKEN = 0
    SEQUENCE_PAD_TOKEN = 1
    SEQUENCE_EOS_TOKEN = 2
    SEQUENCE_CHAINBREAK_TOKEN = 31
    SEQUENCE_MASK_TOKEN = 32

    VQVAE_CODEBOOK_SIZE = 4096
    VQVAE_SPECIAL_TOKENS = {
        "MASK": VQVAE_CODEBOOK_SIZE,
        "EOS": VQVAE_CODEBOOK_SIZE + 1,
        "BOS": VQVAE_CODEBOOK_SIZE + 2,
        "PAD": VQVAE_CODEBOOK_SIZE + 3,
        "CHAINBREAK": VQVAE_CODEBOOK_SIZE + 4,
    }
    VQVAE_DIRECTION_LOSS_BINS = 16
    VQVAE_PAE_BINS = 64
    VQVAE_MAX_PAE_BIN = 31.0
    VQVAE_PLDDT_BINS = 50

    STRUCTURE_MASK_TOKEN = VQVAE_SPECIAL_TOKENS["MASK"]
    STRUCTURE_BOS_TOKEN = VQVAE_SPECIAL_TOKENS["BOS"]
    STRUCTURE_EOS_TOKEN = VQVAE_SPECIAL_TOKENS["EOS"]
    STRUCTURE_PAD_TOKEN = VQVAE_SPECIAL_TOKENS["PAD"]
    STRUCTURE_CHAINBREAK_TOKEN = VQVAE_SPECIAL_TOKENS["CHAINBREAK"]
    STRUCTURE_UNDEFINED_TOKEN = 955

    SASA_PAD_TOKEN = 0

    SS8_PAD_TOKEN = 0

    INTERPRO_PAD_TOKEN = 0

    RESIDUE_PAD_TOKEN = 0

    CHAIN_BREAK_STR = "|"

    SEQUENCE_BOS_STR = "<cls>"
    SEQUENCE_EOS_STR = "<eos>"

    MASK_STR_SHORT = "_"
    SEQUENCE_MASK_STR = "<mask>"
    SASA_MASK_STR = "<unk>"
    SS8_MASK_STR = "<unk>"

    # fmt: off
    SEQUENCE_VOCAB = [
        "<cls>", "<pad>", "<eos>", "<unk>",
        "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
        "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
        "O", ".", "-", "|",
        "<mask>",
    ]
    # fmt: on

    SSE_8CLASS_VOCAB = "GHITEBSC"
    SSE_3CLASS_VOCAB = "HEC"
    SSE_8CLASS_TO_3CLASS_MAP = {
        "G": "H",
        "H": "H",
        "I": "H",
        "T": "C",
        "E": "E",
        "B": "E",
        "S": "C",
        "C": "C",
    }

    SASA_DISCRETIZATION_BOUNDARIES = [
        0.8,
        4.0,
        9.6,
        16.4,
        24.5,
        32.9,
        42.0,
        51.5,
        61.2,
        70.9,
        81.6,
        93.3,
        107.2,
        125.4,
        151.4,
    ]

    MAX_RESIDUE_ANNOTATIONS = 16

    TFIDF_VECTOR_SIZE = 58641

    @staticmethod
    @cache
    def data_root():
        if "INFRA_PROVIDER" in os.environ:
            return Path("")
        # Try to download from hugginface if it doesn't exist
        path = Path(snapshot_download(repo_id="EvolutionaryScale/esm3-sm-open-v1"))
        return path

    IN_REPO_DATA_FOLDER = Path(__file__).parents[2] / "data"

    INTERPRO_ENTRY = IN_REPO_DATA_FOLDER / "entry_list_safety_29026.list"
    INTERPRO_HIERARCHY = IN_REPO_DATA_FOLDER / "ParentChildTreeFile.txt"
    INTERPRO2GO = IN_REPO_DATA_FOLDER / "ParentChildTreeFile.txt"
    INTERPRO_2ID = "data/tag_dict_4_safety_filtered.json"

    LSH_TABLE_PATHS = {
        "8bit": "data/hyperplanes_8bit_58641.npz",
    }

    KEYWORDS_VOCABULARY = (
        IN_REPO_DATA_FOLDER / "keyword_vocabulary_safety_filtered_58641.txt"
    )
    KEYWORDS_IDF = IN_REPO_DATA_FOLDER / "keyword_idf_safety_filtered_58641.npy"

    RESID_CSV = "data/uniref90_and_mgnify90_residue_annotations_gt_1k_proteins.csv"
    INTERPRO2KEYWORDS = IN_REPO_DATA_FOLDER / "interpro_29026_to_keywords_58641.csv"


ESM_CONSTANTS = ESM_CONSTANT_CLASS()


def get_model(
    model: str = M.ESM3_OPEN_SMALL, device: str = "cuda:0", HF_TOKEN: str | None = None
) -> ESM3:
    """
    Load the ESM-3 model.

    Returns:
    ESM3
        The ESM-3 model.
    """

    if model != M.ESM3_OPEN_SMALL:
        raise ValueError(
            f"Model {model} is not supported. Only {M.ESM3_OPEN_SMALL} is currently supported."
        )

    d: torch.device = torch.device(device)
    model: ESM3 = ESM3_sm_open_v0(device=d, HF_TOKEN=HF_TOKEN)
    return model


@dataclass
class TokenizerCollection:
    sequence: EsmSequenceTokenizer
    structure: StructureTokenizer
    secondary_structure: SecondaryStructureTokenizer
    sasa: SASADiscretizingTokenizer
    function: InterProQuantizedTokenizer
    residue_annotations: ResidueAnnotationsTokenizer


def get_tokenizers(model: str = M.ESM3_OPEN_SMALL) -> TokenizerCollection:
    """
    Get the ESM-3 tokenizers.

    Returns:
    tuple
        The ESM-3 tokenizers.
    """
    return TokenizerCollection(
        sequence=EsmSequenceTokenizer(),
        structure=StructureTokenizer(),
        secondary_structure=SecondaryStructureTokenizer(kind="ss8"),
        sasa=SASADiscretizingTokenizer(),
        function=InterProQuantizedTokenizer(),
        residue_annotations=ResidueAnnotationsTokenizer(),
    )


def ESM3_sm_open_v0(device: torch.device | str = "cpu", HF_TOKEN: str | None = None):
    _ = huggingface_hub.snapshot_download(
        repo_id="EvolutionaryScale/esm3-sm-open-v1", token=HF_TOKEN
    )

    weights_path = huggingface_hub.hf_hub_download(
        repo_id="EvolutionaryScale/esm3-sm-open-v1",
        filename="data/weights/esm3_sm_open_v1.pth",
        token=HF_TOKEN,
    )

    with torch.device(device):
        model = ESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_fn=ESM3_structure_encoder_v0,
            structure_decoder_fn=ESM3_structure_decoder_v0,
            function_decoder_fn=ESM3_function_decoder_v0,
            tokenizers=get_tokenizers(M.ESM3_OPEN_SMALL),
        ).eval()

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def get_structure_vae(
    device: torch.device | str = "cpu", HF_TOKEN: str | None = None
) -> torch.nn.Module:
    """
    Get the ESM-3 structure encoder.

    Returns:
    ESM3InferenceClient
        The ESM-3 structure encoder.
    """

    _ = huggingface_hub.snapshot_download(
        repo_id="EvolutionaryScale/esm3-sm-open-v1", token=HF_TOKEN
    )

    weights_path = huggingface_hub.hf_hub_download(
        repo_id="EvolutionaryScale/esm3-sm-open-v1",
        filename="data/weights/esm3_structure_encoder_v0.pth",
        token=HF_TOKEN,
    )

    with torch.device(device):
        encoder = StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        ).eval()

    state_dict = torch.load(weights_path, map_location=device)
    encoder.load_state_dict(state_dict)
    return encoder


# def _tokenize_chain(esmc) -> torch.Tensor:
#     struct_encoder = _get_structure_vae()[0]
#     struct_tokenizer = _get_tokenizers().structure

#     _, _, struct_tokens = tokenize_structure(
#         torch.from_numpy(esmc.atom37_positions),
#         structure_encoder = struct_encoder,
#         structure_tokenizer = struct_tokenizer,
#         reference_sequence = esmc.sequence,
#     )
#     return struct_tokens[1:-1].cpu()
