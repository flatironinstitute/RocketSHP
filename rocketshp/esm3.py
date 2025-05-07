import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch
from esm.models.esm3 import ESM3
from esm.pretrained import ESM3_structure_encoder_v0
from esm.tokenization import get_esm3_model_tokenizers
from esm.utils.constants import models as M
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


def _auth_huggingface(token):
    import os

    os.environ["HF_TOKEN"] = token


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


class EsmSequenceTokenizer(PreTrainedTokenizerFast):
    """
    Constructs an ESM tokenizer.
    """

    model_input_names = ["sequence_tokens", "attention_mask"]

    def __init__(
        self,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        chain_break_token="|",
        **kwargs,
    ):
        all_tokens = ESM_CONSTANTS.SEQUENCE_VOCAB
        token_to_id = {tok: ind for ind, tok in enumerate(all_tokens)}

        # a character-level tokenizer is the same as BPE with no token merges
        bpe = BPE(token_to_id, merges=[], unk_token=unk_token)
        tokenizer = Tokenizer(bpe)
        special_tokens = [
            cls_token,
            pad_token,
            mask_token,
            eos_token,
            chain_break_token,
        ]
        self.cb_token = chain_break_token
        additional_special_tokens = [chain_break_token]

        tokenizer.add_special_tokens(
            special_tokens,
        )

        # This is where we configure the automatic addition of special tokens when we call
        # tokenizer(text, add_special_tokens=True). Note that you can also configure how two
        # sequences are merged if you want.
        tokenizer.post_processor = TemplateProcessing(  # type: ignore
            single="<cls> $A <eos>",
            special_tokens=[
                ("<cls>", tokenizer.token_to_id("<cls>")),
                ("<eos>", tokenizer.token_to_id("<eos>")),
            ],
        )
        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    # These are a footgun, we never use the `bos` token anywhere so we're just overriding it here.
    @property
    def bos_token(self):
        return self.cls_token

    @property
    def bos_token_id(self):
        return self.cls_token_id

    @property
    def chain_break_token(self):
        return self.cb_token

    @property
    def chain_break_token_id(self):
        return self.convert_tokens_to_ids(self.chain_break_token)

    @property
    def all_token_ids(self):
        return list(range(self.vocab_size))

    @property
    def special_token_ids(self):
        return self.all_special_ids


def get_model(model: str = M.ESM3_OPEN_SMALL, device: str = "cuda:0") -> ESM3:
    """
    Load the ESM-3 model.

    Returns:
    ESM3
        The ESM-3 model.
    """
    d: torch.device = torch.device(device)
    model: ESM3 = ESM3.from_pretrained(model).to(d)
    return model


def get_tokenizers(model: str = M.ESM3_OPEN_SMALL) -> tuple:
    """
    Get the ESM-3 tokenizers.

    Returns:
    tuple
        The ESM-3 tokenizers.
    """
    return get_esm3_model_tokenizers(model)


def get_structure_vae() -> torch.nn.Module:
    """
    Get the ESM-3 structure encoder.

    Returns:
    ESM3InferenceClient
        The ESM-3 structure encoder.
    """
    encoder = ESM3_structure_encoder_v0()

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
