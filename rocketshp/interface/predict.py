from pathlib import Path

import typer
import numpy as np
from loguru import logger
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import torch
from loguru import logger

from rocketshp.esm3 import (
    get_model,
    get_structure_vae,
    get_tokenizers,
)
from rocketshp.features import (
    esm3_sequence,
    esm3_vqvae,
    ramachandran_angles
)
from rocketshp.structure.protein_chain import ProteinChain

from rocketshp.modeling.architectures import DynCorrModelWithTemperature

app = typer.Typer(pretty_exceptions_enable=False)

# DEFAULT_SEQUENCE = "MEDGHSKTVEQSLNFFGTDPERGLTLDQIKANQKKYGPNELPTEEGKSIWQLVLEQFDDLLVKILLLAAIISFVLALFEEHEETFTAFVEPLVILLILIANAVVGVWQERNAESAIEALKEYEPEMGKVVRQDKSGIQKVRAKEIVPGDLVEVSVGDKIPADIRITHIYSTTLRIDQSILTGESVSVIKHTDAIPDPRAVNQDKKNILFSGTNVAAGKARGVVIGTGLSTAIGKIRTEMSETEEIKTPLQQKLDEFGEQLSKVISVICVAVWAINIGHFNDPAHGGSWIKGAIYYFKIAVALAVAAIPEGLPAVITTCLALGTRRMAKKNAIVRSLPSVETLGCTSVICSDKTGTLTTNQMSVSRMFIFDKVEGNDSSFLEFEMTGSTYEPIGEVFLNGQRIKAADYDTLQELSTICIMCNDSAIDYNEFKQAFEKVGEATETALIVLAEKLNSFSVNKSGLDRRSAAIACRGEIETKWKKEFTLEFSRDRKSMSSYCTPLKASRLGTGPKLFVKGAPEGVLERCTHARVGTTKVPLTSALKAKILALTGQYGTGRDTLRCLALAVADSPMKPDEMDLGDSTKFYQYEVNLTFVGVVGMLDPPRKEVFDSIVRCRAAGIRVIVITGDNKATAEAICRRIGVFAEDEDTTGKSYSGREFDDLSPTEQKAAVARSRLFSRVEPQHKSKIVEFLQSMNEISAMTGDGVNDAPALKKAEIGIAMGSGTAVAKSAAEMVLADDNFSSIVSAVEEGRAIYNNMKQFIRYLISSNIGEVVSIFLTAALGLPEALIPVQLLWVNLVTDGLPATALGFNPPDLDIMEKPPRKADEGLISGWLFFRYMAIGFYVGAATVGAAAWWFVFSDEGPKLSYWQLTHHLSCLGGGDEFKGVDCKIFSDPHAMTMALSVLVTIEMLNAMNSLSENQSLITMPPWCNLWLIGSMALSFTLHFVILYVDVLSTVFQVTPLSAEEWITVMKFSIPVVLLDETLKFVARKIADGESPIYKMHGIVLMWAVFFGLLYAMML"
# DEFAULT_CONFIG = "/mnt/home/ssledzieski/Projects/rocketshp/configs/default_config.yaml"
# DEFAULT_MODEL = "/mnt/home/ssledzieski/Projects/rocketshp/models/grad_norm_alpha0.12_lr1e-5/model-epoch=19-train_loss=0.55.pt.ckpt"

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    pdb_path: Path,
    config_path: Path,
    model_path: Path,
    out_file: Path,
    device: str = "cuda:0",
    save_results: bool = True,
    # -----------------------------------------
):
    device = torch.device(device)
    chain = ProteinChain.from_pdb(pdb_path)
    logger.info(len(chain.sequence))
    config = OmegaConf.load(config_path)

    logger.info("Loading RSHP Model")
    rshp_model = DynCorrModelWithTemperature.load_from_checkpoint(model_path, strict=True)
    rshp_model.eval().to(device)

    logger.info("Loading structure features")
    struct_stage = config["struct_stage"]
    if not config["struct_features"]:
        struct_embeddings = torch.zeros(1, 1, 1).to(device)
    elif struct_stage in ["encoded","pre-quantized", "quantized"]:
        struct_encoder = get_structure_vae()
        struct_encoder.eval().to(device)

        with torch.inference_mode():
            struct_embeddings = esm3_vqvae(chain, struct_encoder, stage=config["struct_stage"]).squeeze().detach()
    elif struct_stage == "ramachandran":
        struct_embeddings = ramachandran_angles(chain).squeeze()

    logger.info("Loading sequence features")
    esm_model = get_model("esm3-open")
    esm_model.eval().to(device)
    tokenizers = get_tokenizers("esm3-open")

    with torch.inference_mode():
        seq_embeddings = esm3_sequence(chain, esm_model, tokenizers).squeeze()[1:-1].detach()

    temperature = (torch.ones(seq_embeddings.shape[0]).to(device) * 290)

    logger.info(seq_embeddings.size())
    logger.info(struct_embeddings.size())
    logger.info(temperature.size())

    logger.info("Performing inference")
    with torch.inference_mode():
        y_hat = rshp_model({
            "seq_feats": seq_embeddings.to(device).unsqueeze(0),
            "struct_feats": struct_embeddings.to(device).unsqueeze(0),
            "temp": temperature.to(device).unsqueeze(0),
        })

    if save_results:    
        logger.info("Saving results")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(y_hat, out_file.with_suffix(".pt"))

        plt.plot(np.arange(len(chain.sequence)), y_hat["rmsf"].squeeze().cpu().detach().numpy(), label="RMSF")
        plt.title(pdb_path.stem)
        plt.ylabel("RMSF")
        plt.savefig(out_file.with_suffix(".rmsf.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plt.imshow(y_hat["ca_dist"].squeeze().cpu().detach().numpy(), cmap="viridis")
        plt.title(pdb_path.stem)
        plt.xlabel("Contact Map")
        plt.savefig(out_file.with_suffix(".contacts.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plt.imshow(y_hat["dyn_corr"].squeeze().cpu().detach().numpy(), cmap="viridis")
        plt.title(pdb_path.stem)
        plt.xlabel("Dynamic Correlation")
        plt.savefig(out_file.with_suffix(".dyncorr.png"), dpi=300, bbox_inches="tight")

def __app__():
    app()

if __name__ == "__main__":
    app()
