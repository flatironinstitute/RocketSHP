import torch
import typer
from pathlib import Path
from torch.nn.functional import softmax

from rocketshp.modeling.architectures import RocketSHPModel as RocketSHP
from rocketshp.plot import plot_predictions
from rocketshp.features import esm3_sequence, esm3_vqvae
from rocketshp.esm3 import get_model, get_tokenizers, get_structure_vae
from rocketshp.structure.protein_chain import ProteinChain

import biotite.structure as bs
from biotite.structure.io import pdb
from biotite.structure import to_sequence

app = typer.Typer(pretty_exceptions_enable=False)

def load_sequence(sequence: str, device: torch.device = torch.device("cuda:0")):
    """
    Load default model and tokenizer and get sequence features
    """

    device = torch.device(device)

    # Load the model and tokenizer
    esm_model, esm_tokenizers = get_model(), get_tokenizers()
    esm_model = esm_model.to("cuda:0")

    # Get the sequence features
    sequence_features = esm3_sequence(
        sequence=sequence,
        model=esm_model,
        tokenizers=esm_tokenizers,
    )[:, 1:-1, :]

    return sequence_features


def load_structure(
    structure: bs.AtomArray,
    device: torch.device = torch.device("cuda:0"),
    stage: str = "encoded",
):
    """
    Load default model and tokenizer and get structure features
    """

    device = torch.device(device)

    # Load the model and tokenizer
    esm_structure_model = get_structure_vae()
    esm_structure_model = esm_structure_model.to(device)

    # Get the structure features
    chain = ProteinChain.from_atomarray(structure)
    structure_features = esm3_vqvae(
        chain=chain,
        esm_struct_encoder=esm_structure_model,
        stage=stage,
    )

    return structure_features

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    pdb_path: Path,
    run_id: str,
    model_path: str = "latest",
    device: str = "cuda:0",
    # -----------------------------------------
):
    # Run ID
    run_id = Path(run_id).resolve()

    # Set compute device
    device = torch.device(device)

    # Load the model
    model = RocketSHP.load_from_checkpoint(model_path).to(device)

    # Load structure file (PDB)
    structure = pdb.PDBFile.read(pdb_path).get_structure()
    struct_features = load_structure(structure, device=device)

    # Get sequence from structure
    sequence = str(to_sequence(structure)[0][0])
    seq_features = load_sequence(sequence, device=device)

    # Predict dynamics with both sequence and structure
    with torch.no_grad():
        dynamics_pred = model({
            "seq_feats": seq_features,
            "struct_feats": struct_features,
        })

    # Access prediction results
    rmsf = dynamics_pred["rmsf"].squeeze().cpu().numpy()
    gcc_lmi = dynamics_pred["gcc_lmi"].squeeze().cpu().numpy()
    ca_dist = dynamics_pred["ca_dist"].squeeze().cpu().numpy()
    shp = softmax(dynamics_pred["shp"].squeeze(), dim=1).cpu().numpy()

    # Visualize results
    plot_predictions(rmsf, gcc_lmi, shp, run_id.stem, run_id.with_suffix(".png"))

    # Save results to file
    torch.save(dynamics_pred, run_id.with_suffix(".pt"))
    print(f"Results saved to {run_id.with_suffix('.pt')}")

def __app__():
    app()


if __name__ == "__main__":
    app()
