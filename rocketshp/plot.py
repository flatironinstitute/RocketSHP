from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from rocketshp.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def plot_predictions(
    rmsf: np.ndarray,
    gcc_lmi: np.ndarray,
    shp: np.ndarray,
    title: str = "RocketSHP Predictions",
    output_path: Path = "rocketshp_predictions.png",
):
    # RMSF and GCC LMI on top, SHP wide on bottom
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])  # Top left
    ax2 = fig.add_subplot(gs[0, 1])  # Top right
    ax3 = fig.add_subplot(gs[1, :])  # Bottom (spans both columns)

    fig.suptitle(title)

    ax1.plot(rmsf, label="RMSF")
    ax1.set_title("RMSF")
    ax1.set_xlabel("Residue Index")
    ax1.set_ylabel("RMSF (Å)")

    ax2.imshow(gcc_lmi, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax2.set_title("GCC-LMI")
    ax2.set_xlabel("Residue Index")
    ax2.set_ylabel("Residue Index")

    ax3.imshow(shp.T, cmap="binary", vmin=0, vmax=1)
    ax3.set_title("SHP")
    ax3.set_xlabel("Residue Index")
    ax3.set_ylabel("Structure Token Index")
    ax3.set_ylim(21, -1)

    plt.tight_layout()
    plt.savefig(output_path)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
