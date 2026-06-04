from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from planet_md.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def plot_predictions(
    rmsf: np.ndarray,
    gcc_lmi: np.ndarray,
    shp: np.ndarray,
    title: str = "PLANET-MD Predictions",
    output_path: Path = "planet_md_predictions.png",
    font_scale: float = 1.0,
):
    with plt.style.context(
        {
            "font.size": 12 * font_scale,
            "legend.fontsize": 12 * font_scale,
            "axes.labelsize": 12 * font_scale,
            "axes.titlesize": 12 * font_scale,
        }
    ):
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        fig.suptitle(title)

        ax1.plot(rmsf, label="RMSF")
        ax1.set_title("RMSF")
        ax1.set_xlabel("Residue Index")
        ax1.set_ylabel("RMSF (Å)")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        ax2.imshow(gcc_lmi, cmap="viridis", aspect="equal", vmin=0, vmax=1)
        ax2.set_title("GCC-LMI")
        ax2.set_xlabel("Residue Index")
        ax2.set_ylabel("Residue Index")

        ax3.imshow(shp.T, cmap="binary", vmin=0, vmax=1, interpolation="none")
        ax3.set_title("SHP")
        ax3.set_xlabel("Residue Index")
        ax3.set_ylabel("Structure Token\nIndex")
        ax3.set_ylim(21, -1)

        plt.tight_layout()
        plt.savefig(output_path)


if __name__ == "__main__":
    app()
