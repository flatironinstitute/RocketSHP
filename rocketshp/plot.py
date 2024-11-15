from pathlib import Path

import nglview as nv
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from rocketshp.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def value_to_hex(value, cmap="red-blue"):
    """Map value between 0 and 1 to hex color code"""
    # Ensure value is between 0 and 1
    value = max(0, min(1, value))

    if cmap == "red-blue":
        # Red (value=1) to Blue (value=0)
        red = int(255 * value)
        blue = int(255 * (1 - value))
        return f"#{red:02x}00{blue:02x}"

    elif cmap == "blue-red":
        # Blue (value=1) to Red (value=0)
        red = int(255 * (1 - value))
        blue = int(255 * value)
        return f"#{red:02x}00{blue:02x}"


def display_trajectory(traj, coloring="residueindex", bfactor=None, normalize=True, RMAX=1):
    view = nv.show_mdtraj(traj)
    view.clear()

    if coloring in nv.color.COLOR_SCHEMES and coloring != "bfactor":
        view.add_representation("cartoon", colorScheme=coloring)
        return view
    elif coloring == "bfactor":
        assert bfactor is not None
        if not isinstance(bfactor, np.ndarray):
            bfactor = np.array(bfactor)
        if normalize:
            bfactor = (bfactor.clip(0, RMAX) - bfactor.clip(0, RMAX).min()) / (
                bfactor.clip(0, RMAX).max() - bfactor.clip(0, RMAX).min()
            )

        from nglview.color import ColormakerRegistry

        cm = ColormakerRegistry

        scheme = []
        for i, value in enumerate(bfactor):
            scheme.append([value_to_hex(value), f"{i+1}"])

        cm.add_selection_scheme("_bfact", scheme)
        # cm.add_selection_scheme('_bfact', [["#b30000", '1-10'], ['yellow', '222-484']])

        view.add_representation("cartoon", color="_bfact")

        return view
    else:
        raise ValueError(f"Coloring scheme {coloring} not supported")


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
