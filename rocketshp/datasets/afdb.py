import os
from pathlib import Path

import foldcomp
import typer

from rocketshp.config import RAW_DATA_DIR

app = typer.Typer()


@app.command()
def download(
    subset: str = "uniprot",
    version: int = 4,
    target_directory: Path = f"{RAW_DATA_DIR}/afdb_uniprot_v4",
):
    os.chdir(target_directory)
    foldcomp.setup(f"afdb_{subset}_v{version}")


if __name__ == "__main__":
    app()
