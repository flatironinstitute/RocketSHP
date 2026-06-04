import os
import warnings
from functools import partial

import dotenv
import torch
import typer
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from loguru import logger as stdout_logger
from omegaconf import OmegaConf

from planet_md.config import DEFAULT_PARAMETERS, PROCESSED_DATA_DIR
from planet_md.data.atlas import ATLASDataModule
from planet_md.data.mdcath import MDCathDataModule
from planet_md.modeling.architectures import (
    PlanetMDModel,
)
from planet_md.modeling.pt_lightning import LightningWrapper
from planet_md.utils import configure_logger, seed_everything

warnings.simplefilter(action="ignore", category=FutureWarning)

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    run_id: str, config: str | None = None, debug: bool = False, dataset: str = "atlas"
):
    dotenv.load_dotenv()

    assert dataset in ["atlas", "mdcath"], "Dataset must be either 'atlas' or 'mdcath'"

    PARAMS = DEFAULT_PARAMETERS
    if config:
        CFG_PARAMS = OmegaConf.load(config)
        PARAMS.update(CFG_PARAMS)
    stdout_logger.info(PARAMS)

    loggers = []

    seed_everything(PARAMS.random_seed)
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Early sanity check on hyperparameters
    assert PARAMS.d_model % PARAMS.n_heads == 0, (
        f"d_model ({PARAMS.d_model}) must be divisible by n_heads ({PARAMS.n_heads})"
    )

    if debug:

        def simple_repr(self):
            if isinstance(self, torch.Tensor):
                return f"Tensor(size={list(self.size())},{self.device},grad={self.requires_grad})"
            elif isinstance(self, torch.nn.Module):
                return f"{self.__class__.__name__} with {sum(p.numel() for p in self.parameters())} parameters"
            else:
                return f"{self.__class__.__name__}(...) with {sum(p.numel() for p in self.parameters())} parameters"

        # torch.Tensor.__repr__ = simple_repr
        # torch.nn.Module.__repr__ = simple_repr
        # L.LightningModule.__repr__ = simple_repr

        # PARAMS.epoch_scale = 100
        os.environ["LOGURU_LEVEL"] = "DEBUG"
        configure_logger("DEBUG")
        stdout_logger.debug("Running in debug mode")

        # PARAMS.train_pct = 0.001

    else:
        # PARAMS.epoch_scale = -1
        wandb_logger = WandbLogger(
            project=PARAMS.get("wandb_project", "PLANET_MD"),
            entity=PARAMS.get("wandb_entity", None),
            name=run_id,
            log_model=True,
        )
        loggers.append(wandb_logger)

        configure_logger("INFO")

    loggers.append(CSVLogger("logs", name=run_id))

    model = PlanetMDModel(
        embedding_dim=PARAMS.embedding_dim,
        output_dim=PARAMS.output_dim,
        d_model=PARAMS.d_model,
        n_heads=PARAMS.n_heads,
        n_layers=PARAMS.n_layers,
        seq_only=not PARAMS.struct_features,
        struct_stage=PARAMS.struct_stage,
        struct_dim=PARAMS.struct_dim,
    )
    PARAMS.num_parameters = model._num_parameters()

    stdout_logger.info(model)
    lightning_model = LightningWrapper(model, PARAMS)

    if not debug:
        wandb_logger.log_hyperparams(params=PARAMS.__dict__)

    torch.set_float32_matmul_precision(PARAMS.precision)

    if dataset == "atlas":
        dmodule = partial(
            ATLASDataModule,
            processed_h5=PROCESSED_DATA_DIR / "atlas/atlas_processed.h5",
        )
    elif dataset == "mdcath":
        dmodule = partial(
            MDCathDataModule,
            processed_h5=PROCESSED_DATA_DIR / "mdcath/mdcath_processed.h5",
        )

    datamod = dmodule(
        # processed_h5=PROCESSED_DATA_DIR / "atlas/atlas_processed.h5",
        # datamod = MDCathDataModule(
        # processed_h5=PROCESSED_DATA_DIR / "mdcath/mdcath_processed.h5",
        seq_features=PARAMS.seq_features,
        struct_features=PARAMS.struct_features,
        batch_size=1,
        crop_size=PARAMS.crop_size,
        num_workers=PARAMS.num_data_workers,
        shuffle=PARAMS.shuffle,
        random_seed=PARAMS.random_seed,
        train_pct=PARAMS.train_pct,
        val_pct=PARAMS.val_pct,
        struct_stage=PARAMS.struct_stage,
    )

    # Set up ModelCheckpoint to monitor val_loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename=run_id
        + "/model-{epoch:02d}-{val_loss:.5f}.pt",  # Using exact metric name
        monitor="val_loss",  # Matches the exact metric name used in log_metrics
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    # lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        logger=loggers,
        max_epochs=PARAMS.max_epochs,
        # max_steps=PARAMS.epoch_scale,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=PARAMS.batch_size,
        gradient_clip_val=1.0,
        log_every_n_steps=5,
    )

    trainer.fit(lightning_model, datamodule=datamod)
    trainer.test(lightning_model, datamodule=datamod)


def __app__():
    app()


if __name__ == "__main__":
    app()
