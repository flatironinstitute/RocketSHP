import logging
import os

import dotenv
import neptune
import torch
import typer
from loguru import logger as stdout_logger
from omegaconf import OmegaConf

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, NeptuneLogger

from rocketshp.config import PROCESSED_DATA_DIR, DEFAULT_PARAMETERS
from rocketshp.dataset import ATLASDataModule, ATLASDataset
from rocketshp.modeling.architectures import FlexibilityModel
from rocketshp.modeling.pt_lightning import LightningWrapper


class _FilterCallback(logging.Filterer):
    def filter(self, record: logging.LogRecord):
        return not (
            record.name == "neptune"
            and record.getMessage().startswith(
                "Error occurred during asynchronous operation processing: X-coordinates (step) must be strictly increasing for series attribute"
            )
        )


neptune.internal.operation_processors.async_operation_processor.logger.addFilter(
    _FilterCallback()
)

app = typer.Typer()


@app.command()
def main(
    run_id: str,
    config: str | None = None,
    debug: bool = False
):
    dotenv.load_dotenv()

    PARAMS = DEFAULT_PARAMETERS
    if config:
        CFG_PARAMS = OmegaConf.load(config)
        PARAMS.update(CFG_PARAMS)
    stdout_logger.info(PARAMS)

    loggers = []
    if not debug:
        neptune_logger = NeptuneLogger(
            project="samsl-flatiron/RocketSHP",
            name=run_id,
            api_key=os.getenv("NEPTUNE_API_TOKEN"),
            log_model_checkpoints=True,
        )
        loggers.append(neptune_logger)
    loggers.append(CSVLogger("logs", name=run_id))

    model = FlexibilityModel(
        embedding_dim = PARAMS.embedding_dim,
        output_dim = PARAMS.output_dim,
        d_model = PARAMS.d_model,
        n_heads = PARAMS.n_heads,
        n_layers = PARAMS.n_layers,
    )
    PARAMS.num_parameters = model._num_parameters()

    stdout_logger.info(model)
    lightning_model = LightningWrapper(model, PARAMS)
    
    if not debug: neptune_logger.log_hyperparams(params=PARAMS.__dict__)
    torch.set_float32_matmul_precision(PARAMS.precision)

    adl = ATLASDataModule(
        processed_h5=PROCESSED_DATA_DIR / "atlas/atlas_processed.h5",
        target="rmsf",
        seq_features=True,
        struct_features=True,
        batch_size=PARAMS.batch_size,
        num_workers=PARAMS.num_data_workers,
        shuffle=PARAMS.shuffle,
    )

    # Set up ModelCheckpoint to monitor val_loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=run_id
        + "/model-{epoch:02d}-{train_loss:.2f}.pt",  # Using exact metric name
        monitor="val_loss",  # Matches the exact metric name used in log_metrics
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    trainer = Trainer(
        logger=loggers,
        max_epochs=PARAMS.max_epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=5
    )
    trainer.fit(lightning_model, datamodule=adl)
    trainer.test(lightning_model, datamodule=adl)

if __name__ == "__main__":
    app()
