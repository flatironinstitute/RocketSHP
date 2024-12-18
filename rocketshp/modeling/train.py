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
from rocketshp.datasets.atlas import ATLASDataModule
from rocketshp.datasets.mdcath import MDCathDataModule
from rocketshp.modeling.architectures import FlexibilityModelWithTemperature, DynCorrModelWithTemperature
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

app = typer.Typer()#pretty_exceptions_enable=False)


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
    if debug:
        os.environ["LOGURU_LEVEL"] = "DEBUG"
        def simple_repr(self):
            return f"Tensor(size={list(self.size())})"

        torch.Tensor.__repr__ = simple_repr
        torch.Tensor.__str__ = simple_repr

        PARAMS.epoch_scale = 100
    else:
        neptune_logger = NeptuneLogger(
            project="samsl-flatiron/RocketSHP",
            name=run_id,
            api_key=os.getenv("NEPTUNE_API_TOKEN"),
            log_model_checkpoints=True,
        )
        loggers.append(neptune_logger)
        os.environ["LOGURU_LEVEL"] = "INFO"

    loggers.append(CSVLogger("logs", name=run_id))

    # model = FlexibilityModelWithTemperature(
    model = DynCorrModelWithTemperature(
        embedding_dim = PARAMS.embedding_dim,
        output_dim = PARAMS.output_dim,
        d_model = PARAMS.d_model,
        n_heads = PARAMS.n_heads,
        n_layers = PARAMS.n_layers,
        seq_only = not PARAMS.struct_features,
    )
    PARAMS.num_parameters = model._num_parameters()

    stdout_logger.info(model)
    lightning_model = LightningWrapper(model, PARAMS)
    
    if not debug: neptune_logger.log_hyperparams(params=PARAMS.__dict__)
    torch.set_float32_matmul_precision(PARAMS.precision)

    datamod = ATLASDataModule(
        processed_h5=PROCESSED_DATA_DIR / "atlas/atlas_processed.h5",
    # datamod = MDCathDataModule(
        # processed_h5=PROCESSED_DATA_DIR / "mdcath/mdcath_processed.h5",
        seq_features=PARAMS.seq_features,
        struct_features=PARAMS.struct_features,
        batch_size=PARAMS.batch_size,
        num_workers=PARAMS.num_data_workers,
        shuffle=PARAMS.shuffle,
        random_seed=PARAMS.random_seed,
        train_pct=PARAMS.train_pct,
        val_pct=PARAMS.val_pct,
    )

    # Set up ModelCheckpoint to monitor val_loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
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
        max_steps=PARAMS.epoch_scale,
        callbacks=[checkpoint_callback],
        log_every_n_steps=5
    )
    trainer.fit(lightning_model, datamodule=datamod)
    trainer.test(lightning_model, datamodule=datamod)

if __name__ == "__main__":
    app()
