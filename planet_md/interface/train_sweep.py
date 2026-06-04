import json
import os
import random
import warnings
from functools import partial
from itertools import product
from typing import Any

import dotenv
import pandas as pd
import torch
import typer
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from loguru import logger as stdout_logger
from omegaconf import OmegaConf

import wandb
from planet_md.config import DEFAULT_PARAMETERS, PROCESSED_DATA_DIR
from planet_md.data.atlas import ATLASDataModule
from planet_md.data.mdcath import MDCathDataModule
from planet_md.modeling.architectures import PlanetMDModel
from planet_md.modeling.pt_lightning import LightningWrapper
from planet_md.utils import configure_logger, seed_everything

warnings.simplefilter(action="ignore", category=FutureWarning)

app = typer.Typer(pretty_exceptions_enable=False)


def generate_sweep_configs(
    sweep_config_path: str, count: int = 50
) -> list[dict[str, Any]]:
    """Generate hyperparameter configurations from sweep definition."""

    with open(sweep_config_path) as f:
        sweep_def = yaml.safe_load(f)

    parameters = sweep_def.get("parameters", {})
    strategy = sweep_def.get("strategy", {}).get("type", "random")

    configs = []

    if strategy == "grid":
        # Generate all combinations for grid search
        param_names = []
        param_values = []

        for param_name, param_def in parameters.items():
            param_names.append(param_name)
            if param_def["type"] == "categorical":
                param_values.append(param_def["values"])
            elif param_def["type"] == "double":
                # For grid search, create a reasonable number of values
                min_val = float(param_def["min"])
                max_val = float(param_def["max"])
                scale = param_def.get("scale", "linear")

                if scale == "log":
                    import numpy as np

                    values = np.logspace(
                        np.log10(min_val), np.log10(max_val), 5
                    ).tolist()
                else:
                    values = [min_val + i * (max_val - min_val) / 4 for i in range(5)]
                param_values.append(values)

        # Generate all combinations
        for combination in product(*param_values):
            config = dict(zip(param_names, combination))
            # Flatten model_config if present
            config = _flatten_model_config(config)
            configs.append(config)

        # Limit to requested count
        if len(configs) > count:
            configs = random.sample(configs, count)

    elif strategy == "random":
        # Generate random configurations
        for _ in range(count):
            config = {}
            for param_name, param_def in parameters.items():
                if param_def["type"] == "categorical":
                    config[param_name] = random.choice(param_def["values"])
                elif param_def["type"] == "double":
                    min_val = param_def["min"]
                    max_val = param_def["max"]
                    scale = param_def.get("scale", "linear")

                    if scale == "log":
                        import math

                        log_min = math.log10(min_val)
                        log_max = math.log10(max_val)
                        log_val = random.uniform(log_min, log_max)
                        config[param_name] = 10**log_val
                    else:
                        config[param_name] = random.uniform(min_val, max_val)

            # Flatten model_config if present
            config = _flatten_model_config(config)
            configs.append(config)

    return configs


def _flatten_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten model_config parameter into individual d_model and n_heads parameters."""
    if "model_config" in config:
        model_config = config.pop("model_config")
        if isinstance(model_config, dict):
            config.update(model_config)
    return config


@app.command()
def single_run(
    run_id: str,
    config: str | None = None,
    debug: bool = False,
    dataset: str = "atlas",
    sweep_params: str | None = None,  # JSON string of hyperparameters
):
    """Run a single training run, optionally as part of a sweep."""

    dotenv.load_dotenv()
    assert dataset in ["atlas", "mdcath"], "Dataset must be either 'atlas' or 'mdcath'"

    PARAMS = DEFAULT_PARAMETERS.copy()

    # Load base config
    if config:
        CFG_PARAMS = OmegaConf.load(config)
        PARAMS.update(CFG_PARAMS)

    # If we have sweep parameters, parse and apply them
    if sweep_params:
        try:
            sweep_param_dict = json.loads(sweep_params)
            PARAMS.update(sweep_param_dict)
            run_name = f"{run_id}_sweep"
            is_sweep = True
        except json.JSONDecodeError:
            stdout_logger.error(f"Failed to parse sweep_params: {sweep_params}")
            return float("inf")
    else:
        run_name = run_id
        is_sweep = False

    stdout_logger.info(f"Parameters: {PARAMS}")

    # Initialize W&B run
    wandb_run = None
    if not debug:
        tags = [dataset]
        if is_sweep:
            tags.append("sweep")

        wandb_run = wandb.init(
            project=PARAMS.get("wandb_project", "RocketSHP"),
            entity=PARAMS.get("wandb_entity", None),
            name=run_name,
            tags=tags,
            config=PARAMS.__dict__,
        )

    loggers = []
    seed_everything(PARAMS.random_seed)
    torch.multiprocessing.set_sharing_strategy("file_system")

    if debug:
        os.environ["LOGURU_LEVEL"] = "DEBUG"
        configure_logger("DEBUG")
        stdout_logger.debug("Running in debug mode")
    else:
        if wandb_run:
            wandb_logger = WandbLogger(experiment=wandb_run, log_model=False)
            loggers.append(wandb_logger)
        configure_logger("INFO")

    loggers.append(CSVLogger("logs", name=run_name))

    # Create model with hyperparameters
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

    lightning_model = LightningWrapper(model, PARAMS)

    # Hyperparams already logged via wandb.init(config=...)

    torch.set_float32_matmul_precision(PARAMS.precision)

    # Data module setup
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

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename=run_name + "/model-{epoch:02d}-{val_loss:.5f}.pt",
        monitor="val_loss",
        mode="min",
        save_top_k=1 if is_sweep else 3,  # Save fewer models for sweep runs
        save_last=not is_sweep,
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, mode="min", verbose=True
    )

    trainer = Trainer(
        logger=loggers,
        max_epochs=PARAMS.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        accumulate_grad_batches=PARAMS.batch_size,
        gradient_clip_val=1.0,
        log_every_n_steps=50 if is_sweep else 5,
        enable_progress_bar=not is_sweep,  # Disable progress bar for sweeps
    )

    trainer.fit(lightning_model, datamodule=datamod)

    val_loss = trainer.callback_metrics.get("val_loss", float("inf"))

    # Log final metrics to W&B
    if not debug and wandb_run:
        wandb.log({"final/val_loss": val_loss})
        wandb.log(
            {
                "final/train_loss": trainer.callback_metrics.get(
                    "train_loss", float("inf")
                )
            }
        )
        wandb.finish()

    return val_loss


@app.command()
def generate_sweep(
    sweep_config: str, output_file: str, count: int = 50, name: str | None = None
):
    """Generate sweep configuration file with hyperparameter combinations."""

    configs = generate_sweep_configs(sweep_config, count)

    sweep_data = {
        "name": name or f"sweep_{os.path.basename(sweep_config).split('.')[0]}",
        "total_configs": len(configs),
        "configs": configs,
    }

    with open(output_file, "w") as f:
        json.dump(sweep_data, f, indent=2)

    stdout_logger.info(f"Generated {len(configs)} sweep configurations")
    stdout_logger.info(f"Sweep data saved to: {output_file}")
    stdout_logger.info(
        f"To run sweep, use: python -m rocketshp.interface.train_sweep run-sweep-from-file {output_file}"
    )

    return output_file


@app.command()
def run_sweep_from_file(
    sweep_file: str,
    job_index: int = 0,
    dataset: str = "atlas",
    config: str | None = None,
):
    """Run a single job from a sweep file (useful for SLURM array jobs)."""

    with open(sweep_file) as f:
        sweep_data = json.load(f)

    configs = sweep_data["configs"]

    if job_index >= len(configs):
        stdout_logger.error(
            f"Job index {job_index} out of range. Total configs: {len(configs)}"
        )
        return

    sweep_params = configs[job_index]
    run_id = f"{sweep_data['name']}_job_{job_index:03d}"

    stdout_logger.info(f"Running sweep job {job_index + 1}/{len(configs)}")
    stdout_logger.info(f"Hyperparameters: {sweep_params}")

    val_loss = single_run(
        run_id=run_id,
        config=config,
        dataset=dataset,
        sweep_params=json.dumps(sweep_params),
    )

    stdout_logger.info(f"Completed {run_id} with val_loss: {val_loss}")
    return val_loss


@app.command()
def run_local_sweep(
    sweep_file: str,
    max_jobs: int = 5,
    dataset: str = "atlas",
    config: str | None = None,
):
    """Run multiple sweep jobs locally (for testing)."""

    with open(sweep_file) as f:
        sweep_data = json.load(f)

    configs = sweep_data["configs"]
    total_jobs = min(max_jobs, len(configs))

    results = []

    for i in range(total_jobs):
        stdout_logger.info(f"Running local sweep job {i + 1}/{total_jobs}")

        try:
            val_loss = run_sweep_from_file(
                sweep_file=sweep_file, job_index=i, dataset=dataset, config=config
            )
            results.append(
                {"job_index": i, "val_loss": val_loss, "status": "completed"}
            )

        except Exception as e:
            stdout_logger.error(f"Failed job {i}: {e}")
            results.append(
                {"job_index": i, "val_loss": float("inf"), "status": "failed"}
            )

    # Print summary
    successful_jobs = [r for r in results if r["status"] == "completed"]
    if successful_jobs:
        best_job = min(successful_jobs, key=lambda x: x["val_loss"])
        stdout_logger.info(
            f"Best job: {best_job['job_index']} with val_loss: {best_job['val_loss']:.4f}"
        )

    return results


@app.command()
def analyze_sweep(
    sweep_name: str,
    init_config: str,
    new_config: str | None = None,
    output_table: str | None = None,
):
    """Analyze results from a W&B sweep by fetching runs with sweep tag."""

    dotenv.load_dotenv()

    # Load config to get W&B project and entity
    PARAMS = DEFAULT_PARAMETERS
    if init_config:
        CFG_PARAMS = OmegaConf.load(init_config)
        PARAMS.update(CFG_PARAMS)

    api = wandb.Api()
    entity = PARAMS.get("wandb_entity")
    project = PARAMS.get("wandb_project", "RocketSHP")

    # Construct project path
    if entity:
        project_path = f"{entity}/{project}"
    else:
        project_path = project

    # Get all runs with sweep tag
    runs = api.runs(project_path, filters={"tags": "sweep"})

    # Filter by sweep name if provided
    if sweep_name:
        runs = [run for run in runs if sweep_name in run.name]

    if len(runs) == 0:
        stdout_logger.info(f"No sweep runs found matching: {sweep_name}")
        return None

    # Build dataframe from runs
    data = []
    for run in runs:
        row = {
            "id": run.id,
            "name": run.name,
            "final/val_loss": run.summary.get("final/val_loss"),
            "final/train_loss": run.summary.get("final/train_loss"),
            "lr": run.config.get("lr"),
            "batch_size": run.config.get("batch_size"),
            "d_model": run.config.get("d_model"),
            "n_heads": run.config.get("n_heads"),
            "n_layers": run.config.get("n_layers"),
            "rmsf_alpha": run.config.get("rmsf_alpha"),
            "ca_dist_alpha": run.config.get("ca_dist_alpha"),
            "gcc_lmi_alpha": run.config.get("gcc_lmi_alpha"),
            "shp_alpha": run.config.get("shp_alpha"),
        }
        data.append(row)

    runs_table = pd.DataFrame(data)

    # Sort by validation loss
    runs_table = runs_table.sort_values("final/val_loss")

    stdout_logger.info(f"\nSweep '{sweep_name}' Results:")
    stdout_logger.info(f"Total runs: {len(runs_table)}")
    stdout_logger.info("\nTop 10 runs:")
    stdout_logger.info(runs_table.head(10))

    # Find best hyperparameters
    best_run = runs_table.iloc[0]
    stdout_logger.info(f"\nBest run ID: {best_run['id']}")
    stdout_logger.info(f"Best run name: {best_run['name']}")
    stdout_logger.info(f"Best val_loss: {best_run['final/val_loss']:.4f}")
    stdout_logger.info("\nBest hyperparameters:")
    for param in [
        "lr",
        "batch_size",
        "d_model",
        "n_heads",
        "n_layers",
        "rmsf_alpha",
        "ca_dist_alpha",
        "gcc_lmi_alpha",
        "shp_alpha",
    ]:
        if param in best_run and pd.notna(best_run[param]):
            stdout_logger.info(f"  {param}: {best_run[param]}")

    if output_table:
        runs_table.to_csv(output_table, index=False)
        stdout_logger.info(f"\nResults saved to: {output_table}")

    if new_config is None:
        new_config = f"configs/{sweep_name}_best.yaml"

    # Save best hyperparameters to new config file
    prev_config = OmegaConf.load(init_config)
    best_hyperparams = {
        "lr": float(best_run["lr"]),
        "batch_size": int(best_run["batch_size"]),
        "d_model": int(best_run["d_model"]),
        "n_heads": int(best_run["n_heads"]),
        "n_layers": int(best_run["n_layers"]),
        "rmsf_alpha": float(best_run["rmsf_alpha"]),
        "ca_dist_alpha": float(best_run["ca_dist_alpha"]),
        "gcc_lmi_alpha": float(best_run["gcc_lmi_alpha"]),
        "shp_alpha": float(best_run["shp_alpha"]),
    }
    prev_config.update(best_hyperparams)
    with open(new_config, "w") as f:
        yaml.dump(OmegaConf.to_container(prev_config), f, default_flow_style=False)

    return runs_table


def __app__():
    app()


if __name__ == "__main__":
    app()
