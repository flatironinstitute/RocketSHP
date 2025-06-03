import logging
import os
import warnings
from functools import partial
import json
import random
from itertools import product
from typing import Dict, Any, List

import dotenv
import neptune
import torch
import typer
import pandas as pd
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, NeptuneLogger
from loguru import logger as stdout_logger
from omegaconf import OmegaConf
import yaml

from rocketshp.config import DEFAULT_PARAMETERS, PROCESSED_DATA_DIR
from rocketshp.data.atlas import ATLASDataModule
from rocketshp.data.mdcath import MDCathDataModule
from rocketshp.modeling.architectures import RocketSHPModel
from rocketshp.modeling.pt_lightning import LightningWrapper
from rocketshp.utils import configure_logger, seed_everything

warnings.simplefilter(action="ignore", category=FutureWarning)

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

app = typer.Typer(pretty_exceptions_enable=False)

def generate_sweep_configs(sweep_config_path: str, count: int = 50) -> List[Dict[str, Any]]:
    """Generate hyperparameter configurations from sweep definition."""
    
    with open(sweep_config_path, 'r') as f:
        sweep_def = yaml.safe_load(f)
    
    parameters = sweep_def.get('parameters', {})
    strategy = sweep_def.get('strategy', {}).get('type', 'random')
    
    configs = []
    
    if strategy == 'grid':
        # Generate all combinations for grid search
        param_names = []
        param_values = []
        
        for param_name, param_def in parameters.items():
            param_names.append(param_name)
            if param_def['type'] == 'categorical':
                param_values.append(param_def['values'])
            elif param_def['type'] == 'double':
                # For grid search, create a reasonable number of values
                min_val = float(param_def['min'])
                max_val = float(param_def['max'])
                scale = param_def.get('scale', 'linear')
                
                if scale == 'log':
                    import numpy as np
                    values = np.logspace(np.log10(min_val), np.log10(max_val), 5).tolist()
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
            
    elif strategy == 'random':
        # Generate random configurations
        for _ in range(count):
            config = {}
            for param_name, param_def in parameters.items():
                if param_def['type'] == 'categorical':
                    config[param_name] = random.choice(param_def['values'])
                elif param_def['type'] == 'double':
                    min_val = param_def['min']
                    max_val = param_def['max']
                    scale = param_def.get('scale', 'linear')
                    
                    if scale == 'log':
                        import math
                        log_min = math.log10(min_val)
                        log_max = math.log10(max_val)
                        log_val = random.uniform(log_min, log_max)
                        config[param_name] = 10 ** log_val
                    else:
                        config[param_name] = random.uniform(min_val, max_val)
            
            # Flatten model_config if present
            config = _flatten_model_config(config)
            configs.append(config)
    
    return configs

def _flatten_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten model_config parameter into individual d_model and n_heads parameters."""
    if 'model_config' in config:
        model_config = config.pop('model_config')
        if isinstance(model_config, dict):
            config.update(model_config)
    return config

@app.command()
def single_run(
    run_id: str, 
    config: str | None = None, 
    debug: bool = False, 
    dataset: str = "atlas",
    sweep_params: str | None = None  # JSON string of hyperparameters
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
            return float('inf')
    else:
        run_name = run_id
        is_sweep = False

    stdout_logger.info(f"Parameters: {PARAMS}")

    # Initialize Neptune run
    run_for_logger = None
    if not debug:
        tags = [dataset]
        if is_sweep:
            tags.append("sweep")
            
        run_for_logger = neptune.init_run(
            project="samsl-flatiron/RocketSHP-hparams",
            name=run_name,
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
            tags=tags,
        )

    loggers = []
    seed_everything(PARAMS.random_seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    if debug:
        os.environ["LOGURU_LEVEL"] = "DEBUG"
        configure_logger("DEBUG")
        stdout_logger.debug("Running in debug mode")
    else:
        if run_for_logger:
            neptune_logger = NeptuneLogger(run=run_for_logger, log_model_checkpoints=False)
            loggers.append(neptune_logger)
        configure_logger("INFO")

    loggers.append(CSVLogger("logs", name=run_name))

    # Create model with hyperparameters
    model = RocketSHPModel(
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

    if not debug and run_for_logger:
        neptune_logger.log_hyperparams(params=PARAMS.__dict__)
    
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
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True
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
    
    val_loss = trainer.callback_metrics.get("val_loss", float('inf'))
    
    # Log final metrics to Neptune
    if not debug and run_for_logger:
        run_for_logger["final/val_loss"] = val_loss
        run_for_logger["final/train_loss"] = trainer.callback_metrics.get("train_loss", float('inf'))
        run_for_logger.stop()

    return val_loss

@app.command()
def generate_sweep(
    sweep_config: str,
    output_file: str,
    count: int = 50,
    name: str | None = None
):
    """Generate sweep configuration file with hyperparameter combinations."""
    
    configs = generate_sweep_configs(sweep_config, count)
    
    sweep_data = {
        'name': name or f"sweep_{os.path.basename(sweep_config).split('.')[0]}",
        'total_configs': len(configs),
        'configs': configs
    }
    
    with open(output_file, 'w') as f:
        json.dump(sweep_data, f, indent=2)
    
    stdout_logger.info(f"Generated {len(configs)} sweep configurations")
    stdout_logger.info(f"Sweep data saved to: {output_file}")
    stdout_logger.info(f"To run sweep, use: python -m rocketshp.interface.train_sweep run-sweep-from-file {output_file}")
    
    return output_file

@app.command()
def run_sweep_from_file(
    sweep_file: str,
    job_index: int = 0,
    dataset: str = "atlas",
    config: str | None = None
):
    """Run a single job from a sweep file (useful for SLURM array jobs)."""
    
    with open(sweep_file, 'r') as f:
        sweep_data = json.load(f)
    
    configs = sweep_data['configs']
    
    if job_index >= len(configs):
        stdout_logger.error(f"Job index {job_index} out of range. Total configs: {len(configs)}")
        return
    
    sweep_params = configs[job_index]
    run_id = f"{sweep_data['name']}_job_{job_index:03d}"
    
    stdout_logger.info(f"Running sweep job {job_index + 1}/{len(configs)}")
    stdout_logger.info(f"Hyperparameters: {sweep_params}")
    
    val_loss = single_run(
        run_id=run_id,
        config=config,
        dataset=dataset,
        sweep_params=json.dumps(sweep_params)
    )
    
    stdout_logger.info(f"Completed {run_id} with val_loss: {val_loss}")
    return val_loss

@app.command()
def run_local_sweep(
    sweep_file: str,
    max_jobs: int = 5,
    dataset: str = "atlas",
    config: str | None = None
):
    """Run multiple sweep jobs locally (for testing)."""
    
    with open(sweep_file, 'r') as f:
        sweep_data = json.load(f)
    
    configs = sweep_data['configs']
    total_jobs = min(max_jobs, len(configs))
    
    results = []
    
    for i in range(total_jobs):
        stdout_logger.info(f"Running local sweep job {i + 1}/{total_jobs}")
        
        try:
            val_loss = run_sweep_from_file(
                sweep_file=sweep_file,
                job_index=i,
                dataset=dataset,
                config=config
            )
            results.append({'job_index': i, 'val_loss': val_loss, 'status': 'completed'})
            
        except Exception as e:
            stdout_logger.error(f"Failed job {i}: {e}")
            results.append({'job_index': i, 'val_loss': float('inf'), 'status': 'failed'})
    
    # Print summary
    successful_jobs = [r for r in results if r['status'] == 'completed']
    if successful_jobs:
        best_job = min(successful_jobs, key=lambda x: x['val_loss'])
        stdout_logger.info(f"Best job: {best_job['job_index']} with val_loss: {best_job['val_loss']:.4f}")
    
    return results

@app.command()
def analyze_sweep(
    sweep_name: str,
    init_config: str,
    new_config: str | None = None,
    output_table: str | None = None

):
    """Analyze results from a Neptune sweep by fetching runs with sweep tag."""
    
    dotenv.load_dotenv()
    
    project = neptune.init_project(
        project="samsl-flatiron/RocketSHP-hparams",
        api_token=os.getenv("NEPTUNE_API_TOKEN")
    )

    # Get all runs with sweep tag
    runs_table = project.fetch_runs_table(
        tag="sweep",
        columns=["sys/id", "sys/name", "final/val_loss", "final/train_loss", 
                "training/hyperparams/lr",
                "training/hyperparams/batch_size",
                "training/hyperparams/d_model",
                "training/hyperparams/n_heads",
                "training/hyperparams/n_layers", 
                "training/hyperparams/rmsf_alpha",
                "training/hyperparams/ca_dist_alpha",
                "training/hyperparams/gcc_lmi_alpha",
                "training/hyperparams/shp_alpha"
                ]
    ).to_pandas()
    
    # Filter for this specific sweep by name pattern
    if sweep_name:
        runs_table = runs_table[runs_table['sys/name'].str.contains(sweep_name, na=False)]
    
    if len(runs_table) == 0:
        stdout_logger.info(f"No sweep runs found matching: {sweep_name}")
        return None
    
    # Sort by validation loss
    runs_table = runs_table.sort_values("final/val_loss")
    
    stdout_logger.info(f"\nSweep '{sweep_name}' Results:")
    stdout_logger.info(f"Total runs: {len(runs_table)}")
    stdout_logger.info("\nTop 10 runs:")
    stdout_logger.info(runs_table.head(10))
    
    # Find best hyperparameters
    best_run = runs_table.iloc[0]
    stdout_logger.info(f"\nBest run ID: {best_run['sys/id']}")
    stdout_logger.info(f"Best run name: {best_run['sys/name']}")
    stdout_logger.info(f"Best val_loss: {best_run['final/val_loss']:.4f}")
    stdout_logger.info("\nBest hyperparameters:")
    for param in [
        "training/hyperparams/lr",
        "training/hyperparams/batch_size",
        "training/hyperparams/d_model",
        "training/hyperparams/n_heads",
        "training/hyperparams/n_layers", 
        "training/hyperparams/rmsf_alpha",
        "training/hyperparams/ca_dist_alpha",
        "training/hyperparams/gcc_lmi_alpha",
        "training/hyperparams/shp_alpha",
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
        "lr": float(best_run["training/hyperparams/lr"]),
        "batch_size": int(best_run["training/hyperparams/batch_size"]),
        "d_model": int(best_run["training/hyperparams/d_model"]),
        "n_heads": int(best_run["training/hyperparams/n_heads"]),
        "n_layers": int(best_run["training/hyperparams/n_layers"]),
        "rmsf_alpha": float(best_run["training/hyperparams/rmsf_alpha"]),
        "ca_dist_alpha": float(best_run["training/hyperparams/ca_dist_alpha"]),
        "gcc_lmi_alpha": float(best_run["training/hyperparams/gcc_lmi_alpha"]),
        "shp_alpha": float(best_run["training/hyperparams/shp_alpha"]),
    }
    prev_config.update(best_hyperparams)
    with open(new_config, 'w') as f:
        yaml.dump(OmegaConf.to_container(prev_config), f, default_flow_style=False)
    
    return runs_table

def __app__():
    app()

if __name__ == "__main__":
    app()