"""Analyze sweep results and generate best config."""

import argparse
import neptune
import pandas as pd
from dotenv import load_dotenv
import os
import yaml
from pathlib import Path

load_dotenv()

def analyze_and_generate_config(sweep_id: str, output_dir: str = "configs/optimized"):
    """Analyze sweep results and generate optimized config file."""
    
    project = neptune.init_project(
        project="samsl-flatiron/RocketSHP",
        api_token=os.getenv("NEPTUNE_API_TOKEN")
    )
    
    # Get all runs from the sweep
    runs_table = project.fetch_runs_table(
        tag="sweep",
        columns=["sys/id", "final/val_loss", "final/train_loss", "lr", "batch_size", 
                "d_model", "n_heads", "n_layers", "rmsf_alpha", "ca_dist_alpha", 
                "gcc_lmi_alpha", "shp_alpha"]
    ).to_pandas()
    
    # Filter for this specific sweep
    runs_table = runs_table[runs_table['sys/id'].str.contains(sweep_id)]
    
    if len(runs_table) == 0:
        print(f"No runs found for sweep {sweep_id}")
        return
    
    # Sort by validation loss
    runs_table = runs_table.sort_values("final/val_loss")
    
    # Get best run
    best_run = runs_table.iloc[0]
    
    print(f"\nBest run from sweep {sweep_id}:")
    print(f"Run ID: {best_run['sys/id']}")
    print(f"Val Loss: {best_run['final/val_loss']:.4f}")
    
    # Create optimized config
    optimized_config = {
        'random_seed': 0,
        'num_data_workers': 31,
        'crop_size': 512,
        'shuffle': True,
        'train_pct': 0.8,
        'val_pct': 0.1,
        
        # Optimized parameters from sweep
        'lr': float(best_run['lr']) if pd.notna(best_run['lr']) else 0.00005,
        'batch_size': int(best_run['batch_size']) if pd.notna(best_run['batch_size']) else 16,
        'd_model': int(best_run['d_model']) if pd.notna(best_run['d_model']) else 512,
        'n_heads': int(best_run['n_heads']) if pd.notna(best_run['n_heads']) else 8,
        'n_layers': int(best_run['n_layers']) if pd.notna(best_run['n_layers']) else 8,
        
        # Loss weights if they were swept
        'rmsf_alpha': float(best_run['rmsf_alpha']) if pd.notna(best_run['rmsf_alpha']) else 1.0,
        'ca_dist_alpha': float(best_run['ca_dist_alpha']) if pd.notna(best_run['ca_dist_alpha']) else 1.0,
        'gcc_lmi_alpha': float(best_run['gcc_lmi_alpha']) if pd.notna(best_run['gcc_lmi_alpha']) else 1.0,
        'shp_alpha': float(best_run['shp_alpha']) if pd.notna(best_run['shp_alpha']) else 0.01,
        
        # Fixed parameters
        'embedding_dim': 1536,
        'output_dim': 1,
        'seq_features': True,
        'struct_features': True,
        'struct_stage': "encoded",
        'struct_dim': 1024,
        'max_epochs': 75,
        'epoch_scale': -1,
        'precision': "medium",
        'square_loss': False,
        'variance_norm': False,
        'grad_norm': False,
        'dyn_corr_alpha': 0.0,
        'autocorr_alpha': 0.0,
    }
    
    # Save optimized config
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / f"optimized_from_sweep_{sweep_id}.yml"
    
    with open(output_file, 'w') as f:
        yaml.dump(optimized_config, f, default_flow_style=False)
    
    print(f"\nOptimized config saved to: {output_file}")
    print("\nOptimized hyperparameters:")
    for key, value in optimized_config.items():
        if key in ['lr', 'batch_size', 'd_model', 'n_heads', 'n_layers', 'rmsf_alpha', 'ca_dist_alpha']:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_id", help="Neptune sweep ID")
    parser.add_argument("--output-dir", default="configs/optimized", help="Output directory for optimized config")
    
    args = parser.parse_args()
    analyze_and_generate_config(args.sweep_id, args.output_dir)