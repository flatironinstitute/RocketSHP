from dataclasses import dataclass

import numpy as np
import torch
from lightning import LightningModule
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import functional as F
from loguru import logger as stdout_logger
from omegaconf import OmegaConf
from functools import partial

def compute_masked_mse_loss(outputs, labels, lengths, rmse=False, pad_value=0.0):
   # Create sequence mask based on lengths 
   mask = (torch.arange(outputs.shape[1], device=outputs.device)[None, :] < lengths[:, None])
   mask = mask.unsqueeze(-1)  # Add feature dimension
   
   # Use built-in MSELoss with reduction='none' to get per-element loss
   mse_loss = nn.MSELoss(reduction='none')
   loss = mse_loss(outputs, labels)
   
   # Apply mask and compute mean over valid elements
   masked_loss = loss * mask
   mse = masked_loss.sum() / mask.sum().clamp(min=1)
   
   return torch.sqrt(mse) if rmse else mse

# def compute_masked_mse_loss(outputs, labels, lengths, pad_value=0.0, reduction='mean'):
#     """
#     Compute MSE loss with masking for variable length sequences.
    
#     Parameters:
#     -----------
#     outputs : torch.Tensor
#         Model predictions, shape (batch_size, max_seq_len, feature_dim)
#     labels : torch.Tensor
#         Ground truth labels, shape (batch_size, max_seq_len, feature_dim)
#     lengths : torch.Tensor
#         Actual sequence lengths, shape (batch_size,)
#     pad_value : float
#         Value used for padding in labels
#     reduction : str
#         Reduction method: 'none', 'mean', or 'sum'
        
#     Returns:
#     --------
#     loss : torch.Tensor
#         Computed loss (scalar if reduction='mean'/'sum', otherwise tensor of shape (batch_size, max_seq_len, feature_dim))
#     """
#     # Validate inputs
#     assert outputs.shape == labels.shape, f"Shape mismatch: outputs {outputs.shape}, labels {labels.shape}"
#     assert lengths.shape[0] == outputs.shape[0], f"Batch size mismatch: lengths {lengths.shape}, outputs {outputs.shape}"
    
#     _, max_seq_len, _ = outputs.shape
    
#     # Create sequence mask based on lengths
#     # Shape: (batch_size, max_seq_len, 1)
#     seq_mask = (torch.arange(max_seq_len, device=outputs.device)[None, :] < lengths[:, None])
#     seq_mask = seq_mask.unsqueeze(-1)
    
#     # Create padding mask based on pad_value
#     # Shape: (batch_size, max_seq_len, feature_dim)
#     pad_mask = (labels != pad_value)
    
#     # Combine masks and expand to feature dimension if needed
#     # Shape: (batch_size, max_seq_len, feature_dim)
#     mask = seq_mask & pad_mask
    
#     # Compute MSE loss
#     squared_diff = (outputs - labels) ** 2
    
#     # Apply mask
#     masked_squared_diff = squared_diff * mask.float()
    
#     # Apply reduction
#     if reduction == 'none':
#         return masked_squared_diff
#     elif reduction == 'sum':
#         return masked_squared_diff.sum()
#     elif reduction == 'mean':
#         # Compute mean only over valid positions
#         return masked_squared_diff.sum() / mask.float().sum().clamp(min=1)
#     else:
#         raise ValueError(f"Unknown reduction method: {reduction}")

class LightningWrapper(LightningModule):
    def __init__(self, model: nn.Module, params: OmegaConf):
        super().__init__()
        self.child_model = model
        self.save_hyperparameters(dict(params))
        self.training_step_outputs = []
        self.validation_step_outputs = []

        if params.rmsf_loss == "mse":
            self.rmsf_loss = partial(compute_masked_mse_loss, rmse=False)
        elif params.rmsf_loss == "rmse":
            self.rmsf_loss = partial(compute_masked_mse_loss, rmse=True)

    def forward(self, x):
        return self.child_model(x)

    def training_step(self, batch, batch_idx):
        (seq, struct), mask, y = batch
        y_hat = self((seq, struct))
        loss = compute_masked_mse_loss(y_hat, y.unsqueeze(2), mask)

        self.log_dict({"batch_loss": loss}, on_step=True, on_epoch=False)
        self.log_dict({"train_loss": loss}, on_step=False, on_epoch=True)

        return {"loss": loss}

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        (seq, struct), mask, y = batch
        y_hat = self((seq, struct))
        loss = compute_masked_mse_loss(y_hat, y.unsqueeze(2), mask)

        self.log_dict(
            {"val_loss": loss}, on_epoch=True, on_step=False
        )

        return {"loss": loss}

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        (seq, struct), mask, y = batch
        y_hat = self((seq, struct))
        loss = self.rmsf_loss(y_hat, y.unsqueeze(2), mask)

        self.log_dict(
            {"test_loss": loss}, on_epoch=True, on_step=False
        )

        # y_true = y.cpu().detach().numpy()
        # y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()

        # rdict = {"loss": loss, "y_true": y_true, "y_pred": y_pred}
        # self.validation_step_outputs.append(rdict)

        return {"loss": loss}

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _num_parameters(self, requires_grad: bool = True):
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad == requires_grad
        )
