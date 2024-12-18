from dataclasses import dataclass

import numpy as np
import torch
import lightning as L
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import functional as F
from loguru import logger as stdout_logger
from omegaconf import OmegaConf
from functools import partial
from loguru import logger

from rocketshp.datasets.data_utils import _unstack_variable_length_tensors, _unstack_variable_size_squareforms

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

def compute_square_masked_mse_loss(outputs, labels, lengths, rmse=False, pad_value=0.0):
    linmask = (torch.arange(outputs.shape[1], device=outputs.device)[None, :] < lengths[:, None])
    squaremask = (linmask.unsqueeze(2) & linmask.unsqueeze(1))
    
    # Use built-in MSELoss with reduction='none' to get per-element loss
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(outputs, labels)

    # Apply mask and compute mean over valid elements
    masked_loss = loss * squaremask
    # mse = masked_loss.sum() / squaremask.sum().clamp(min=1)
    mse = (masked_loss.sum(axis=(1,2)) / squaremask.sum(axis=(1,2))).mean()

    return torch.sqrt(mse) if rmse else mse

class LightningWrapper(L.LightningModule):
    def __init__(self, model: nn.Module, params: OmegaConf):
        super().__init__()
        self.child_model = model
        self.save_hyperparameters(dict(params))
        self.training_step_outputs = []
        self.validation_step_outputs = []

        if params.square_loss:
            self.rmsf_loss_fn = partial(compute_masked_mse_loss, rmse=False)
            self.ca_loss_fn = partial(compute_square_masked_mse_loss, rmse=False)
            self.dyn_corr_loss_fn = partial(compute_square_masked_mse_loss, rmse=False)
        else:
            self.rmsf_loss_fn = partial(compute_masked_mse_loss, rmse=True)
            self.ca_loss_fn = partial(compute_square_masked_mse_loss, rmse=True)
            self.dyn_corr_loss_fn = partial(compute_square_masked_mse_loss, rmse=True)

        self.rmsf_alpha = params.rmsf_alpha
        self.ca_alpha = params.ca_alpha
        self.dyn_corr_alpha = params.dyn_corr_alpha
        self.variance_norm = params.variance_norm

    def forward(self, x):
        return self.child_model(x)

    def _get_loss(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x)

        return_dict = {}

        loss = 0
        if "rmsf" in y_hat:
            rmsf_loss = self.rmsf_loss_fn(y_hat["rmsf"], y["rmsf"].unsqueeze(2), mask)
            loss += self.rmsf_alpha * rmsf_loss
            return_dict["rmsf_loss"] = rmsf_loss
        if "ca_dist" in y_hat:
            ca_dist_loss = self.ca_loss_fn(y_hat["ca_dist"], y["ca_dist"], mask)
            loss += self.ca_alpha * ca_dist_loss
            return_dict["ca_loss"] = ca_dist_loss
        if "dyn_corr" in y_hat:
            dyn_corr_loss = self.dyn_corr_loss_fn(y_hat["dyn_corr"], y["dyn_corr"], mask)
            loss += self.dyn_corr_alpha * dyn_corr_loss
            return_dict["corr_loss"] = dyn_corr_loss

        return_dict["batch_loss"] = loss

        return return_dict
    
    def training_step(self, batch, batch_idx):
        loss_dict = self._get_loss(batch, batch_idx)

        self.log_dict(loss_dict, on_step=True, on_epoch=False)
        self.log_dict({"train_loss": loss_dict["batch_loss"]}, on_step=False, on_epoch=True
)
        return {"loss": loss_dict["batch_loss"]}

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        loss_dict = self._get_loss(batch, batch_idx)

        self.log_dict(
            {"val_loss": loss_dict["batch_loss"]}, on_epoch=True, on_step=False
        )

        return {"loss": loss_dict["batch_loss"]}

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        loss_dict = self._get_loss(batch, batch_idx)

        self.log_dict(
            {"test_loss": loss_dict["batch_loss"]}, on_epoch=True, on_step=False
        )

        return {"loss": loss_dict["batch_loss"]}

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _num_parameters(self, requires_grad: bool = True):
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad == requires_grad
        )
