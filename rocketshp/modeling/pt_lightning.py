from dataclasses import dataclass

import numpy as np
import torch
import lightning as L
from lightning.pytorch.utilities import grad_norm
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import functional as F
from loguru import logger as stdout_logger
from omegaconf import OmegaConf
from functools import partial
from loguru import logger

from rocketshp.datasets.utils import _unstack_variable_length_tensors, _unstack_variable_size_squareforms

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

        self.norm_grad = params.grad_norm
        self.rmsf_alpha = params.rmsf_alpha
        self.ca_alpha = params.ca_alpha
        self.dyn_corr_alpha = params.dyn_corr_alpha
        self.variance_norm = params.variance_norm

    def forward(self, x):
        return self.child_model(x)

    def _get_loss(self, batch, batch_idx, stage="train"):
        x, y, mask = batch
        y_hat = self(x)

        return_dict = {}

        if self.norm_grad and stage == "train":
            rmsf_loss = self.rmsf_loss_fn(y_hat["rmsf"], y["rmsf"].unsqueeze(2), mask)
            # return_dict["rmsf_loss"] = rmsf_loss

            ca_dist_loss = self.ca_loss_fn(y_hat["ca_dist"], y["ca_dist"], mask)
            # return_dict["ca_loss"] = ca_dist_loss

            dyn_corr_loss = self.dyn_corr_loss_fn(y_hat["dyn_corr"], y["dyn_corr"], mask)
            # return_dict["corr_loss"] = dyn_corr_loss

            weighted_loss, weighted_losses, grad_loss = self.child_model.grad_norm(
                [rmsf_loss, ca_dist_loss, dyn_corr_loss]
            )

            return_dict["rmsf_loss"] = rmsf_loss
            return_dict["ca_loss"] = ca_dist_loss
            return_dict["corr_loss"] = dyn_corr_loss
            return_dict["batch_loss"] = weighted_loss
            return_dict["grad_loss"] = grad_loss

        else:
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
        loss_dict = self._get_loss(batch, batch_idx, "train")
        # if "grad_loss" in loss_dict:
        #     total_loss = loss_dict["batch_loss"] + loss_dict["grad_loss"]
        # else:
        #     total_loss = loss_dict["batch_loss"]
        total_loss = loss_dict["batch_loss"]

        self.log_dict(loss_dict, on_step=True, on_epoch=False)           
        self.log_dict({"task_weights/rmsf": self.child_model.grad_norm.task_weights[0]}, on_step=True, on_epoch=False)
        self.log_dict({"task_weights/ca_dist": self.child_model.grad_norm.task_weights[1]}, on_step=True, on_epoch=False)
        self.log_dict({"task_weights/dyn_corr": self.child_model.grad_norm.task_weights[2]}, on_step=True, on_epoch=False)
        if batch_idx % 1000 == 0:
            logger.debug(f"rmsf_weight: {self.child_model.grad_norm.task_weights[0]:.3f}, ca_dist_weight: {self.child_model.grad_norm.task_weights[1]:.3f}, dyn_corr_weight: {self.child_model.grad_norm.task_weights[2]:.3f}, total_weights: {torch.sum(self.child_model.grad_norm.task_weights):.3f}")
            logger.debug(f"rmsf_weighted_loss: {self.child_model.grad_norm.task_weights[0]*loss_dict['rmsf_loss']:.3f}, ca_loss: {self.child_model.grad_norm.task_weights[1]*loss_dict['ca_loss']:.3f}, dyn_corr_loss: {self.child_model.grad_norm.task_weights[2]*loss_dict['corr_loss']:.3f}")
            # logger.debug(f"grad_loss: {loss_dict['grad_loss']:.3f}, total_loss: {total_loss:.3f}")

        self.log_dict({"train_loss": loss_dict["batch_loss"]}, on_step=False, on_epoch=True)

        return {"loss": total_loss}

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        loss_dict = self._get_loss(batch, batch_idx, "validation")

        self.log_dict(
            {"val_loss": loss_dict["batch_loss"]}, on_epoch=True, on_step=False
        )

        return {"loss": loss_dict["batch_loss"]}

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        loss_dict = self._get_loss(batch, batch_idx, "test")

        self.log_dict(
            {"test_loss": loss_dict["batch_loss"]}, on_epoch=True, on_step=False
        )

        return {"loss": loss_dict["batch_loss"]}

    def on_test_epoch_end(self):
        pass

    # def on_before_optimizer_step(self, optimizer):
    #     # Log gradient norms for all modules
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             self.log(f"grad_norm/{name}", grad_norm(param.grad))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _num_parameters(self, requires_grad: bool = True):
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad == requires_grad
        )
