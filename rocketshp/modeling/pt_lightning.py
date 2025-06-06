from functools import partial

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from loguru import logger


def compute_masked_mse_loss(outputs, labels, lengths, rmse=False, pad_value=0.0):
    # Create sequence mask based on lengths
    mask = (
        torch.arange(outputs.shape[1], device=outputs.device)[None, :]
        < lengths[:, None]
    )
    mask = mask.unsqueeze(-1)  # Add feature dimension

    # Use built-in MSELoss with reduction='none' to get per-element loss
    mse_loss = nn.MSELoss(reduction="none")
    loss = mse_loss(outputs, labels)

    # Apply mask and compute mean over valid elements
    masked_loss = loss * mask
    mse = masked_loss.sum() / mask.sum().clamp(min=1)

    return torch.sqrt(mse) if rmse else mse


def compute_square_masked_mse_loss(outputs, labels, lengths, rmse=False, pad_value=0.0):
    linmask = (
        torch.arange(outputs.shape[1], device=outputs.device)[None, :]
        < lengths[:, None]
    )
    squaremask = linmask.unsqueeze(2) & linmask.unsqueeze(1)

    # Use built-in MSELoss with reduction='none' to get per-element loss
    mse_loss = nn.MSELoss(reduction="none")
    loss = mse_loss(outputs, labels)

    # Apply mask and compute mean over valid elements
    masked_loss = loss * squaremask
    # mse = masked_loss.sum() / squaremask.sum().clamp(min=1)
    mse = (masked_loss.sum(axis=(1, 2)) / squaremask.sum(axis=(1, 2))).mean()

    return torch.sqrt(mse) if rmse else mse


def compute_categorical_masked_cross_entropy_loss(
    outputs, labels, lengths, pad_value=0.0
):
    # Create sequence mask based on lengths
    mask = (
        torch.arange(outputs.shape[1], device=outputs.device)[None, :]
        < lengths[:, None]
    )

    # Use built-in BCELoss with reduction='none' to get per-element loss
    ce_loss = nn.CrossEntropyLoss(reduction="none")
    loss = ce_loss(outputs.transpose(1, 2), labels.transpose(1, 2))

    # Apply mask and compute mean over valid elements
    masked_loss = loss * mask
    ce = masked_loss.sum() / (mask.sum().clamp(min=1))

    return ce


def compute_masked_kl_div_loss(outputs, labels, lengths, pad_value=0.0):
    # Create sequence mask based on lengths
    mask = (
        torch.arange(outputs.shape[1], device=outputs.device)[None, :]
        < lengths[:, None]
    ).unsqueeze(-1)

    outputs = F.log_softmax(outputs, dim=1)

    # KL divergence: sum(target * (log(target) - log(pred)))
    # We can ignore the log(target) term as it's constant w.r.t. optimization
    kld_loss = nn.KLDivLoss(reduction="none", log_target=False)
    kl_div = kld_loss(outputs, labels)
    masked_kl_div = kl_div * mask  # Apply mask

    # Sum over the distribution dimension (D)
    summed_kl_div = masked_kl_div.sum(dim=1)  # Now (B, L)

    # Take mean over batch and sequence length
    loss = summed_kl_div.mean()

    return loss


class LightningWrapper(L.LightningModule):
    def __init__(self, model: nn.Module, params: OmegaConf):
        super().__init__()
        self.child_model = model
        self.save_hyperparameters(dict(params))
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.rmsf_loss_fn = partial(
            compute_masked_mse_loss, rmse=not params.square_loss
        )
        self.ca_loss_fn = partial(
            compute_square_masked_mse_loss, rmse=not params.square_loss
        ) 
        self.gcc_loss_fn = partial(
            compute_square_masked_mse_loss, rmse=not params.square_loss
        )
        self.shp_loss_fn = compute_masked_kl_div_loss

        self.norm_grad = params.grad_norm
        self.rmsf_alpha = params.rmsf_alpha
        self.ca_dist_alpha = params.ca_dist_alpha
        self.gcc_lmi_alpha = params.gcc_lmi_alpha
        self.shp_alpha = params.shp_alpha
        self.variance_norm = params.variance_norm

        self.crop_size = 512

        self.validation_losses = []

    def forward(self, x):
        return self.child_model(x)

    def _get_loss(self, batch, batch_idx, stage="train"):
        x, y, mask = batch
        y_hat = self(x)

        return_dict = {}
        if "rmsf" in y_hat:
            rmsf_loss = self.rmsf_loss_fn(
                y_hat["rmsf"], y["rmsf"].unsqueeze(2), mask
            )
            return_dict["rmsf_loss"] = rmsf_loss
        if "ca_dist" in y_hat:
            ca_dist_loss = self.ca_loss_fn(y_hat["ca_dist"], y["ca_dist"], mask)
            return_dict["ca_loss"] = ca_dist_loss
        if "gcc_lmi" in y_hat:
            gcc_loss = self.gcc_loss_fn(y_hat["gcc_lmi"], y["gcc_lmi"], mask)
            return_dict["gcc_lmi_loss"] = gcc_loss
        if "shp" in y_hat:
            shp_loss = self.shp_loss_fn(y_hat["shp"], y["shp"], mask)
            return_dict["shp_loss"] = shp_loss

        if self.norm_grad:
            raise NotImplementedError("Gradient normalization is not implemented yet.")
        elif stage == "train":
            loss = 0
            loss += self.rmsf_alpha * rmsf_loss
            loss += self.shp_alpha * shp_loss
            loss += self.gcc_lmi_alpha * gcc_loss
            loss += self.ca_dist_alpha * ca_dist_loss
            return_dict["batch_loss"] = loss
        elif (stage == "validation") or (stage == "test"):
            # use unweighted geometric mean of each individual relevant loss to measure overall performance
            loss = (
                rmsf_loss*shp_loss*gcc_loss
            ) ** (1 / 3)
            return_dict["batch_loss"] = loss
        else:
            raise ValueError(f"Unknown stage: {stage}")

        return return_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self._get_loss(batch, batch_idx, "train")
        # if "grad_loss" in loss_dict:
        #     total_loss = loss_dict["batch_loss"] + loss_dict["grad_loss"]
        # else:
        #     total_loss = loss_dict["batch_loss"]
        total_loss = loss_dict["batch_loss"]

        self.log_dict(loss_dict, on_step=True, on_epoch=False)

        self.log_dict(
            {"train_loss": loss_dict["batch_loss"]}, on_step=False, on_epoch=True
        )

        # if batch_idx % 1000 == 0:
        #     with torch.no_grad():
        #         has_extreme = False
        #         for (name, param) in self.named_parameters():
        #             if torch.isnan(param).any() or torch.isinf(param).any():
        #                 logger.warning(
        #                     f"NaN or Inf detected in parameter {name} at batch {batch_idx} during training."
        #                 )
        #                 has_extreme = True

        #             max_val = param.abs().max().item()
        #             if max_val > 1e5:
        #                 logger.warning(
        #                     f"Parameter {name} has a max value of {max_val} at batch {batch_idx} during training."
        #                 )
        #                 has_extreme = True

        #         if has_extreme:
        #             torch.save(
        #                 self.state_dict(),
        #                 f"extreme_parameters_batch_{batch_idx}.pt"
        #             )

        return {"loss": total_loss}
    
    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        loss_dict = self._get_loss(batch, batch_idx, "validation")
        loss = loss_dict["batch_loss"]

        if torch.isfinite(loss):
            self.log_dict(
                {"val_loss": loss_dict["batch_loss"]}, on_epoch=True, on_step=False
            )
            # self.validation_losses.append(loss.detach())
        else:
            logger.warning(f"Infinite loss detected at batch {batch_idx} during validation.")

        return {"loss": loss_dict["batch_loss"]}
    
    def test_step(self, batch, batch_idx):
        loss_dict = self._get_loss(batch, batch_idx, "test")

        self.log_dict(
            {"test_loss": loss_dict["batch_loss"]}, on_epoch=True, on_step=False
        )

        return {"loss": loss_dict["batch_loss"]}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=getattr(self.hparams, 'weight_decay', 0.0)
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-7, 
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
    
    def on_validation_epoch_end(self):
        # Log additional metrics for sweep optimization
        if hasattr(self.trainer, 'callback_metrics'):
            val_loss = self.trainer.callback_metrics.get('val_loss')
            if val_loss is not None:
                self.log("hp/val_loss", val_loss, prog_bar=True)
                
        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("hp/learning_rate", current_lr)

    def _num_parameters(self, requires_grad: bool = True):
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad == requires_grad
        )
