import torch
from esm.layers.regression_head import RegressionHead
from esm.layers.transformer_stack import TransformerStack
from torch import nn

from esm.utils.constants import esm3 as ESM_CONSTANTS
from loguru import logger

class StructEncoder(nn.Module):
    def __init__(self, d_model: int, n_tokens: int = ESM_CONSTANTS.VQVAE_CODEBOOK_SIZE):
        super().__init__()

        self.n_tokens = n_tokens
        self.embedding = nn.Embedding(n_tokens, d_model)

    def forward(self, x):
        """
        x \\in [1, n_tokens]^{batch_size \times N})
        """
        return self.embedding(x)


class SeqEncoder(nn.Module):
    def __init__(self, embedding_dim: int, d_model: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        """
        x \\in R^{batch_size \times N \times embedding_dim})
        """
        return self.linear(x)


class JointStructAndSequenceEncoder(nn.Module):
    def __init__(self, embedding_dim: int, d_model: int, n_tokens: int = ESM_CONSTANTS.VQVAE_CODEBOOK_SIZE, seq_only: bool = False):
        super().__init__()

        self.seq_emb = SeqEncoder(embedding_dim, d_model)
        self.struct_emb = StructEncoder(d_model, n_tokens=n_tokens)
        self.seq_only = seq_only

    def forward(self, x):
        """
        x = (seq_embeddings, struct_tokens)
        seq_embeddings \\in R^{batch_size \times N \times embedding_dim}
        struct_tokens \\in [1, n_tokens]^{batch_size \times N}
        """
        
        if self.seq_only:
            seq_embeddings = self.seq_emb(x["seq_feats"])
            return seq_embeddings
        else:
            seq_embeddings, struct_tokens = x["seq_feats"], x["struct_feats"]
            seq_embeddings = self.seq_emb(seq_embeddings)
            struct_tokens = self.struct_emb(struct_tokens)
            return seq_embeddings + struct_tokens

class GradNorm(nn.Module):
    def __init__(self, model, num_tasks, alpha=0.12, grad_lr = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.grad_lr = grad_lr
        self.num_tasks = num_tasks
        
        # Task weights - initialized to 1
        self.task_weights = nn.Parameter(torch.ones(num_tasks))
        
        # Get shared parameters that gradient norm will be computed over
        # Usually the last shared layer
        self.shared_parameters = []
        for param in self._get_shared_parameters(model):
            if param.requires_grad:
                self.shared_parameters.append(param)
        
        # Initialize tracking variables
        self.initial_losses = None
        self.initial_L = None
    
    def _get_shared_parameters(self, model):
        return model.transformer.blocks[-1].parameters()

    def forward(self, losses):
        if not isinstance(losses, list):
            losses = list(losses)
            for l in losses:
                l.requires_grad = True
        
        # Initialize initial losses and L
        if self.initial_losses is None:
            self.initial_losses = torch.tensor([l.item() for l in losses])
            self.initial_L = torch.sum(self.initial_losses)
        
        # Compute unweighted loss ratios
        loss_ratios = torch.stack([loss / init_loss 
                                for loss, init_loss in zip(losses, self.initial_losses)])

        # Compute gradient norms while maintaining graph connection
        grad_norms = []
        for i, (weight, loss) in enumerate(zip(self.task_weights, losses)):
            weighted_loss = weight * loss
            with torch.no_grad():
                grads = torch.autograd.grad(
                    weighted_loss, 
                    self.shared_parameters,
                    retain_graph=True
                )
                grad_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
            
            grad_norms.append(grad_norm * weight / weight.detach())  # Detach gradient norms
        grad_norms = torch.stack(grad_norms)
        mean_norm = torch.mean(grad_norms)

        # Compute relative inverse training rate r_i
        rel_inv_rates = loss_ratios / torch.sum(loss_ratios)
        
        # Calculate target gradient norms
        target_norms = mean_norm * (rel_inv_rates ** self.alpha)
        
        # Compute gradient norm loss
        grad_norm_loss = torch.nn.functional.l1_loss(grad_norms, target_norms)
        
        # Update task weights
        task_weight_grad = torch.autograd.grad(
            grad_norm_loss, self.task_weights, retain_graph=True
        )
        # grad_norm_loss.backward(retain_graph=True)
    
        with torch.no_grad():
            # logger.debug(f"Task weights grad: {task_weight_grad[0]}")
            # logger.debug(f"Task weights: {self.task_weights.detach().cpu().numpy()}")

            self.task_weights.data -= self.grad_lr * task_weight_grad[0]
            # logger.debug(f"Task weights updated: {self.task_weights.detach().cpu().numpy()}")
            # self.task_weights.grad.zero_()
            
            # Normalize weights to sum to num_tasks
            self.task_weights.data = torch.nn.functional.softmax(self.task_weights, dim=0) * self.num_tasks
            # normalize_coeff = self.num_tasks / self.task_weights.sum()
            # self.task_weights.data *= normalize_coeff

        # Compute weighted loss without detaching
        weighted_losses = torch.stack([w * l for w, l in zip(self.task_weights, losses)])
        total_loss = torch.sum(weighted_losses)

        # logger.debug("----")
                
        # logger.debug(f"Losses: {torch.stack(losses)}")
        # logger.debug(f"Grad norms: {grad_norms}")
        # logger.debug(f"Loss ratios: {loss_ratios}")
        # logger.debug(f"Rel inv rates: {rel_inv_rates}")
        # logger.debug(f"Mean grad norm: {mean_norm:.3f}")
        # logger.debug(f"Target norms: {target_norms}")
        
        # logger.debug(f"Weighted losses: {weighted_losses}")
        # logger.debug("#####")

        return total_loss, weighted_losses, grad_norm_loss.detach()

class FlexibilityModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        k_size: int = 1,
    ):
        super().__init__()

        self.encoder = JointStructAndSequenceEncoder(embedding_dim, d_model)
        self.transformer = TransformerStack(
            d_model, n_heads, n_layers=n_layers, v_heads=0, n_layers_geom=0
        )
        self.squareformer = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=k_size, stride=1),
            nn.GELU(),
            nn.Conv2d(d_model, 1, kernel_size=k_size, stride=1),
            nn.GELU(),
        )
        self.output = RegressionHead(d_model, output_dim)

    def _transform(self, x):
        x = self.encoder(x)
        x, _ = self.transformer(x)
        return x
    
    def _cross_transform(self, x):
        x = self._transform(x)
        return x @ x.transpose(1,2)
    
    def squareform(self, x):
        x = self._transform(x)
        sqform = (x.unsqueeze(1) * x.unsqueeze(2)).transpose(1,3)
        return self.squareformer(sqform).squeeze(1)
    
    def rmsf(self, x):
        x = self._transform(x)
        return self.output(x)

    def forward(self, x):
        """
        x = {"seq_feats": v, "struct_feats": v, "temp": v}
        seq_embeddings \\in R^{batch_size \times N \times embedding_dim}
        struct_tokens \\in [1, n_tokens]^{batch_size \times N}
        """
        x = self._transform(x)
        rmsf_pred = self.output(x)

        sqform = (x.unsqueeze(1) * x.unsqueeze(2)).transpose(1,3)
        sqform_pred = self.squareformer(sqform).squeeze(1)
        return {
            "rmsf": rmsf_pred,
            "ca_dist": sqform_pred,
        }
    
    def _num_parameters(self, requires_grad: bool = True):
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad == requires_grad
        )
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, strict: bool = True):
        chk = torch.load(checkpoint_path)
        hp = chk["hyper_parameters"]
        state_dict = {}
        fm = cls(hp["embedding_dim"], hp["output_dim"], hp["d_model"], hp["n_heads"], hp["n_layers"])
        for k,v in chk["state_dict"].items():
            new_k = k.replace("child_model.","")
            state_dict[new_k] = v
        fm.load_state_dict(state_dict, strict=strict)
        fm.eval()
        return fm
    
class FlexibilityModelWithTemperature(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        k_size: int = 1,
        seq_only: bool = False,
    ):
        super().__init__()

        self.encoder = JointStructAndSequenceEncoder(embedding_dim, d_model, seq_only=seq_only)
        self.transformer = TransformerStack(
            d_model, n_heads, n_layers=n_layers, v_heads=0, n_layers_geom=0
        )
        self.squareformer = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=k_size, stride=1),
            nn.GELU(),
            nn.Conv2d(d_model, 1, kernel_size=k_size, stride=1),
            nn.GELU(),
        )
        self.output = RegressionHead(d_model + 1, output_dim)

    def _transform(self, x):
        x = self.encoder(x)
        tout = self.transformer(x)
        x = tout[0]
        return x
    
    def _cross_transform(self, x):
        x = self._transform(x)
        return x @ x.transpose(1,2)
    
    def squareform(self, x):
        x = self._transform(x)
        sqform = (x.unsqueeze(1) * x.unsqueeze(2)).transpose(1,3)
        return self.squareformer(sqform).squeeze(1)
    
    def rmsf(self, x):
        x = self._transform(x)
        return self.output(x)

    def forward(self, x):
        """
        x = (seq_embeddings, struct_tokens)
        seq_embeddings \\in R^{batch_size \times N \times embedding_dim}
        struct_tokens \\in [1, n_tokens]^{batch_size \times N}
        """
        temperature = x["temp"]
        x = self._transform(x)
        rmsf_pred = self.output(torch.cat([x, temperature.unsqueeze(-1)], dim=-1))

        sqform = (x.unsqueeze(1) * x.unsqueeze(2)).transpose(1,3)
        sqform_pred = self.squareformer(sqform).squeeze(1)
        return {
            "rmsf": rmsf_pred,
            "ca_dist": sqform_pred,
            # "dyn_corr": sqform_pred,
        }
    
    def _num_parameters(self, requires_grad: bool = True):
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad == requires_grad
        )
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, strict: bool = True):
        chk = torch.load(checkpoint_path)
        hp = chk["hyper_parameters"]
        state_dict = {}
        fm = cls(hp["embedding_dim"], hp["output_dim"], hp["d_model"], hp["n_heads"], hp["n_layers"])
        for k,v in chk["state_dict"].items():
            new_k = k.replace("child_model.","")
            state_dict[new_k] = v
        fm.load_state_dict(state_dict, strict=strict)
        fm.eval()
        return fm
    
class DynCorrModelWithTemperature(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        k_size: int = 1,
        seq_only: bool = False,
    ):
        super().__init__()

        self.encoder = JointStructAndSequenceEncoder(embedding_dim, d_model, seq_only=seq_only)
        self.transformer = TransformerStack(
            d_model, n_heads, n_layers=n_layers, v_heads=0, n_layers_geom=0
        )
        self.rmsf_head = RegressionHead(d_model + 1, output_dim)
        self.ca_dist_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=k_size, stride=1),
            nn.GELU(),
            nn.Conv2d(d_model, 1, kernel_size=k_size, stride=1),
            nn.GELU(),
        )
        self.dyn_corr_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=k_size, stride=1),
            nn.GELU(),
            nn.Conv2d(d_model, 1, kernel_size=k_size, stride=1),
            nn.Sigmoid(),
        )

        self.grad_norm = GradNorm(self, num_tasks=3, alpha=1.5)
        # self.grad_norm.task_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def _transform(self, x):
        x = self.encoder(x)
        tout = self.transformer(x)
        x = tout[0]
        return x
    
    def _cross_transform(self, x):
        x = self._transform(x)
        return x @ x.transpose(1,2)
    
    def rmsf(self, x):
        x = self._transform(x)
        return self.output(x)

    def ca_dist(self, x):
        x = self._transform(x)
        sqform = (x.unsqueeze(1) * x.unsqueeze(2)).transpose(1,3)
        return self.ca_dist_head(sqform).squeeze(1)
    
    def dyn_corr(self, x):
        x = self._transform(x)
        sqform = (x.unsqueeze(1) * x.unsqueeze(2)).transpose(1,3)
        return self.dyn_corr_head(sqform).squeeze

    def forward(self, x):
        """
        x = {"seq_feats", "struct_feats", "temp"}
        seq_embeddings \\in R^{batch_size \times N \times embedding_dim}
        struct_tokens \\in [1, n_tokens]^{batch_size \times N}
        """
        if "temp" in x:
            temperature = x["temp"]

        x = self._transform(x)
        rmsf_pred = self.rmsf_head(torch.cat([x, temperature.unsqueeze(-1)], dim=-1))
        sqform = (x.unsqueeze(1) * x.unsqueeze(2)).transpose(1,3)
        
        ca_dist_pred = self.ca_dist_head(sqform).squeeze(1)
        dyn_corr_pred = self.dyn_corr_head(sqform).squeeze(1)

        return {
            "rmsf": rmsf_pred,
            "ca_dist": ca_dist_pred,
            "dyn_corr": dyn_corr_pred,
        }
    
    def _num_parameters(self, requires_grad: bool = True):
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad == requires_grad
        )
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, strict: bool = True):
        chk = torch.load(checkpoint_path)
        hp = chk["hyper_parameters"]
        state_dict = {}
        fm = cls(hp["embedding_dim"], hp["output_dim"], hp["d_model"], hp["n_heads"], hp["n_layers"], seq_only=not hp["struct_features"])
        for k,v in chk["state_dict"].items():
            new_k = k.replace("child_model.","")
            state_dict[new_k] = v
        if not "grad_norm.task_weights" in chk["state_dict"]:
            state_dict["grad_norm.task_weights"] = torch.tensor([1.0, 1.0, 1.0])
        fm.load_state_dict(state_dict, strict=strict)
        fm.eval()
        return fm