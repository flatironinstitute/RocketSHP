import torch
from esm.layers.transformer_stack import TransformerStack
from esm.utils.constants import esm3 as ESM_CONSTANTS
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from pathlib import Path
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
        logger.debug(f"StructEncoder input shape: {x.shape}")
        logger.debug(f"Token max: {x.max()}")
        logger.debug(f"n tokens: {self.n_tokens}")
        logger.debug(f"embedding dim: {self.embedding.weight.shape}")
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
    def __init__(
        self,
        embedding_dim: int,
        d_model: int,
        n_tokens: int = ESM_CONSTANTS.VQVAE_CODEBOOK_SIZE,
        seq_only: bool = False,
        struct_stage: str = "quantized",
        struct_dim: int = None,
    ):
        super().__init__()

        self.seq_emb = SeqEncoder(embedding_dim, d_model)

        if struct_stage == "quantized":
            self.struct_emb = StructEncoder(d_model, n_tokens=n_tokens)
        else:
            self.struct_emb = SeqEncoder(struct_dim, d_model)

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
            logger.debug(f"seq_embeddings: {seq_embeddings.shape}")
            logger.debug(f"struct_tokens: {struct_tokens.shape}")
            return seq_embeddings + struct_tokens


class GradNorm(nn.Module):
    def __init__(self, model, num_tasks, alpha=0.12, grad_lr=1e-5):
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
        loss_ratios = torch.stack(
            [loss / init_loss for loss, init_loss in zip(losses, self.initial_losses)]
        )

        # Compute gradient norms while maintaining graph connection
        grad_norms = []
        for i, (weight, loss) in enumerate(zip(self.task_weights, losses)):
            weighted_loss = weight * loss
            with torch.no_grad():
                grads = torch.autograd.grad(
                    weighted_loss, self.shared_parameters, retain_graph=True
                )
                grad_norm = torch.sqrt(sum((g**2).sum() for g in grads))

            grad_norms.append(
                grad_norm * weight / weight.detach()
            )  # Detach gradient norms
        grad_norms = torch.stack(grad_norms)
        mean_norm = torch.mean(grad_norms)

        # Compute relative inverse training rate r_i
        rel_inv_rates = loss_ratios / torch.sum(loss_ratios)

        # Calculate target gradient norms
        target_norms = mean_norm * (rel_inv_rates**self.alpha)

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
            self.task_weights.data = (
                torch.nn.functional.softmax(self.task_weights, dim=0) * self.num_tasks
            )
            # normalize_coeff = self.num_tasks / self.task_weights.sum()
            # self.task_weights.data *= normalize_coeff

        # Compute weighted loss without detaching
        weighted_losses = torch.stack(
            [w * l for w, l in zip(self.task_weights, losses)]
        )
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


def CategoricalHead(
    d_model: int,
    output_dim: int,
    hidden_dim: int | None = None,
) -> nn.Module:
    """Single-hidden layer MLP for supervised output of categorical variable.

    Args:
        d_model: input dimension
        output_dim: dimensionality of the output.
        hidden_dim: optional dimension of hidden layer, defaults to d_model.
    Returns:
        output MLP module.
    """
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
        # nn.Softmax(dim=-1),
    )


def RegressionHead(
    d_model: int,
    output_dim: int,
    hidden_dim: int | None = None,
) -> nn.Module:
    """Single-hidden layer MLP for supervised output.

    Args:
        d_model: input dimension
        output_dim: dimensionality of the output.
        hidden_dim: optional dimension of hidden layer, defaults to d_model.
    Returns:
        output MLP module.
    """
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )

def PairwiseDistanceHead(
    d_model: int, kernel_size: int, hidden_dim: int | None = None, output_dim: int = 1
) -> nn.Module:
    """Single hidden layer MLP for pairwise output.

    Args:
        d_model: input dimension
        kernel_size: kernel size for convolutional layers
        hidden_dim: optional dimension of hidden layer, defaults to d_model.
        output_dim: dimensionality of the output.
    Returns:
        output MLP module.
    """
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Conv2d(d_model, hidden_dim, kernel_size=kernel_size, stride=1),
        nn.GELU(),
        nn.Conv2d(hidden_dim, output_dim, kernel_size=kernel_size, stride=1),
        nn.ReLU(),
    )

def PairwiseProbabilityHead(
    d_model: int, kernel_size: int, hidden_dim: int | None = None, output_dim: int = 1
) -> nn.Module:
    """Single hidden layer MLP for pairwise output.

    Args:
        d_model: input dimension
        kernel_size: kernel size for convolutional layers
        hidden_dim: optional dimension of hidden layer, defaults to d_model.
        output_dim: dimensionality of the output.
    Returns:
        output MLP module.
    """
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Conv2d(d_model, hidden_dim, kernel_size=kernel_size, stride=1),
        nn.GELU(),
        nn.Conv2d(hidden_dim, output_dim, kernel_size=kernel_size, stride=1),
        nn.Sigmoid(),
    )

class RocketSHPModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        k_size: int = 1,
        seq_only: bool = False,
        struct_stage: str = "quantized",
        struct_dim: int = None,
        n_input_tokens: int = ESM_CONSTANTS.VQVAE_CODEBOOK_SIZE,
        n_shp_tokens: int = 20,
        default_temp: float = 300.0
    ):
        super().__init__()

        self.default_temp = nn.Parameter(
            torch.tensor(default_temp), requires_grad=False
        )

        self.encoder = JointStructAndSequenceEncoder(
            embedding_dim,
            d_model,
            n_tokens=n_input_tokens,
            seq_only=seq_only,
            struct_stage=struct_stage,
            struct_dim=struct_dim,
        )
        self.transformer = TransformerStack(
            d_model, n_heads, n_layers=n_layers, v_heads=0, n_layers_geom=0
        )

        self.rmsf_head = RegressionHead(d_model + 1, output_dim)
        self.ca_dist_head = PairwiseDistanceHead(d_model, k_size)
        self.dyn_corr_head = PairwiseProbabilityHead(d_model, k_size)
        self.autocorr_head = PairwiseProbabilityHead(d_model, k_size)
        self.gcc_lmi_head = PairwiseProbabilityHead(d_model, k_size)
        self.shp_head = CategoricalHead(d_model, n_shp_tokens)

        # for name, module in self.OUTPUT_HEADS.items():
        #     setattr(self, f"{name}_head", module)

        self.grad_norm = GradNorm(self, num_tasks=3, alpha=1.5)
        # self.grad_norm.task_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def _transform(self, x):
        x = self.encoder(x)
        tout = self.transformer(x)
        x = tout[0]
        return x

    def _cross_transform(self, x):
        x = self._transform(x)
        return x @ x.transpose(1, 2)

    def rmsf(self, x):
        x = self._transform(x)
        return self.rmsf_head(x)

    def ca_dist(self, x):
        x = self._transform(x)
        sqform = (x.unsqueeze(1) * x.unsqueeze(2)).transpose(1, 3)
        return self.ca_dist_head(sqform).squeeze(1)

    def dyn_corr(self, x):
        x = self._transform(x)
        sqform = (x.unsqueeze(1) * x.unsqueeze(2)).transpose(1, 3)
        return self.dyn_corr_head(sqform).squeeze

    def autocorr(self, x):
        x = self._transform(x)
        sqform = (x.unsqueeze(1) * x.unsqueeze(2)).transpose(1, 3)
        return self.autocorr_head(sqform).squeeze(1)

    def shp(self, x):
        x = self._transform(x)
        return self.shp_head(x).squeeze(1)

    def forward(self, x):
        """
        x = {"seq_feats", "struct_feats", "temp"}
        seq_embeddings \\in R^{batch_size \times N \times embedding_dim}
        struct_tokens \\in [1, n_tokens]^{batch_size \times N}
        """
        if "temp" in x:
            temperature = x["temp"]
        else:
            temperature = torch.ones(x["seq_feats"].shape[1], device=x["seq_feats"].device).unsqueeze(0) * self.default_temp

        x = self._transform(x)
        feats_with_temp = torch.cat([x, temperature.unsqueeze(-1)], dim=-1)
        rmsf_pred = self.rmsf_head(feats_with_temp)
        # shp_pred = self.shp_head(feats_with_temp).squeeze(1)
        shp_pred = self.shp_head(x).squeeze(1)

        sqform = (x.unsqueeze(1) * x.unsqueeze(2)).transpose(1, 3)
        ca_dist_pred = self.ca_dist_head(sqform).squeeze(1)
        # autocorr_pred = self.autocorr_head(sqform).squeeze(1)
        gcc_pred = self.gcc_lmi_head(sqform).squeeze(1)

        return {
            "rmsf": rmsf_pred,
            "ca_dist": ca_dist_pred,
            # "autocorr": autocorr_pred,
            "gcc_lmi": gcc_pred,
            "shp": shp_pred,
        }

    def _num_parameters(self, requires_grad: bool = True):
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad == requires_grad
        )

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str = "latest", strict: bool = True):
        """Load a model from a checkpoint."
        """
        from rocketshp.config import PRETRAINED_MODELS
        from huggingface_hub import hf_hub_download

        if checkpoint_path in PRETRAINED_MODELS:
            checkpoint_path = hf_hub_download(
                repo_id="samsl/rocketshp",
                filename=PRETRAINED_MODELS[checkpoint_path],
                subfolder="checkpoints",
            )
        elif not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Checkpoint path {checkpoint_path} does not exist."
            )
        elif not checkpoint_path.endswith(".ckpt"):
            raise ValueError(
                f"Checkpoint path {checkpoint_path} must end with .ckpt."
            )
        else:
            logger.error(f"If you are trying to download from HuggingFace, available models are: {PRETRAINED_MODELS.keys()}")
            raise FileNotFoundError(
                f"Checkpoint path {checkpoint_path} does not exist."
            )

        chk = torch.load(checkpoint_path, weights_only=True)
        hp = chk["hyper_parameters"]
        state_dict = {}
        if "struct_stage" not in hp:
            hp["struct_stage"] = "quantized"
        if "struct_dim" not in hp:
            hp["struct_dim"] = None
        fm = cls(
            hp["embedding_dim"],
            hp["output_dim"],
            hp["d_model"],
            hp["n_heads"],
            hp["n_layers"],
            seq_only=not hp["struct_features"],
            struct_stage=hp["struct_stage"],
            struct_dim=hp["struct_dim"],
        )
        for k, v in chk["state_dict"].items():
            new_k = k.replace("child_model.", "")
            state_dict[new_k] = v
        if "grad_norm.task_weights" not in chk["state_dict"]:
            state_dict["grad_norm.task_weights"] = torch.tensor([1.0, 1.0, 1.0])
        fm.load_state_dict(state_dict, strict=strict)
        fm.eval()
        return fm

    def save_to_safetensors(self, checkpoint_path: str):
        state_dict = self.state_dict()
        save_file(state_dict, checkpoint_path)
        return checkpoint_path

    def load_from_safetensors(self, checkpoint_path: str):
        state_dict = {}
        with safe_open(checkpoint_path, framework="pt", device=0) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
