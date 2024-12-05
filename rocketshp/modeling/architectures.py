import torch
from esm.layers.regression_head import RegressionHead
from esm.layers.transformer_stack import TransformerStack
from torch import nn

from esm.utils.constants import esm3 as ESM_CONSTANTS

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
    def __init__(self, embedding_dim: int, d_model: int, n_tokens: int = ESM_CONSTANTS.VQVAE_CODEBOOK_SIZE):
        super().__init__()

        self.seq_emb = SeqEncoder(embedding_dim, d_model)
        self.struct_emb = StructEncoder(d_model, n_tokens=n_tokens)

    def forward(self, x):
        """
        x = (seq_embeddings, struct_tokens)
        seq_embeddings \\in R^{batch_size \times N \times embedding_dim}
        struct_tokens \\in [1, n_tokens]^{batch_size \times N}
        """

        seq_embeddings, struct_tokens = x
        seq_embeddings = self.seq_emb(seq_embeddings)
        struct_tokens = self.struct_emb(struct_tokens)
        return seq_embeddings + struct_tokens


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
        x = (seq_embeddings, struct_tokens)
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