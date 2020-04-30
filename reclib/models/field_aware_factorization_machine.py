import torch

from reclib.modules import FieldAwareFactorizationLayer
from reclib.modules.embedders import LinearEmbedder


class FieldAwareFactorizationMachine(torch.nn.Module):
    """
    A pytorch implementation of Field-aware Factorization Machine.
    Parameters
    ----------
    
    Reference:
        Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.linear = LinearEmbedder(field_dims, 1)
        self.ffm = FieldAwareFactorizationLayer(field_dims, embed_dim)

    def forward(self, x):
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        return torch.sigmoid(x.squeeze(1))
