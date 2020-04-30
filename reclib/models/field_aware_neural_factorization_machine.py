import torch
from torch.nn import Linear

from reclib.modules import FeedForward
from reclib.modules import FieldAwareFactorizationLayer
from reclib.modules.embedders import LinearEmbedder


class FieldAwareNeuralFactorizationMachine(torch.nn.Module):
    """
    This class implements Field-aware Neural Factorization Machine.
    `"Field-aware Neural Factorization Machine for Click-Through Rate Prediction"
    <https://arxiv.org/abs/1902.09096>`_
    by L Zhang, et al.,  2019.

    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts):
        super().__init__()
        self.linear = LinearEmbedder(field_dims, 1)
        self.ffm = FieldAwareFactorizationLayer(field_dims, embed_dim)
        self.ffm_output_dim = len(field_dims) * (len(field_dims) - 1) // 2 * embed_dim
        self.bn = torch.nn.BatchNorm1d(self.ffm_output_dim)
        self.dp = torch.nn.Dropout(dropouts[0])
        self.mlp = torch.nn.Sequential(FeedForward(1,
                                                   self.ffm_output_dim,
                                                   mlp_dims,
                                                   batch_norm=True,
                                                   activations=torch.nn.ReLU(),
                                                   dropouts=dropouts[1]),
                                       Linear(mlp_dims[-1], 1))

    def forward(self, x: torch.LongTensor):
        """
        Parameters
        ----------
        x: torch.LongTensor
            Shape of ``(batch_size, num_fields)``

        Returns
        -------
        label_logits:
            A tensor of shape ``(batch_size, num_labels)`` representing un-normalised log
            probabilities of the label.
        """
        cross_term = self.ffm(x).view(-1, self.ffm_output_dim)
        cross_term = self.dp(self.bn(cross_term))
        x = self.linear(x) + self.mlp(cross_term)
        label_logits = torch.sigmoid(x.squeeze(1))
        return label_logits
