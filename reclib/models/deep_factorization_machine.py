import torch
from torch.nn import Linear

from reclib.modules import FactorizationMachine
from reclib.modules import FeedForward
from reclib.modules.embedders import LinearEmbedder, Embedding


class DeepFactorizationMachine(torch.nn.Module):
    """
    This implements implementation of DeepFM from
    `"DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
    <https://arxiv.org/abs/1703.04247>`_ by H Guo, et al. , 2017.
    Parameters
    ----------


    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = LinearEmbedder(field_dims, 1)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = Embedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = torch.nn.Sequential(FeedForward(2,
                                                   self.embed_output_dim,
                                                   mlp_dims,
                                                   True,
                                                   torch.nn.ReLU(),
                                                   dropout),
                                       Linear(mlp_dims[-1], 1))

    def forward(self, x):
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
        embed_x = self.embedding(x)
        x = self.linear(x) + \
            self.fm(embed_x) + \
            self.mlp(embed_x.view(-1, self.embed_output_dim))
        label_logits = torch.sigmoid(x.squeeze(1))
        return label_logits
