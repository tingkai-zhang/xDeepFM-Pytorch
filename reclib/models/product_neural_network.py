import torch

from reclib.modules import FeedForward
from reclib.modules import InnerProductNetwork, OuterProductNetwork
from reclib.modules.embedders import LinearEmbedder, Embedding


class ProductNeuralNetwork(torch.nn.Module):
    """
    This implements the inner/outer Product Neural Network from
    `"Product-based Neural Networks for User Response Prediction"
    <https://arxiv.org/abs/1611.00144>`_ by Y Qu, et al., 2016.
    """

    def __init__(self,
                 field_dims,
                 embed_dim,
                 mlp_dims,
                 dropout,
                 method='inner'):
        super().__init__()
        num_fields = len(field_dims)
        if method == 'inner':
            self.pn = InnerProductNetwork()
        elif method == 'outer':
            self.pn = OuterProductNetwork(num_fields, embed_dim)
        else:
            raise ValueError('unknown product type: ' + method)
        self.embedding = Embedding(field_dims, embed_dim)
        self.linear = LinearEmbedder(field_dims, embed_dim)
        self.embed_output_dim = num_fields * embed_dim

        self.mlp = FeedForward(2,
                               num_fields * (num_fields - 1) // 2 +
                               self.embed_output_dim,
                               [mlp_dims, 1],
                               True,
                               ['relu', 'linear'],
                               [dropout, 0])

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
        cross_term = self.pn(embed_x)
        x = torch.cat([embed_x.view(-1, self.embed_output_dim), cross_term],
                      dim=1)
        x = self.mlp(x)
        label_logits = torch.sigmoid(x.squeeze(1))
        return label_logits
