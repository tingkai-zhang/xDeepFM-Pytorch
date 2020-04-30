import torch
from torch import nn

from reclib.modules import FeedForward


class AttentionalFactorizationLayer(torch.nn.Module):

    def __init__(self, embed_dim, attn_size, dropouts):
        super().__init__()
        activations = [nn.ReLU(), nn.Softmax(dim=1)]
        self.attention_layer = FeedForward(2,
                                           embed_dim,
                                           [attn_size, 1],
                                           False,
                                           activations,
                                           [0, dropouts[0]])

        self.linear_layer = torch.nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(p=dropouts[1])

    def forward(self, x):
        """
        Parameters
        ----------
        x
        Size of ``(batch_size, num_fields, embed_dim)``

        Returns
        -------

        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        inner_product = p * q
        attention_scores = self.attention_layer(inner_product)
        attention_output = torch.sum(attention_scores * inner_product, dim=1)
        attention_output = self.dropout(attention_output)
        return self.linear_layer(attention_output)
