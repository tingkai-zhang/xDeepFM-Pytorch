from typing import List

import torch


class CompressedInteractionNetwork(torch.nn.Module):

    def __init__(self,
                 input_dim: int,
                 cross_layer_sizes: List[int],
                 split_half=True):
        """

        Parameters
        ----------
        input_dim: ``int``
            the number of fields, which is len(field_dims)
        cross_layer_sizes:

        split_half
        """
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for cross_layer_size in cross_layer_sizes:
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        """
        Parameters
        ----------
        x:
        Float tensor of size of ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        # x0 (batch_size, num_fields, 1, embed_dim)
        # h (batch_size, num_fields, embed_dim)
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1) # (batch_size, num_fields, num_fields, embed_dim)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = self.activation(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))
