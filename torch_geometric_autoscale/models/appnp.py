from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_sparse import SparseTensor

from torch_geometric_autoscale.models import ScalableGNN


class APPNP(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 out_channels: int, num_layers: int, alpha: float,
                 dropout: float = 0.0, pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, out_channels, num_layers, pool_size,
                         buffer_size, device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.dropout = dropout

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.reg_modules = self.lins[:1]
        self.nonreg_modules = self.lins[1:]

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[0](x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[1](x)
        x_0 = x[:adj_t.size(0)]

        for history in self.histories:
            x = (1 - self.alpha) * (adj_t @ x) + self.alpha * x_0
            x = self.push_and_pull(history, x, *args)

        x = (1 - self.alpha) * (adj_t @ x) + self.alpha * x_0
        return x

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        if layer == 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[0](x)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x_0 = self.lins[1](x)
            state['x_0'] = x_0[:adj_t.size(0)]

        x = (1 - self.alpha) * (adj_t @ x) + self.alpha * state['x_0']
        return x
