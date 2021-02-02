from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Identity
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_sparse import SparseTensor
from torch_geometric.nn import GINConv
from torch_geometric.nn.inits import reset

from .base import HistoryGNN


class GIN(HistoryGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 out_channels: int, num_layers: int, residual: bool = False,
                 dropout: float = 0.0, device=None, dtype=None):
        super(GIN, self).__init__(num_nodes, hidden_channels, num_layers,
                                  device, dtype)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual
        self.dropout = dropout

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = ModuleList()
        for _ in range(num_layers):
            conv = GINConv(nn=Identity(), train_eps=True)
            self.convs.append(conv)

        self.post_nns = ModuleList()
        for i in range(num_layers):
            post_nn = Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm1d(hidden_channels, track_running_stats=False),
                ReLU(inplace=True),
                Linear(hidden_channels, hidden_channels),
                ReLU(inplace=True),
            )
            self.post_nns.append(post_nn)

    def reset_parameters(self):
        super(GIN, self).reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for post_nn in self.post_nns:
            reset(post_nn)
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor,
                batch_size: Optional[int] = None,
                n_id: Optional[Tensor] = None) -> Tensor:

        x = self.lins[0](x).relu()

        for conv, post_nn, history in zip(self.convs[:-1], self.post_nns[:-1],
                                          self.histories):
            if batch_size is not None:
                h = torch.zeros_like(x)
                h[:batch_size] = post_nn(conv(x, adj_t)[:batch_size])
            else:
                h = post_nn(conv(x, adj_t))

            x = h.add_(x) if self.residual else h
            x = self.push_and_pull(history, x, batch_size, n_id)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if batch_size is not None:
            h = self.post_nns[-1](self.convs[-1](x, adj_t)[:batch_size])
            x = x[:batch_size]
        else:
            h = self.post_nns[-1](self.convs[-1](x, adj_t))

        x = h.add_(x) if self.residual else h
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[1](x)
        return x
