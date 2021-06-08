from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn import GCN2Conv

from torch_geometric_autoscale.models import ScalableGNN


class GCN2(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 out_channels: int, num_layers: int, alpha: float,
                 theta: float, shared_weights: bool = True,
                 dropout: float = 0.0, drop_input: bool = True,
                 batch_norm: bool = False, residual: bool = False,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size,
                         buffer_size, device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.drop_input = drop_input
        self.batch_norm = batch_norm
        self.residual = residual

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = ModuleList()
        for i in range(num_layers):
            conv = GCN2Conv(hidden_channels, alpha=alpha, theta=theta,
                            layer=i + 1, shared_weights=shared_weights,
                            normalize=False)
            self.convs.append(conv)

        self.bns = ModuleList()
        for i in range(num_layers):
            bn = BatchNorm1d(hidden_channels)
            self.bns.append(bn)

    @property
    def reg_modules(self):
        return ModuleList(list(self.convs) + list(self.bns))

    @property
    def nonreg_modules(self):
        return self.lins

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x_0 = self.lins[0](x).relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, bn, hist in zip(self.convs[:-1], self.bns[:-1],
                                  self.histories):
            h = conv(x, x_0, adj_t)
            if self.batch_norm:
                h = bn(h)
            if self.residual:
                h += x[:h.size(0)]
            x = h.relu_()
            x = self.push_and_pull(hist, x, *args)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[-1](x, x_0, adj_t)
        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual:
            h += x[:h.size(0)]
        x = h.relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[1](x)
        return x

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        if layer == 0:
            if self.drop_input:
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = x_0 = self.lins[0](x).relu_()
            state['x_0'] = x_0[:adj_t.size(0)]

        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[layer](x, state['x_0'], adj_t)
        if self.batch_norm:
            h = self.bns[layer](h)
        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        x = h.relu_()

        if layer == self.num_layers - 1:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[1](x)

        return x
