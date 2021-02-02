from typing import Optional, Dict, Any

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv

from scaling_gnns.models.base2 import ScalableGNN


class GCN(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 drop_input: bool = True, batch_norm: bool = False,
                 residual: bool = False, pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super(GCN, self).__init__(num_nodes, hidden_channels, num_layers,
                                  pool_size, buffer_size, device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.drop_input = drop_input
        self.batch_norm = batch_norm
        self.residual = residual

        self.convs = ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            conv = GCNConv(in_dim, out_dim, normalize=False)
            self.convs.append(conv)

        self.bns = ModuleList()
        for i in range(num_layers - 1):
            bn = BatchNorm1d(hidden_channels)
            self.bns.append(bn)

    @property
    def reg_modules(self):
        return ModuleList(list(self.convs[:-1]) + list(self.bns))

    @property
    def nonreg_modules(self):
        return self.convs[-1:]

    def reset_parameters(self):
        super(GCN, self).reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor,
                batch_size: Optional[int] = None,
                n_id: Optional[Tensor] = None, offset: Optional[Tensor] = None,
                count: Optional[Tensor] = None) -> Tensor:

        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, bn, hist in zip(self.convs[:-1], self.bns, self.histories):
            h = conv(x, adj_t)
            if self.batch_norm:
                h = bn(h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            x = h.relu_()
            x = self.push_and_pull(hist, x, batch_size, n_id, offset, count)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj_t)
        return x

    @torch.no_grad()
    def forward_layer(self, layer: int, x: Tensor, adj_t: SparseTensor,
                      state: Dict[Any]) -> Tensor:
        if layer == 0 and self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[layer](x, adj_t)

        if layer < self.num_layers - 1:
            if self.batch_norm:
                h = self.bns[layer](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            h = h.relu_()
        return h
