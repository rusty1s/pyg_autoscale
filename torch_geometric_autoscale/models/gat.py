from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv

from torch_geometric_autoscale.models import ScalableGNN


class GAT(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 hidden_heads: int, out_channels: int, out_heads: int,
                 num_layers: int, residual: bool = False, dropout: float = 0.0,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super(GAT, self).__init__(num_nodes, hidden_channels * hidden_heads,
                                  num_layers, pool_size, buffer_size, device)

        self.in_channels = in_channels
        self.hidden_heads = hidden_heads
        self.out_channels = out_channels
        self.out_heads = out_heads
        self.residual = residual
        self.dropout = dropout

        self.convs = ModuleList()
        for i in range(num_layers - 1):
            in_dim = in_channels if i == 0 else hidden_channels * hidden_heads
            conv = GATConv(in_dim, hidden_channels, hidden_heads, concat=True,
                           dropout=dropout, add_self_loops=False)
            self.convs.append(conv)

        conv = GATConv(hidden_channels * hidden_heads, out_channels, out_heads,
                       concat=False, dropout=dropout, add_self_loops=False)
        self.convs.append(conv)

        self.lins = ModuleList()
        if residual:
            self.lins.append(
                Linear(in_channels, hidden_channels * hidden_heads))
            self.lins.append(
                Linear(hidden_channels * hidden_heads, out_channels))

        self.reg_modules = ModuleList([self.convs, self.lins])
        self.nonreg_modules = ModuleList()

    def reset_parameters(self):
        super(GAT, self).reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor,
                batch_size: Optional[int] = None,
                n_id: Optional[Tensor] = None, offset: Optional[Tensor] = None,
                count: Optional[Tensor] = None) -> Tensor:

        for conv, history in zip(self.convs[:-1], self.histories):
            h = F.dropout(x, p=self.dropout, training=self.training)
            h = conv((h, h[:adj_t.size(0)]), adj_t)
            if self.residual:
                x = F.dropout(x, p=self.dropout, training=self.training)
                h += x if h.size(-1) == x.size(-1) else self.lins[0](x)
            x = F.elu(h)
            x = self.push_and_pull(history, x, batch_size, n_id, offset, count)

        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[-1]((h, h[:adj_t.size(0)]), adj_t)
        if self.residual:
            x = F.dropout(x, p=self.dropout, training=self.training)
            h += self.lins[1](x)
        return h

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[layer]((h, h[:adj_t.size(0)]), adj_t)

        if layer == 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[0](x)

        if layer == self.num_layers - 1:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[1](x)

        if self.residual:
            x = F.dropout(x, p=self.dropout, training=self.training)
            h += x

        if layer < self.num_layers - 1:
            h = h.elu()

        return h
