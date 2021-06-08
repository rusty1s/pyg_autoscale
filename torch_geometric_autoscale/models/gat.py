from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv

from torch_geometric_autoscale.models import ScalableGNN


class GAT(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 hidden_heads: int, out_channels: int, out_heads: int,
                 num_layers: int, dropout: float = 0.0,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, hidden_channels * hidden_heads, num_layers,
                         pool_size, buffer_size, device)

        self.in_channels = in_channels
        self.hidden_heads = hidden_heads
        self.out_channels = out_channels
        self.out_heads = out_heads
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

        self.reg_modules = self.convs
        self.nonreg_modules = ModuleList()

    def reset_parameters(self):
        super().reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        for conv, history in zip(self.convs[:-1], self.histories):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv((x, x[:adj_t.size(0)]), adj_t)
            x = F.elu(x)
            x = self.push_and_pull(history, x, *args)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1]((x, x[:adj_t.size(0)]), adj_t)
        return x

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[layer]((x, x[:adj_t.size(0)]), adj_t)

        if layer < self.num_layers - 1:
            x = x.elu()

        return x
