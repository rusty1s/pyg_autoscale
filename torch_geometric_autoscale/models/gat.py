from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv

from .base import HistoryGNN


class GAT(HistoryGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 hidden_heads: int, out_channels: int, out_heads: int,
                 num_layers: int, residual: bool = False, dropout: float = 0.0,
                 device=None, dtype=None):
        super(GAT, self).__init__(num_nodes, hidden_channels * hidden_heads,
                                  num_layers, device, dtype)

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
                n_id: Optional[Tensor] = None) -> Tensor:

        for conv, history in zip(self.convs[:-1], self.histories):
            h = F.dropout(x, p=self.dropout, training=self.training)
            h = conv(h, adj_t)
            if self.residual:
                x = F.dropout(x, p=self.dropout, training=self.training)
                h = h + x if h.size(-1) == x.size(-1) else h + self.lins[0](x)
            x = F.elu(h)
            x = self.push_and_pull(history, x, batch_size, n_id)

        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[-1](h, adj_t)
        if self.residual:
            x = F.dropout(x, p=self.dropout, training=self.training)
            h = h + self.lins[1](x)
        if batch_size is not None:
            h = h[:batch_size]
        return h

    @torch.no_grad()
    def mini_inference(self, x: Tensor, loader) -> Tensor:
        for conv, history in zip(self.convs[:-1], self.histories):
            for info in loader:
                info = info.to(self.device)
                batch_size, n_id, adj_t, e_id = info

                r = x[n_id]
                h = conv(r, adj_t)
                if self.residual:
                    if h.size(-1) == r.size(-1):
                        h = h + r
                    else:
                        h = h + self.lins[0](r)
                h = F.elu(h)
                history.push_(h[:batch_size], n_id[:batch_size])

            x = history.pull()

        out = x.new_empty(self.num_nodes, self.out_channels)
        for info in loader:
            info = info.to(self.device)
            batch_size, n_id, adj_t, e_id = info
            r = x[n_id]
            h = self.convs[-1](r, adj_t)[:batch_size]
            if self.residual:
                h = h + self.lins[1](r)
            out[n_id[:batch_size]] = h

        return out
