from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_sparse import SparseTensor

from .base import HistoryGNN


class APPNP(HistoryGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 out_channels: int, num_layers: int, alpha: float,
                 dropout: float = 0.0, device=None, dtype=None):
        super(APPNP, self).__init__(num_nodes, out_channels, num_layers,
                                    device, dtype)

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
        super(APPNP, self).reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor,
                batch_size: Optional[int] = None,
                n_id: Optional[Tensor] = None) -> Tensor:

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[0](x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x_0 = self.lins[1](x)

        for history in self.histories:
            x = (1 - self.alpha) * (adj_t @ x) + self.alpha * x_0
            x = self.push_and_pull(history, x, batch_size, n_id)

        x = (1 - self.alpha) * (adj_t @ x) + self.alpha * x_0
        if batch_size is not None:
            x = x[:batch_size]
        return x

    @torch.no_grad()
    def mini_inference(self, x: Tensor, loader) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[0](x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x_0 = self.lins[1](x)

        for history in self.histories:
            for info in loader:
                info = info.to(self.device)
                batch_size, n_id, adj_t, e_id = info

                h = x[n_id]
                h_0 = x_0[n_id]
                h = (1 - self.alpha) * (adj_t @ h) + self.alpha * h_0
                history.push_(h[:batch_size], n_id[:batch_size])

            x = history.pull()

        out = x.new_empty(self.num_nodes, self.out_channels)
        for info in loader:
            info = info.to(self.device)
            batch_size, n_id, adj_t, e_id = info
            h = x[n_id]
            h_0 = x_0[n_id]
            h = (1 - self.alpha) * (adj_t @ h) + self.alpha * h_0
            out[n_id[:batch_size]] = h

        return out
