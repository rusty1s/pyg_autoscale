from typing import Optional, Callable

import warnings

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from scaling_gnns.history2 import History
from scaling_gnns.pool import AsyncIOPool


class ScalableGNN(torch.nn.Module):
    def __init__(self, num_nodes: int, hidden_channels: int, num_layers: int,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super(ScalableGNN, self).__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.pool_size = num_layers if pool_size is None else pool_size
        self.buffer_size = buffer_size

        self.histories = torch.nn.ModuleList([
            History(num_nodes, hidden_channels, device)
            for _ in range(num_layers - 1)
        ])

        self.pool = None
        self._async = False
        self.__out__ = None

    @property
    def emb_device(self):
        return self.histories[0].emb.device

    @property
    def device(self):
        return self.histories[0]._device

    @property
    def _out(self):
        if self.__out__ is None:
            self.__out__ = torch.empty(self.num_nodes, self.out_channels,
                                       pin_memory=True)
        return self.__out__

    def _apply(self, fn: Callable) -> None:
        super(ScalableGNN, self)._apply(fn)
        if (str(self.emb_device) == 'cpu' and str(self.device)[:4] == 'cuda'
                and self.pool_size is not None
                and self.buffer_size is not None):
            self.pool = AsyncIOPool(self.pool_size, self.buffer_size,
                                    self.histories[0].embedding_dim)
            self.pool.to(self.device)
        return self

    def reset_parameters(self):
        for history in self.histories:
            history.reset_parameters()

    def __call__(self, x: Optional[Tensor] = None,
                 adj_t: Optional[SparseTensor] = None,
                 batch_size: Optional[int] = None,
                 n_id: Optional[Tensor] = None,
                 offset: Optional[Tensor] = None,
                 count: Optional[Tensor] = None, loader=None,
                 **kwargs) -> Tensor:

        if loader is not None:
            return self.mini_inference(loader)

        self._async = (self.pool is not None and batch_size is not None
                       and n_id is not None and offset is not None
                       and count is not None)

        if batch_size is not None and not self._async:
            warnings.warn('Asynchronous I/O disabled, although history and '
                          'model sit on different devices.')

        if self._async:
            for hist in self.histories:
                self.pool.async_pull(hist.emb, None, None, n_id[batch_size:])

        out = self.forward(x=x, adj_t=adj_t, batch_size=batch_size, n_id=n_id,
                           offset=offset, count=count, **kwargs)

        if self._async:
            for hist in self.histories:
                self.pool.synchronize_push()

        self._async = False

        return out

    def push_and_pull(self, history, x: Tensor,
                      batch_size: Optional[int] = None,
                      n_id: Optional[Tensor] = None,
                      offset: Optional[Tensor] = None,
                      count: Optional[Tensor] = None) -> Tensor:

        if n_id is None and x.size(0) != self.num_nodes:
            return x  # Do nothing...

        if n_id is None and x.size(0) == self.num_nodes:
            history.push(x)
            return x

        assert n_id is not None

        if batch_size is None:
            history.push(x, n_id)
            return x

        if not self._async:
            history.push(x[:batch_size], n_id[:batch_size], offset, count)
            h = history.pull(n_id[batch_size:])
            return torch.cat([x[:batch_size], h], dim=0)

        out = self.pool.synchronize_pull()[:n_id.numel() - batch_size]
        self.pool.async_push(x[:batch_size], offset, count, history.emb)
        out = torch.cat([x[:batch_size], out], dim=0)
        self.pool.free_pull()
        return out

    @torch.no_grad()
    def mini_inference(self, loader) -> Tensor:
        loader = [data + ({}, ) for data in loader]

        for batch, batch_size, n_id, offset, count, state in loader:
            x = batch.x.to(self.device)
            adj_t = batch.adj_t.to(self.device)
            out = self.forward_layer(0, x, adj_t, state)[:batch_size]
            self.pool.async_push(out, offset, count, self.histories[0].emb)
        self.pool.synchronize_push()

        for i in range(1, len(self.histories)):
            for _, batch_size, n_id, offset, count, _ in loader:
                self.pool.async_pull(self.histories[i - 1].emb, offset, count,
                                     n_id[batch_size:])

            for batch, batch_size, n_id, offset, count, state in loader:
                adj_t = batch.adj_t.to(self.device)
                x = self.pool.synchronize_pull()[:n_id.numel()]
                out = self.forward_layer(i, x, adj_t, state)[:batch_size]
                self.pool.async_push(out, offset, count, self.histories[i].emb)
                self.pool.free_pull()
            self.pool.synchronize_push()

        for _, batch_size, n_id, offset, count, _ in loader:
            self.pool.async_pull(self.histories[-1].emb, offset, count,
                                 n_id[batch_size:])

        for batch, batch_size, n_id, offset, count, state in loader:
            adj_t = batch.adj_t.to(self.device)
            x = self.pool.synchronize_pull()[:n_id.numel()]
            out = self.forward_layer(self.num_layers - 1, x, adj_t,
                                     state)[:batch_size]
            self.pool.async_push(out, offset, count, self._out)
            self.pool.free_pull()
        self.pool.synchronize_push()

        return self._out
