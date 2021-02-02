from typing import Optional

import torch
from torch import Tensor


class History(torch.nn.Module):
    r"""A node embedding storage module with asynchronous I/O support between
    devices."""
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None):
        super(History, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        pin_memory = device is None or str(device) == 'cpu'
        self.emb = torch.empty(num_embeddings, embedding_dim, device=device,
                               pin_memory=pin_memory)

        self._device = torch.device('cpu')

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.fill_(0)

    def _apply(self, fn):
        self._device = fn(torch.zeros(1)).device
        return self

    @torch.no_grad()
    def pull(self, index: Optional[Tensor] = None) -> Tensor:
        out = self.emb
        if index is not None:
            assert index.device == self.emb.device
            out = out.index_select(0, index)
        return out.to(device=self._device)

    @torch.no_grad()
    def push(self, x, index: Optional[Tensor] = None,
             offset: Optional[Tensor] = None, count: Optional[Tensor] = None):

        if index is None and x.size(0) != self.num_embeddings:
            raise ValueError

        elif index is None and x.size(0) == self.num_embeddings:
            self.emb.copy_(x)

        elif index is not None and (offset is None or count is None):
            assert index.device == self.emb.device
            self.emb[index] = x.to(self.emb.device)

        else:
            x_o = 0
            x = x.to(self.emb.device)
            for o, c, in zip(offset.tolist(), count.tolist()):
                self.emb[o:o + c] = x[x_o:x_o + c]
                x_o += c

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_embeddings}, '
                f'{self.embedding_dim}, emb_device={self.emb.device}, '
                f'device={self._device})')
