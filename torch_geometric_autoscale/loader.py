from typing import Optional, Union, Tuple, NamedTuple, List

import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.data import Data

relabel_fn = torch.ops.torch_geometric_autoscale.relabel_one_hop


class SubData(NamedTuple):
    data: Union[Data, SparseTensor]
    batch_size: int
    n_id: Tensor
    offset: Optional[Tensor]
    count: Optional[Tensor]

    def to(self, *args, **kwargs):
        return SubData(self.data.to(*args, **kwargs), self.batch_size,
                       self.n_id, self.offset, self.count)


class SubgraphLoader(DataLoader):
    r"""A simple subgraph loader that, given a randomly sampled or
    pre-partioned batch of nodes, returns the subgraph of this batch
    (including its 1-hop neighbors)."""
    def __init__(
        self,
        data: Union[Data, SparseTensor],
        ptr: Optional[Tensor] = None,
        batch_size: int = 1,
        bipartite: bool = True,
        log: bool = True,
        **kwargs,
    ):
        self.__data__ = None if isinstance(data, SparseTensor) else data
        self.__adj_t__ = data if isinstance(data, SparseTensor) else data.adj_t

        self.__N__ = self.__adj_t__.size(1)
        self.__E__ = self.__adj_t__.nnz()
        self.__ptr__ = ptr
        self.__bipartite__ = bipartite

        if ptr is not None:
            n_id = torch.arange(self.__N__)
            batches = n_id.split((ptr[1:] - ptr[:-1]).tolist())
            batches = [(i, batches[i]) for i in range(len(batches))]

            if batch_size > 1:
                super(SubgraphLoader,
                      self).__init__(batches,
                                     collate_fn=self.sample_partitions,
                                     batch_size=batch_size, **kwargs)
            else:
                if log:
                    t = time.perf_counter()
                    print('Pre-processing subgraphs...', end=' ', flush=True)

                data_list = [
                    data for data in DataLoader(
                        batches, collate_fn=self.sample_partitions,
                        batch_size=batch_size, **kwargs)
                ]

                if log:
                    print(f'Done! [{time.perf_counter() - t:.2f}s]')

                super(SubgraphLoader,
                      self).__init__(data_list, batch_size=1,
                                     collate_fn=lambda x: x[0], **kwargs)

        else:
            super(SubgraphLoader,
                  self).__init__(range(self.__N__),
                                 collate_fn=self.sample_nodes,
                                 batch_size=batch_size, **kwargs)

    def sample_partitions(self, batches: List[Tuple[int, Tensor]]) -> SubData:
        ptr_ids, n_ids = zip(*batches)

        n_id = torch.cat(n_ids, dim=0)
        batch_size = n_id.numel()

        ptr_id = torch.tensor(ptr_ids)
        offset = self.__ptr__[ptr_id]
        count = self.__ptr__[ptr_id.add_(1)].sub_(offset)

        rowptr, col, value = self.__adj_t__.csr()
        rowptr, col, value, n_id = relabel_fn(rowptr, col, value, n_id,
                                              self.__bipartite__)

        adj_t = SparseTensor(rowptr=rowptr, col=col, value=value,
                             sparse_sizes=(rowptr.numel() - 1, n_id.numel()),
                             is_sorted=True)

        if self.__data__ is None:
            return SubData(adj_t, batch_size, n_id, offset, count)

        data = self.__data__.__class__(adj_t=adj_t)
        for key, item in self.__data__:
            if isinstance(item, Tensor) and item.size(0) == self.__N__:
                data[key] = item.index_select(0, n_id)
            elif isinstance(item, SparseTensor):
                pass
            else:
                data[key] = item

        return SubData(data, batch_size, n_id, offset, count)

    def sample_nodes(self, n_ids: List[int]) -> SubData:
        n_id = torch.tensor(n_ids)
        batch_size = n_id.numel()

        rowptr, col, value = self.__adj_t__.csr()
        rowptr, col, value, n_id = relabel_fn(rowptr, col, value, n_id,
                                              self.__bipartite__)

        adj_t = SparseTensor(rowptr=rowptr, col=col, value=value,
                             sparse_sizes=(rowptr.numel() - 1, n_id.numel()),
                             is_sorted=True)

        if self.__data__ is None:
            return SubData(adj_t, batch_size, n_id, None, None)

        data = self.__data__.__class__(adj_t=adj_t)
        for key, item in self.__data__:
            if isinstance(item, Tensor) and item.size(0) == self.__N__:
                data[key] = item.index_select(0, n_id)
            elif isinstance(item, SparseTensor):
                pass
            else:
                data[key] = item

        return SubData(data, batch_size, n_id, None, None)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class EvalSubgraphLoader(SubgraphLoader):
    def __init__(
        self,
        data: Union[Data, SparseTensor],
        ptr: Optional[Tensor] = None,
        batch_size: int = 1,
        bipartite: bool = True,
        log: bool = True,
        **kwargs,
    ):

        num_nodes = ptr[-1]
        ptr = ptr[::batch_size]
        if int(ptr[-1]) != int(num_nodes):
            ptr = torch.cat([ptr, num_nodes.unsqueeze(0)], dim=0)

        super(EvalSubgraphLoader,
              self).__init__(data, ptr, 1, bipartite, log, num_workers=0,
                             shuffle=False, **kwargs)
