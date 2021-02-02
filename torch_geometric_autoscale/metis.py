import time
import copy
from typing import Union, Tuple

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.data import Data


def metis(adj_t: SparseTensor, num_parts: int, recursive: bool = False,
          log: bool = True) -> Tuple[Tensor, Tensor]:

    if log:
        t = time.perf_counter()
        print(f'Computing METIS partitioning with {num_parts} parts...',
              end=' ', flush=True)

    num_nodes = adj_t.size(0)

    if num_parts <= 1:
        perm, ptr = torch.arange(num_nodes), torch.tensor([0, num_nodes])
    else:
        rowptr, col, _ = adj_t.csr()
        cluster = torch.ops.torch_sparse.partition(rowptr, col, None,
                                                   num_parts, recursive)
        cluster, perm = cluster.sort()
        ptr = torch.ops.torch_sparse.ind2ptr(cluster, num_parts)

    if log:
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    return perm, ptr


def permute(data: Union[Data, SparseTensor], perm: Tensor,
            log: bool = True) -> Union[Data, SparseTensor]:

    if log:
        t = time.perf_counter()
        print('Permuting data...', end=' ', flush=True)

    if isinstance(data, Data):
        data = copy.copy(data)
        for key, item in data:
            if isinstance(item, Tensor) and item.size(0) == data.num_nodes:
                data[key] = item[perm]
            if isinstance(item, Tensor) and item.size(0) == data.num_edges:
                raise NotImplementedError
            if isinstance(item, SparseTensor):
                data[key] = permute(item, perm, log=False)
    else:
        data = data.permute(perm)

    if log:
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    return data
