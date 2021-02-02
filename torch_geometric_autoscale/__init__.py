import importlib
import os.path as osp

import torch

__version__ = '0.0.0'

for library in ['_relabel', '_async']:
    torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
        library, [osp.dirname(__file__)]).origin)

from .history import History  # noqa
from .loader import SubgraphLoader  # noqa
from .data import get_data  # noqa
from .utils import compute_acc  # noqa

__all__ = [
    'History',
    'SubgraphLoader',
    'get_data',
    'compute_acc',
    '__version__',
]
