<h1 align="center">PyGAS: Auto-Scaling GNNs in PyG</h1>

<img width="100%" src="https://raw.githubusercontent.com/rusty1s/pyg_autoscale/master/figures/overview.png?token=ABU7ZAXZ7WT3RIOSYHIDIVDAEI3SY" />

--------------------------------------------------------------------------------

*PyGAS* is the practical realization of our *<ins>G</ins>NN<ins>A</ins>uto<ins>S</ins>cale* (GAS) framework, which scales arbitrary message-passing GNNs to large graphs.
*PyGAS* allows for training and inference of GNNs with a constant GPU memory footprint, while it does not drop any input data in comparison to related scalability approaches.
Our approach based on historical node embeddings is provably able to keep the existing expressiveness properties of the underlying message passing implementation.

*PyGAS* is implemented in [PyTorch](https://pytorch.org/) and utilizes the [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) (PyG) library.
It provides an easy-to-use interface to convert common and custom GNNs from PyG into its scalable variant:

```python
from torch_geometric.nn import GCNConv
from torch_geometric_autoscale import ScalableGNN


class GNN(ScalableGNN):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, num_layers):
        super(GNN, self).__init__(num_nodes, hidden_channels, num_layers)

        self.convs = ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, adj_t, batch_size, n_id):
        for conv, history in zip(self.convs[:-1], self.histories):
            x = conv(x, adj_t).relu_()
            x = self.push_and_pull(history, x, batch_size, n_id)
        return self.convs[-1](x, adj_t)
```

## Installation

* Install [**PyTorch >= 1.7.0**](https://pytorch.org/get-started/locally/)
* Install [**PyTorch Geometric**](https://github.com/rusty1s/pytorch_geometric#pytorch-170171) from **master**:

```
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install git+https://github.com/rusty1s/pytorch_geometric.git
```

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101`, `cu102`, or `cu110` depending on your PyTorch installation.

Then, run:

```
python setup.py install
```

## Project Structure

## Running tests

```
python setup.py test
```
