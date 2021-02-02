<h1 align="center">PyGAS: Auto-Scaling GNNs in PyG</h1>

<img width="100%" src="https://raw.githubusercontent.com/rusty1s/pyg_autoscale/master/figures/overview.png?token=ABU7ZAXZ7WT3RIOSYHIDIVDAEI3SY" />

--------------------------------------------------------------------------------

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

    def forward(self, x, adj, n_id):
        for conv, history in zip(self.convs[:-1], self.histories):
            x = conv(x, adj).relu()
            x = self.push_and_pull(history, x, n_id)
        return self.convs[-1](x, adj
```
