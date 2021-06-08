import argparse

import torch
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLR
from torch.nn import Identity, Sequential, Linear, ReLU, BatchNorm1d
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn import GINConv
from torch_geometric.data import DataLoader
from torch_geometric.datasets import GNNBenchmarkDataset as SBM

from torch_geometric_autoscale import get_data
from torch_geometric_autoscale import metis, permute
from torch_geometric_autoscale.models import ScalableGNN
from torch_geometric_autoscale import SubgraphLoader, EvalSubgraphLoader

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True,
                    help='Root directory of dataset storage.')
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(123)
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

data, in_channels, out_channels = get_data(args.root, name='CLUSTER')

# Pre-partition the graph using Metis:
perm, ptr = metis(data.adj_t, num_parts=10000, log=True)
data = permute(data, perm, log=True)

train_loader = SubgraphLoader(data, ptr, batch_size=256, shuffle=True,
                              num_workers=6, persistent_workers=True)
eval_loader = EvalSubgraphLoader(data, ptr, batch_size=256)

# We use the regular PyTorch Geometric dataset for evaluation:
kwargs = {'name': 'CLUSTER', 'pre_transform': T.ToSparseTensor()}
val_dataset = SBM(f'{args.root}/SBM', split='val', **kwargs)
test_dataset = SBM(f'{args.root}/SBM', split='test', **kwargs)
val_loader = DataLoader(val_dataset, batch_size=512)
test_loader = DataLoader(test_dataset, batch_size=512)


# We define our own GAS+GIN module:
class GIN(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size=2,
                         buffer_size=60000)
        # pool_size determines the number of pinned CPU buffers
        # buffer_size determines the size of pinned CPU buffers,
        #             i.e. the maximum number of out-of-mini-batch nodes

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GINConv(Identity(), train_eps=True))

        self.mlps = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm1d(hidden_channels, track_running_stats=False),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
            )
            self.mlps.append(mlp)

    def forward(self, x: Tensor, adj_t: SparseTensor, *args):
        x = self.lins[0](x).relu_()

        reg = 0
        it = zip(self.convs[:-1], self.mlps[:-1], self.histories)
        for i, (conv, mlp, history) in enumerate(it):
            h = conv((x, x[:adj_t.size(0)]), adj_t)

            # Regularize Lipschitz continuity via regularization (part 1):
            if i > 0 and self.training:
                approx = mlp(h + 0.1 * torch.randn_like(h))

            h = mlp(h)

            # Regularize Lipschitz continuity via regularization (part 2):
            if i > 0 and self.training:
                diff = (h - approx).norm(dim=-1)
                reg += diff.mean() / len(self.histories)

            h += x[:h.size(0)]  # Simple skip-connection
            x = self.push_and_pull(history, h, *args)

        h = self.convs[-1]((x, x[:adj_t.size(0)]), adj_t)
        h = self.mlps[-1](h)
        h += x[:h.size(0)]
        x = self.lins[1](h)

        return x, reg

    @torch.no_grad()
    def forward_layer(self, layer: int, x: Tensor, adj_t: SparseTensor, state):
        # Perform layer-wise inference:
        if layer == 0:
            x = self.lins[0](x).relu_()

        h = self.convs[layer]((x, x[:adj_t.size(0)]), adj_t)
        h = self.mlps[layer](h)
        h += x[:h.size(0)]

        if layer == self.num_layers - 1:
            h = self.lins[1](h)

        return h


model = GIN(
    num_nodes=data.num_nodes,
    in_channels=in_channels,
    hidden_channels=128,
    out_channels=out_channels,
    num_layers=4,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
scheduler = ReduceLR(optimizer, 'max', factor=0.5, patience=20, min_lr=1e-5)


def train(model, loader, optimizer):
    model.train()

    total_loss = total_examples = 0
    for batch, *args in loader:
        batch = batch.to(model.device)
        optimizer.zero_grad()
        out, reg = model(batch.x, batch.adj_t, *args)
        loss = criterion(out, batch.y[:out.size(0)]) + reg
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * int(out.size(0))
        total_examples += int(out.size(0))

    return total_loss / total_examples


@torch.no_grad()
def mini_test(model, loader, y):
    model.eval()
    out = model(loader=loader)
    return int((out.argmax(dim=-1) == y).sum()) / y.size(0)


@torch.no_grad()
def full_test(model, loader):
    model.eval()

    total_correct = total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        out, _ = model(batch.x, batch.adj_t)
        total_correct += int((out.argmax(dim=-1) == batch.y).sum())
        total_examples += out.size(0)

    return total_correct / total_examples


mini_test(model, eval_loader, data.y)  # Fill history.

for epoch in range(1, 151):
    lr = optimizer.param_groups[0]['lr']
    loss = train(model, train_loader, optimizer)
    train_acc = mini_test(model, eval_loader, data.y)
    val_acc = full_test(model, val_loader)
    test_acc = full_test(model, test_loader)
    scheduler.step(val_acc)
    print(f'Epoch: {epoch:03d}, LR: {lr:.5f} Loss: {loss:.4f}, '
          f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
