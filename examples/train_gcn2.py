import argparse

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric_autoscale.models import GCN2
from torch_geometric_autoscale import metis, permute, SubgraphLoader
from torch_geometric_autoscale import get_data, compute_acc

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True,
                    help='Root directory of dataset storage.')
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(12345)
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

data, in_channels, out_channels = get_data(args.root, name='cora')

# Pre-process adjacency matrix for GCN:
data.adj_t = gcn_norm(data.adj_t, add_self_loops=True)

# Pre-partition the graph using Metis:
perm, ptr = metis(data.adj_t, num_parts=40, log=True)
data = permute(data, perm, log=True)

loader = SubgraphLoader(data, ptr, batch_size=20, shuffle=True)

# Make use of the pre-defined GCN+GAS model:
model = GCN2(
    num_nodes=data.num_nodes,
    in_channels=in_channels,
    hidden_channels=64,
    out_channels=out_channels,
    num_layers=64,
    alpha=0.1,
    theta=0.5,
    shared_weights=True,
    dropout=0.6,
    drop_input=True,
    pool_size=2,  # Number of pinned CPU buffers
    buffer_size=500,  # Size of pinned CPU buffers (max #out-of-batch nodes)
).to(device)

optimizer = torch.optim.Adam([
    dict(params=model.reg_modules.parameters(), weight_decay=0.01),
    dict(params=model.nonreg_modules.parameters(), weight_decay=5e-4)
], lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train(model, loader, optimizer):
    model.train()

    for batch, *args in loader:
        batch = batch.to(model.device)
        optimizer.zero_grad()
        out = model(batch.x, batch.adj_t, *args)
        train_mask = batch.train_mask[:out.size(0)]
        loss = criterion(out[train_mask], batch.y[:out.size(0)][train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def test(model, data):
    model.eval()

    # Full-batch inference since the graph is small
    out = model(data.x.to(model.device), data.adj_t.to(model.device)).cpu()
    train_acc = compute_acc(out, data.y, data.train_mask)
    val_acc = compute_acc(out, data.y, data.val_mask)
    test_acc = compute_acc(out, data.y, data.test_mask)

    return train_acc, val_acc, test_acc


test(model, data)  # Fill the history.

best_val_acc = test_acc = 0
for epoch in range(1, 501):
    train(model, loader, optimizer)
    train_acc, val_acc, tmp_test_acc = test(model, data)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {tmp_test_acc:.4f}, Final: {test_acc:.4f}')
