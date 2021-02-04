import time
import hydra
from omegaconf import OmegaConf

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric_autoscale import (get_data, metis, permute,
                                       SubgraphLoader, EvalSubgraphLoader,
                                       models, compute_acc)
from torch_geometric_autoscale.data import get_ppi

torch.manual_seed(123)


def mini_train(model, loader, criterion, optimizer, max_steps, grad_norm=None):
    model.train()

    total_loss = total_examples = 0
    for i, (batch, batch_size, n_id, offset, count) in enumerate(loader):
        x = batch.x.to(model.device)
        adj_t = batch.adj_t.to(model.device)
        y = batch.y[:batch_size].to(model.device)
        train_mask = batch.train_mask[:batch_size].to(model.device)

        if train_mask.sum() == 0:
            continue

        optimizer.zero_grad()
        out = model(x, adj_t, batch_size, n_id, offset, count)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()

        total_loss += float(loss) * int(train_mask.sum())
        total_examples += int(train_mask.sum())

        if (i + 1) >= max_steps and (i + 1) < len(loader):
            break

    return total_loss / total_examples


@torch.no_grad()
def full_test(model, data):
    model.eval()
    return model(data.x.to(model.device), data.adj_t.to(model.device)).cpu()


@torch.no_grad()
def mini_test(model, loader):
    model.eval()
    return model(loader=loader)


@hydra.main(config_path='conf', config_name='config')
def main(conf):
    conf.model.params = conf.model.params[conf.dataset.name]
    params = conf.model.params
    print(OmegaConf.to_yaml(conf))
    grad_norm = None if isinstance(params.grad_norm, str) else params.grad_norm

    device = f'cuda:{conf.device}' if torch.cuda.is_available() else 'cpu'

    t = time.perf_counter()
    print('Loading data...', end=' ', flush=True)
    data, in_channels, out_channels = get_data(conf.root, conf.dataset.name)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    perm, ptr = metis(data.adj_t, num_parts=params.num_parts, log=True)
    data = permute(data, perm, log=True)

    if conf.model.loop:
        t = time.perf_counter()
        print('Adding self-loops...', end=' ', flush=True)
        data.adj_t = data.adj_t.set_diag()
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    if conf.model.norm:
        t = time.perf_counter()
        print('Normalizing data...', end=' ', flush=True)
        data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    if data.y.dim() == 1:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = SubgraphLoader(data, ptr, batch_size=params.batch_size,
                                  shuffle=True, num_workers=params.num_workers,
                                  persistent_workers=params.num_workers > 0)

    eval_loader = EvalSubgraphLoader(data, ptr,
                                     batch_size=params['batch_size'])

    if conf.dataset.name == 'ppi':
        val_data, _, _ = get_ppi(conf.root, split='val')
        test_data, _, _ = get_ppi(conf.root, split='test')
        if conf.model.loop:
            val_data.adj_t = val_data.adj_t.set_diag()
            test_data.adj_t = test_data.adj_t.set_diag()
        if conf.model.norm:
            val_data.adj_t = gcn_norm(val_data.adj_t, add_self_loops=False)
            test_data.adj_t = gcn_norm(test_data.adj_t, add_self_loops=False)

    t = time.perf_counter()
    print('Calculating buffer size...', end=' ', flush=True)
    buffer_size = max([n_id.numel() for _, _, n_id, _, _ in eval_loader])
    print(f'Done! [{time.perf_counter() - t:.2f}s] -> {buffer_size}')

    kwargs = {}
    if conf.model.name[:3] == 'PNA':
        kwargs['deg'] = data.adj_t.storage.rowcount()

    GNN = getattr(models, conf.model.name)
    model = GNN(
        num_nodes=data.num_nodes,
        in_channels=in_channels,
        out_channels=out_channels,
        pool_size=params.pool_size,
        buffer_size=buffer_size,
        **params.architecture,
        **kwargs,
    ).to(device)

    optimizer = torch.optim.Adam([
        dict(params=model.reg_modules.parameters(),
             weight_decay=params.reg_weight_decay),
        dict(params=model.nonreg_modules.parameters(),
             weight_decay=params.nonreg_weight_decay)
    ], lr=params.lr)

    t = time.perf_counter()
    print('Fill history...', end=' ', flush=True)
    mini_test(model, eval_loader)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    best_val_acc = test_acc = 0
    for epoch in range(1, params.epochs + 1):
        loss = mini_train(model, train_loader, criterion, optimizer,
                          params.max_steps, grad_norm)
        out = mini_test(model, eval_loader)
        train_acc = compute_acc(out, data.y, data.train_mask)

        if conf.dataset.name != 'ppi':
            val_acc = compute_acc(out, data.y, data.val_mask)
            tmp_test_acc = compute_acc(out, data.y, data.test_mask)
        else:
            val_acc = compute_acc(full_test(model, val_data), val_data.y)
            tmp_test_acc = compute_acc(full_test(model, test_data),
                                       test_data.y)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        if epoch % conf.log_every == 0:
            print(f'Epoch: {epoch:04d}, Loss: {loss:.4f}, '
                  f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {tmp_test_acc:.4f}, Final: {test_acc:.4f}')

    print('=========================')
    print(f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')


if __name__ == "__main__":
    main()
