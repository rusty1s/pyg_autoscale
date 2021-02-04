import hydra
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric_autoscale import (get_data, metis, permute,
                                       SubgraphLoader, models, compute_acc)

torch.manual_seed(123)
criterion = torch.nn.CrossEntropyLoss()


def train(run, model, loader, optimizer, grad_norm=None):
    model.train()

    total_loss = total_examples = 0
    for batch, batch_size, n_id, _, _ in loader:
        batch = batch.to(model.device)
        n_id = n_id.to(model.device)

        mask = batch.train_mask[:batch_size]
        mask = mask[:, run] if mask.dim() == 2 else mask
        if mask.sum() == 0:
            continue

        optimizer.zero_grad()
        out = model(batch.x, batch.adj_t, batch_size, n_id)
        loss = criterion(out[mask], batch.y[:batch_size][mask])
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()

        total_loss += float(loss) * int(mask.sum())
        total_examples += int(mask.sum())

    return total_loss / total_examples


@torch.no_grad()
def test(run, model, data):
    model.eval()

    val_mask = data.val_mask
    val_mask = val_mask[:, run] if val_mask.dim() == 2 else val_mask

    test_mask = data.test_mask
    test_mask = test_mask[:, run] if test_mask.dim() == 2 else test_mask

    out = model(data.x, data.adj_t)
    val_acc = compute_acc(out, data.y, val_mask)
    test_acc = compute_acc(out, data.y, test_mask)

    return val_acc, test_acc


@hydra.main(config_path='conf', config_name='config')
def main(conf):
    model_name, dataset_name = conf.model.name, conf.dataset.name
    conf.model.params = conf.model.params[dataset_name]
    params = conf.model.params
    print(OmegaConf.to_yaml(conf))
    if isinstance(params.grad_norm, str):
        params.grad_norm = None

    device = f'cuda:{conf.device}' if torch.cuda.is_available() else 'cpu'

    data, in_channels, out_channels = get_data(conf.root, dataset_name)
    if conf.model.norm:
        data.adj_t = gcn_norm(data.adj_t)
    elif conf.model.loop:
        data.adj_t = data.adj_t.set_diag()

    perm, ptr = metis(data.adj_t, num_parts=params.num_parts, log=True)
    data = permute(data, perm, log=True)

    loader = SubgraphLoader(data, ptr, batch_size=params.batch_size,
                            shuffle=True, num_workers=params.num_workers,
                            persistent_workers=params.num_workers > 0)

    data = data.clone().to(device)  # Let's just store all data on GPU...

    GNN = getattr(models, model_name)
    model = GNN(
        num_nodes=data.num_nodes,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,  # Put histories on GPU.
        **params.architecture,
    ).to(device)

    results = torch.empty(params.runs)
    pbar = tqdm(total=params.runs * params.epochs)
    for run in range(params.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam([
            dict(params=model.reg_modules.parameters(),
                 weight_decay=params.reg_weight_decay),
            dict(params=model.nonreg_modules.parameters(),
                 weight_decay=params.nonreg_weight_decay)
        ], lr=params.lr)

        test(0, model, data)  # Fill history.

        best_val_acc = 0
        for epoch in range(params.epochs):
            train(run, model, loader, optimizer, params.grad_norm)
            val_acc, test_acc = test(run, model, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                results[run] = test_acc

            pbar.set_description(f'Mini Acc: {100 * results[run]:.2f}')
            pbar.update(1)
    pbar.close()
    print(f'Mini Acc: {100 * results.mean():.2f} ± {100 * results.std():.2f}')


if __name__ == "__main__":
    main()
