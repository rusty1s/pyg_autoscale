import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from scaling_gnns import get_data, models, SubgraphLoader, compute_acc

torch.manual_seed(123)
criterion = torch.nn.CrossEntropyLoss()


def train(run, data, model, loader, optimizer, grad_norm=None):
    model.train()

    train_mask = data.train_mask
    train_mask = train_mask[:, run] if train_mask.dim() == 2 else train_mask

    total_loss = total_examples = 0
    for info in loader:
        info = info.to(model.device)
        batch_size, n_id, adj_t, e_id = info

        y = data.y[n_id[:batch_size]]
        mask = train_mask[n_id[:batch_size]]

        if mask.sum() == 0:
            continue

        optimizer.zero_grad()
        out = model(data.x[n_id], adj_t, batch_size, n_id)
        loss = criterion(out[mask], y[mask])
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()

        total_loss += float(loss) * int(mask.sum())
        total_examples += int(mask.sum())

    return total_loss / total_examples


@torch.no_grad()
def test(run, data, model):
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

    loader = SubgraphLoader(
        data.adj_t,
        batch_size=params.batch_size,
        use_metis=True,
        num_parts=params.num_parts,
        shuffle=True,
        num_workers=params.num_workers,
        path=f'../../../metis/{model_name.lower()}_{dataset_name.lower()}',
        log=False,
    )

    data = data.to(device)

    GNN = getattr(models, model_name)
    model = GNN(
        num_nodes=data.num_nodes,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
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

        with torch.no_grad():  # Fill history.
            model.eval()
            model(data.x, data.adj_t)

        best_val_acc = 0
        for epoch in range(params.epochs):
            train(run, data, model, loader, optimizer, params.grad_norm)
            val_acc, test_acc = test(run, data, model)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                results[run] = test_acc

            pbar.set_description(f'Mini Acc: {100 * results[run]:.2f}')
            pbar.update(1)
    pbar.close()
    print(f'Mini Acc: {100 * results.mean():.2f} ± {100 * results.std():.2f}')


if __name__ == "__main__":
    main()
