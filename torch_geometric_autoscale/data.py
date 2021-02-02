import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Batch
from torch_geometric.datasets import (Planetoid, WikiCS, Coauthor, Amazon,
                                      GNNBenchmarkDataset, Yelp, Flickr,
                                      Reddit2, PPI)

from .utils import index2mask, gen_masks


def get_planetoid(root, name):
    dataset = Planetoid(
        f'{root}/Planetoid', name,
        transform=T.Compose([T.NormalizeFeatures(),
                             T.ToSparseTensor()]))

    return dataset[0], dataset.num_features, dataset.num_classes


def get_wikics(root):
    dataset = WikiCS(f'{root}/WIKICS', transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data.val_mask = data.stopping_mask
    data.stopping_mask = None
    return data, dataset.num_features, dataset.num_classes


def get_coauthor(root, name):
    dataset = Coauthor(f'{root}/Coauthor', name, transform=T.ToSparseTensor())
    data = dataset[0]
    torch.manual_seed(12345)
    data.train_mask, data.val_mask, data.test_mask = gen_masks(
        data.y, 20, 30, 20)
    return data, dataset.num_features, dataset.num_classes


def get_amazon(root, name):
    dataset = Amazon(f'{root}/Amazon', name, transform=T.ToSparseTensor())
    data = dataset[0]
    torch.manual_seed(12345)
    data.train_mask, data.val_mask, data.test_mask = gen_masks(
        data.y, 20, 30, 20)
    return data, dataset.num_features, dataset.num_classes


def get_arxiv(root):
    dataset = PygNodePropPredDataset('ogbn-arxiv', f'{root}/OGB',
                                     pre_transform=T.ToSparseTensor())
    data = dataset[0]

    data.adj_t = data.adj_t.to_symmetric()
    data.node_year = None
    data.y = data.y.view(-1)

    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)

    return data, dataset.num_features, dataset.num_classes


def get_products(root):
    dataset = PygNodePropPredDataset('ogbn-products', f'{root}/OGB',
                                     pre_transform=T.ToSparseTensor())
    data = dataset[0]

    data.y = data.y.view(-1)

    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)

    return data, dataset.num_features, dataset.num_classes


def get_proteins(root):
    dataset = PygNodePropPredDataset('ogbn-proteins', f'{root}/OGB',
                                     pre_transform=T.ToSparseTensor())
    data = dataset[0]

    data.node_species = None
    data.y = data.y.to(torch.float)

    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)

    return data, dataset.num_features, data.y.size(-1)


def get_yelp(root):
    dataset = Yelp(f'{root}/YELP', pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes


def get_flickr(root):
    dataset = Flickr(f'{root}/Flickr', pre_transform=T.ToSparseTensor())
    return dataset[0], dataset.num_features, dataset.num_classes


def get_reddit(root):
    dataset = Reddit2(f'{root}/Reddit2', pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes


def get_ppi(root, split='train'):
    dataset = PPI(f'{root}/PPI', split=split, pre_transform=T.ToSparseTensor())

    data = Batch.from_data_list(dataset)
    data.batch = None
    data.ptr = None
    data[f'{split}_mask'] = torch.ones(data.num_nodes, dtype=torch.bool)

    return data, dataset.num_features, dataset.num_classes


def get_sbm(root, name):
    dataset = GNNBenchmarkDataset(f'{root}/SBM', name, split='train',
                                  pre_transform=T.ToSparseTensor())

    data = Batch.from_data_list(dataset)
    data.batch = None
    data.ptr = None

    return data, dataset.num_features, dataset.num_classes


def get_data(root, name):
    if name.lower() in ['cora', 'citeseer', 'pubmed']:
        return get_planetoid(root, name)
    if name.lower() == 'wikics':
        return get_wikics(root)
    if name.lower() in ['coauthorcs', 'coauthorphysics']:
        return get_coauthor(root, name[8:])
    if name.lower() in ['amazoncomputers', 'amazonphoto']:
        return get_amazon(root, name[6:])
    if name.lower() in ['ogbn-arxiv', 'arxiv']:
        return get_arxiv(root)
    if name.lower() in ['ogbn-products', 'products']:
        return get_products(root)
    if name.lower() == ['ogbn-proteins', 'proteins']:
        return get_proteins(root)
    if name.lower() == 'yelp':
        return get_yelp(root)
    if name.lower() == 'flickr':
        return get_flickr(root)
    if name.lower() == 'reddit':
        return get_reddit(root)
    if name.lower() == 'ppi':
        return get_ppi(root)
    if name.lower() in ['cluster', 'pattern']:
        return get_sbm(root, name)
    raise NotImplementedError
