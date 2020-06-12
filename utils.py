import os
import torch
import dgl
import dgl.data.citation_graph as dglcitationgraph
import torch_geometric as pyg 
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch_geometric.utils as pygutils
from os.path import join as opjoin


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def resplit(dataset, data, full_sup, num_classes, num_nodes, num_per_class):
    if dataset in ['cora', 'citeseer', 'pubmed']:
        if full_sup:
            perm = torch.randperm(data[2].shape[0])
            test_index = perm[:500]
            val_index = perm[500:1500]
            train_index = perm[1500:]

            data[3] = index_to_mask(train_index, size=num_nodes)
            data[4] = index_to_mask(val_index, size=num_nodes)
            data[5] = index_to_mask(test_index, size=num_nodes)
        else:
            indices = []
            for i in range(num_classes):
                index = (data[2].long() == i).nonzero().view(-1) 
                index = index[torch.randperm(index.size(0))]
                indices.append(index)

            train_index = torch.cat([i[ : num_per_class] for i in indices], dim=0)

            rest_index = torch.cat([i[num_per_class : ] for i in indices], dim=0)
            rest_index = rest_index[torch.randperm(rest_index.size(0))]

            data[3] = index_to_mask(train_index, size=num_nodes)
            data[4] = index_to_mask(rest_index[:500], size=num_nodes)
            data[5] = index_to_mask(rest_index[500:1500], size=num_nodes)

    elif dataset in ['coauthorcs']:
        if full_sup:
            raise NotImplementedError
        else:
            train_index = []
            val_index = []
            test_index = []
            for i in range(num_classes):
                index = (data[2].long() == i).nonzero().view(-1) 
                index = index[torch.randperm(index.size(0))]
                if len(index) > num_per_class + 30:
                    train_index.append(index[ : num_per_class])
                    val_index.append(index[num_per_class : num_per_class + 30])
                    test_index.append(index[num_per_class + 30:])
                else:
                    continue
                
        train_index = torch.cat(train_index)
        val_index = torch.cat(val_index)
        test_index = torch.cat(test_index)

        data[3] = index_to_mask(train_index, size=num_nodes)
        data[4] = index_to_mask(val_index, size=num_nodes)
        data[5] = index_to_mask(test_index, size=num_nodes)

    return data


def load_data(config):
    if not config['data']['dataset'] in config['data']['all_datasets']:
        raise NotImplementedError

    else:
        if config['data']['implement'] == 'dgl':
            graph = get_dgl_dataset(dataroot=config['path']['dataroot'],
                                    dataset=config['data']['dataset'])
            graph = rewarp_graph(graph, config)
            features = torch.tensor(graph.features, dtype=torch.float)
            if config['data']['feature_prenorm']:
                features = row_normalization(features)

            labels = torch.tensor(graph.labels, dtype=torch.long)
            idx_train = torch.tensor(graph.train_mask, dtype=torch.bool)
            idx_val = torch.tensor(graph.val_mask, dtype=torch.bool)
            idx_test = torch.tensor(graph.test_mask, dtype=torch.bool)
            g_nx = graph.graph

            if config['data']['add_slflp']:
                g_nx.remove_edges_from(nx.selfloop_edges(g_nx))
                g_nx.add_edges(zip(g_nx.nodes(), g_nx.nodes()))
            graph_skeleton = dgl.DGLGraph(g_nx)
            return [graph_skeleton, features, labels, idx_train, idx_val, idx_test]

        elif config['data']['implement'] == 'pyg':
            graph = get_pyg_dataset(dataroot=config['path']['dataroot'], dataset=config['data']['dataset'])
            graph = rewarp_graph(graph, config)
            data = graph.data
            idx_train = data.train_mask
            idx_val = data.val_mask
            idx_test = data.test_mask
            labels = data.y
            num_nodes = data.num_nodes
            features = data.x

            if config['data']['feature_prenorm']:
                features = row_normalization(features)
            edge_index = data.edge_index
            if config['data']['add_slflp']:
                edge_index = pygutils.add_self_loops(edge_index)[0]
            graph_skeleton = dgl.DGLGraph()
            graph_skeleton.add_nodes(num_nodes)
            graph_skeleton.add_edges(edge_index[0, :], edge_index[1, :])
            return [graph_skeleton, features, labels, idx_train, idx_val, idx_test]

        else:
            raise NotImplementedError


def dummy_normalization(mx):
    if isinstance(mx, np.ndarray) or isinstance(mx, sp.csr.csr_matrix):
        pass
    elif isinstance(mx, sp.lil.lil_matrix):
        mx = np.asarray(mx.todense())
    else:
        raise NotImplementedError
    return mx


def get_dgl_dataset(dataroot, dataset):
    dglcitationgraph._normalize = dummy_normalization 
    dglcitationgraph._preprocess_features = dummy_normalization
    if dataset == 'cora':
        graph = dgl.data.CoraDataset()
    elif dataset in ['citeseer', 'pubmed']:
        graph = dgl.data.CitationGraphDataset(name=dataset)
    elif dataset == 'coauthorcs':
        np.load.__defaults__ = (None, True, True, 'ASCII')
        graph = dgl.data.Coauthor(name='cs')
        np.load.__defaults__ = (None, False, True, 'ASCII')
    else:
        raise NotImplementedError
    return graph


def get_pyg_dataset(dataroot, dataset):
    if dataset in ['cora', 'citeseer', 'pubmed']:
        graph = pyg.datasets.Planetoid(root=opjoin(dataroot, dataset), name=dataset.capitalize())
    elif dataset == 'coauthorcs':
        graph = pyg.datasets.Coauthor(root=opjoin(dataroot, dataset), name='CS')
    else:
        raise NotImplementedError
    return graph


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct_sum = correct.sum()
    return correct_sum / len(labels), correct


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# Actually Sacred will automatically save source code files added by its add_source_file method (please see line 28 in config.py)
# But it adds a md5 hash string after the file name, and saves the source codes for every run in the same file
# So here we manually call its save_file method to save the source codes of given name in the location speicified by us
# And finally delete the source codes saved by Sacred 
def save_source(run):

    if run.observers:
        for source_file, _ in run.experiment_info['sources']:
            os.makedirs(os.path.dirname('{0}/source/{1}'.format(run.observers[0].dir, source_file)), exist_ok=True)
            run.observers[0].save_file(source_file, 'source/{0}'.format(source_file))
        sacred_source_path = f'{run.observers[0].basedir}/_sources'
        # if os.path.exists(sacred_source_path):
        #     shutil.rmtree(sacred_source_path)


def adjust_learning_rate(optimizer, epoch, lr_down_epoch_list, logger):

    if epoch != 0 and epoch in lr_down_epoch_list:
        opt_name = list(dict(optimizer=optimizer).keys())[0]
        logger.info('update learning rate of ' + opt_name)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            logger.info(param_group['lr'])


def check_before_pkl(data):
    if type(data) == list or type(data) == tuple:
        for each in data:
            check_before_pkl(each)
    elif type(data) == dict:
        for key in data.keys():
            check_before_pkl(data[key])
    else:
        assert not isinstance(data, torch.Tensor)


def row_normalization(features):
    ## normalize the feature matrix by its row sum
    rowsum = features.sum(dim=1)
    inv_rowsum = torch.pow(rowsum, -1)
    inv_rowsum[torch.isinf(inv_rowsum)] = 0. 
    features = features * inv_rowsum[..., None]

    return features


def rewarp_graph(graph, config):
    if pyg.__version__  in ['1.4.2', '1.3.2']:
        pyg_corafull_type = pyg.datasets.cora_full.CoraFull 
    else:
        pyg_corafull_type = pyg.datasets.citation_full.CoraFull # pylint: disable=no-member

    if isinstance(graph, dgl.data.gnn_benckmark.Coauthor) or \
       isinstance(graph, dgl.data.gnn_benckmark.CoraFull):
        graph = PseudoDGLGraph(graph)
        pseudo_data = [None, None, graph.labels, None, None, None]
        _, _, _, train_mask, val_mask, test_mask = resplit(dataset=config['data']['dataset'],
                                                           data=pseudo_data,
                                                           full_sup=config['data']['full_sup'],
                                                           num_classes=graph.num_classes,
                                                           num_nodes=graph.num_nodes,
                                                           num_per_class=config['data']['label_per_class'])
        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask

    elif isinstance(graph, pyg_corafull_type) or \
         isinstance(graph, pyg.datasets.coauthor.Coauthor):
        pseudo_data = [None, None, graph.data.y, None, None, None]
        _, _, _, train_mask, val_mask, test_mask = resplit(dataset=config['data']['dataset'],
                                                           data=pseudo_data,
                                                           full_sup=config['data']['full_sup'],
                                                           num_classes=torch.unique(graph.data.y).shape[0],
                                                           num_nodes=graph.data.num_nodes,
                                                           num_per_class=config['data']['label_per_class'])
        graph.data.train_mask = train_mask
        graph.data.val_mask = val_mask
        graph.data.test_mask = test_mask 
    else:
        pass
    return graph


class PseudoDGLGraph():
    def __init__(self, graph):
        self.graph = graph.data[0].to_networkx()
        self.features = graph.data[0].ndata['feat']
        self.labels = graph.data[0].ndata['label']
        self.num_classes = torch.unique(self.labels).shape[0]
        self.num_nodes = graph.data[0].number_of_nodes()
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None




