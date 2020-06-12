import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn


class NodeNorm(nn.Module):
    def __init__(self, unbiased=False, eps=1e-5):
        super(NodeNorm, self).__init__()
        self.unbiased = unbiased
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = (torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        return x


class GNNBasicBlock(nn.Module):
    def __init__(self, layer_type, block_type, activation, **core_layer_hyperparms):
        super(GNNBasicBlock, self).__init__()
        self.layer_type = layer_type
        self.block_type = block_type

        if self.layer_type in ['gcn', 'gcn_res']:
            self.core_layer_type = 'gcn'
            self.core_layer = dglnn.GraphConv(in_feats=core_layer_hyperparms['in_channels'],
                                              out_feats=core_layer_hyperparms['out_channels'],
                                              bias=core_layer_hyperparms['bias']
                                              )
        elif self.layer_type in ['gat', 'gat_res']:
            self.core_layer_type = 'gat'
            self.core_layer = dglnn.GATConv(in_feats=core_layer_hyperparms['in_channels'],
                                            out_feats=int(core_layer_hyperparms['out_channels'] / core_layer_hyperparms['num_heads']),
                                            num_heads=core_layer_hyperparms['num_heads'],
                                            feat_drop=core_layer_hyperparms['feat_drop'],
                                            attn_drop=core_layer_hyperparms['attn_drop']
                                            )
        elif self.layer_type in ['sage', 'sage_res']:
            self.core_layer_type = 'sage'
            self.core_layer = dglnn.SAGEConv(in_feats=core_layer_hyperparms['in_channels'],
                                             out_feats=core_layer_hyperparms['out_channels'],
                                             aggregator_type='mean',
                                             bias=core_layer_hyperparms['bias'])

        else:
            raise NotImplementedError

        acti_type, acti_hyperparam = activation
        if acti_type == 'relu':
            self.activation = nn.ReLU(inplace=acti_hyperparam)
        elif acti_type == 'lkrelu':
            self.activation = nn.LeakyReLU(negative_slope=acti_hyperparam)
        elif acti_type == 'elu':
            self.activation = nn.ELU(inplace=acti_hyperparam)
        elif acti_type == 'no':
            self.activation = None
        else:
            raise NotImplementedError
            
        if 'b' in block_type.split('_'):
            self.batch_norm = nn.BatchNorm1d(num_features=core_layer_hyperparms['out_channels'])
        if 'n' in block_type.split('_'):
            self.node_norm = NodeNorm()
        self.block_type_str = self.get_block_type_str()

    def forward(self, graph, x):
        if self.core_layer_type in ['gcn', 'sage']:
            x1 = self.core_layer(graph, x)
        elif self.core_layer_type in ['gat', ]:
            x1 = self.core_layer(graph, x).flatten(1)
        else:
            x1 = self.core_layer(x)
        if self.block_type == 'v': # vallina layers
            if self.activation is not None:
                x1 = self.activation(x1)
            x = x1
        elif self.block_type == 'a_r': # activation then adding residual link
            x1 = self.activation(x1)
            x = x1 + x
        elif self.block_type == 'b_a': # batchnorm then activation
            x = self.batch_norm(x)
            x = self.activation(x)
        elif self.block_type == 'n_a': # nodenorm then activation
            x = self.node_norm(x1)
            x = self.activation(x)
        elif self.block_type == 'b_a_r': # batchnorm, activation then adding residual link
            x1 = self.batch_norm(x1)
            x1 = self.activation(x1)
            x = x1 + x
        elif self.block_type == 'n_a_r': # nodenorm, activation then adding residual link
            x1 = self.node_norm(x1)
            x1 = self.activation(x1)
            x = x1 + x
        return x

    def get_block_type_str(self):
        if self.block_type == 'v':
            block_type_str = 'vallina'
        elif self.block_type == 'a_r':
            block_type_str = 'activation_residual'
        elif self.block_type == 'b_a':
            block_type_str = 'batchnorm_activation'
        elif self.block_type == 'b_a_r':
            block_type_str = 'batchnorm_activation_residual'
        elif self.block_type == 'n_a_r':
            block_type_str = 'node_normalization_activation_residual'
        elif self.block_type == 'n_a':
            block_type_str = 'node_normalization_activation'
        else:
            raise NotImplementedError

        return block_type_str

    def __repr__(self):
        original_str = super().__repr__()
        components = original_str.split('\n')
        block_type_str = f'  (block_type): {self.block_type_str}'
        components.insert(-1, block_type_str)
        new_str = '\n'.join(components)
        return new_str
