import torch.nn as nn
import torch.nn.functional as F
from layers import GNNBasicBlock


class GNN(nn.Module):
    def __init__(self, config):
        super(GNN, self).__init__()
        self.arch = config['arch']
        self.structure = nn.Sequential()
        self.aggregator = {}
        self.num_classes = config['data']['num_classes']
        self.normalization = config['arch']['norm']
        self.construct_from_blocks()
        self.layer_names = [each[0] for each in list(self.structure.named_children())]
        self.batch_size = config['optim']['batch_size']

        assert not self.layer_names[-1].startswith('relu')

    def construct_from_blocks(self):
        for l, block in enumerate(self.arch['structure']):
            layer_type = block[0]

            if layer_type != 'dropout':
                block_type, hyperparams = block[1], block[2]
            else:
                hyperparams = block[1]

            if layer_type in ['gcn', 'gcn_res', 'sage', 'sage_res']:
                in_channels, out_channels, activation, bias = \
                    hyperparams[0], hyperparams[1], hyperparams[2], hyperparams[3],

                self.structure.add_module(
                    f'{l}_{layer_type}', GNNBasicBlock(layer_type=layer_type,
                                                       block_type=block_type,
                                                       activation=activation,
                                                       normalization=self.normalization,
                                                       in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       bias=bias,

                                                       )
                )

            elif layer_type in ['gat', 'gat_res']:
                in_channels, out_channels, num_heads, activation, feat_drop, attn_drop = \
                    hyperparams[0], hyperparams[1], hyperparams[2], hyperparams[3], hyperparams[4], hyperparams[5]

                self.structure.add_module(
                    f'{l}_gat', GNNBasicBlock(layer_type='gat',
                                              block_type=block_type,
                                              activation=activation,
                                              normalization=self.normalization,
                                              in_channels=in_channels,
                                              out_channels=out_channels,
                                              num_heads=num_heads,
                                              feat_drop=feat_drop,
                                              attn_drop=attn_drop,
                                              )
                )
            elif layer_type == 'dropout':
                self.structure.add_module(
                     f'{l}_dropout', nn.Dropout(p=hyperparams[0])
                 )
            else:
                raise NotImplementedError


    def forward(self, x, graph, labels=None, idx_train=None, proto=None):
        node_emb = []
        for l, block in enumerate(self.structure):
            name = self.layer_names[l].split('_')[1]
            if name not in ['dropout', ]:
                x = block(graph, x)
                node_emb.append(x)
            else:
                x = block(x)
        return F.log_softmax(x, dim=1), node_emb


