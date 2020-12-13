# pylint: disable=unused-variable

'''
This is a config.py file of a standard sarcred format
We do not need to modify too much here
'''
import os
from os.path import join as opjoin
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds


sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIGS'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('GCN')

ex.captured_out_filter = apply_backspaces_and_linefeeds


source_root = './'
source_files = list(filter(lambda filename: filename.endswith('.py'), os.listdir('./')))
for source_file in source_files:
    ex.add_source_file(source_file)


@ex.config
def cfg():
    ex_name = ex.path
    ex_type = ''
    fastmode = False
    seed = 42

    use_gpu = True
    gpu_id = 1
    save_model = False


    features_classes = {
        'cora': (1433, 7),
        'citeseer': (3703, 6),
        'pubmed': (500, 3),
        'coauthorcs' : (6805, 15),
        }

    min_num_nodes_class_total_num_nodes = {
        'pyg':{
            'cora': (180, 2708),
            'citeseer': (249, 3327),
            'pubmed': (4103, 19717),
            'coauthorcs' : (118, 18333),
        },
    }


    data = {
        'dataset': 'cora',
        'implement':'pyg',
        'add_slflp': True,
        'feature_prenorm': True, # whether to normalize raw features, applicable to both dgl and pyg
        'binarize': False,  # not in use at present
                            # whether to binarize the raw features. Note that the raw features of cora_full, pubmed and coauthor-cs are not binary, some works binarize them before row-norm them
        'all_datasets': ['cora', 'citeseer', 'pubmed', 'coauthorcs'],
        'random_split':{
                        'use': True, # if false, only run with the given split
                        'num_splits': 50,
                       },
        'label_per_class': 20,
        'full_sup': False, # if true, use fully-supervised setting
    }

    data['num_features'], data['num_classes'] = features_classes.get(data['dataset'])
    optim = {  
        'type': 'adam', #'adam', 'rmsprop', 'sgd'
        'epoch': 400,
        'lr': 5e-3,
        'weight_decay': 0.001,
        'momemtum': 0.9, # momentum is only applicable for sgd
        'down_list': [],
        'l1_weight': 1e-3,
        'batch_size': 500,
        'num_workers':0,
        'fanout':15,
        'gnclip':{
                  'use': True,
                  'ref_window': 50, # positive integer for fixed window size, -1 for the all previous epoch
                  'norm': 1, # 2 or 1
                 },
    }

    arch = {
        'implement': 'dgl',
        'num_hiddens': 64,
        'num_layers': 2,
        'norm': 'node_n',
        'weight_sharing': False,
        'activation': ('relu', False),
        'bias': True,
        'bn': False,
        'nn': True,
        'dropout': {
            'use': True,
            'p': 0.8,  # dropping probability
            'layer': [1,-1],
        },
        'layer_type': 'gcn_res',  # ['sage_res', 'gat_res', 'gcn', 'gat', 'sage']
        'block_type': 'n_a_r',
        'gat': {
            'num_heads': 8,
            'dropout': {
                'use': True,
                'layer': [1, -2],
                'feat_p': 0.6,  # dropping probability
                'attn_p': 0.6,  # dropping probability
            },
        },
    }

    record_grad = False

    path = {
        'log_dir': './runs',  # Specify the root folder to record all experiments and runs
        'dataroot': '../data',  # Sepcify the root of dataset, only for pyg, because dgl is its built-in data path, which can only be modified in its source code
        'output_folder_name': 'output'  # Specify the name of the folder which is used to save all outputs in a run, e.g. models, .pkl files, etc
    }

    cmt_items=["num_layers", "blk_type", "dpl", "dpp", "l1", "wd"]

# after-config-pre-main modifications should be moved to this hook function
@ex.config_hook
def add_observer(config, command_name, logger):
    '''
    config: dictionary; its keys and values are the variables setting in 3the cfg function
    typically we do not need to use command_name and logger here
    '''
    sanity_check(config)
    post_processing(config)
    os.makedirs(config['path']['log_dir'], exist_ok=True) 
    exp_cmt = get_comment(config)
    observer = FileStorageObserver.create(opjoin(config['path']['log_dir'], config['ex_name'], config['data']['dataset'], config['ex_type'], exp_cmt)) 
    ex.observers.append(observer)
    return config


def post_processing(config):
    config['lr_down_flag'] = bool(config['optim']['down_list'] != [])
    config['arch']['structure'] = get_structure(config)


def get_structure(config):
    if config['arch']['layer_type'] in ['gcn', 'gcn_res', 'fc', 'fc_res', 'agg_m', 'sage', 'sage_res']:
        structure = get_general_blocks(config)

    elif config['arch']['layer_type'] in ['gat', 'gat_res']:
        structure = get_gat_blocks(config)

    elif config['arch']['layer_type'] in ['agg_r']:
        structure = get_agg_r_blocks(config)

    else:
        raise NotImplementedError(f"{config['arch']['layer_type']} has not been implemented yet")

    ## dropout
    if config['arch']['dropout']['use'] and config['arch']['layer_type'] not in ['gat', 'gat_res']:
        # dropout of GAT-related model is not set here, but in get_gat_blocks
        dropout_idx = [x + config['arch']['num_layers'] + 1 if x < 0 else x for x in config['arch']['dropout']['layer']]
        dropout_idx = sorted(set(dropout_idx))
        for i, idx in enumerate(dropout_idx):
            structure.insert(idx + i, ('dropout', [config['arch']['dropout']['p'], ]))

    return structure


def get_general_blocks(config):
    if config['arch']['layer_type'] in ['sage', 'sage_res']:
        in_layer_type = 'sage'
        out_layer_type = 'sage'
    else:
        in_layer_type = 'gcn' if config['arch']['layer_type'] not in ['fc', 'fc_res']  else 'fc'
        out_layer_type = in_layer_type

    if config['arch']['bn'] or config['arch']['nn']:
        block_type_ls = config['arch']['block_type'].split('_')
        if 'r' in block_type_ls:
            block_type_ls.remove('r')
        in_block_type = '_'.join(block_type_ls)
    else:
        in_block_type = 'v'

    structure = [
        (in_layer_type, in_block_type, [config['data']['num_features'], config['arch']['num_hiddens'],\
        config['arch']['activation'], config['arch']['bias']]),
        (out_layer_type, 'v', [config['arch']['num_hiddens'], config['data']['num_classes'],\
        ('no', None), config['arch']['bias']]),
    ]

    if config['arch']['layer_type'] != 'agg_m': #### to add here
        hidden_block = (
            config['arch']['layer_type'], config['arch']['block_type'], [config['arch']['num_hiddens'], \
            config['arch']['num_hiddens'], config['arch']['activation'], config['arch']['bias']],
        )
    else: 
        hidden_block = (
            config['arch']['layer_type'], 'v', [config['arch']['num_hiddens'], \
            config['arch']['num_hiddens'], ('no', None), False],
        )

    for i in range(config['arch']['num_layers'] - 2):
        structure.insert(-1, hidden_block)

    return structure


def get_gat_blocks(config):
    if config['arch']['bn'] or config['arch']['nn']:
        block_type_ls = config['arch']['block_type'].split('_')
        if 'r' in block_type_ls:
            block_type_ls.remove('r')
        in_block_type = '_'.join(block_type_ls)
    else:
        in_block_type = 'v'

    in_block_dropout = {}
    in_block_dropout['feat_p'] = config['arch']['gat']['dropout']['feat_p'] if config['arch']['gat']['dropout']['use'] \
                                 and 1 in config['arch']['gat']['dropout']['layer'] else 0
    in_block_dropout['attn_p'] = config['arch']['gat']['dropout']['attn_p'] if config['arch']['gat']['dropout']['use'] \
                                 and 1 in config['arch']['gat']['dropout']['layer'] else 0

    out_block_dropout = {}
    out_block_dropout['feat_p'] = config['arch']['gat']['dropout']['feat_p'] if config['arch']['gat']['dropout']['use'] \
                                 and -1 in config['arch']['gat']['dropout']['layer'] else 0
    out_block_dropout['attn_p'] = config['arch']['gat']['dropout']['attn_p'] if config['arch']['gat']['dropout']['use'] \
                                 and -1 in config['arch']['gat']['dropout']['layer'] else 0

    out_num_heads = 1 # following the GAT paper

    structure = [
        ('gat', in_block_type, [config['data']['num_features'], config['arch']['num_hiddens'], config['arch']['gat']['num_heads'], \
         config['arch']['activation'], in_block_dropout['feat_p'], in_block_dropout['attn_p']]),
        ('gat', 'v', [config['arch']['num_hiddens'], config['data']['num_classes'], out_num_heads, \
         ('no', None), out_block_dropout['feat_p'], out_block_dropout['attn_p']]),
    ]

    hidden_block_dropout = {}
    for l in range(config['arch']['num_layers'] - 2):
        if config['arch']['gat']['dropout']['use'] and \
           ((l + 2) in config['arch']['gat']['dropout']['layer'] or \
           (l + 1 - config['arch']['num_layers']) in config['arch']['gat']['dropout']['layer']):
           hidden_block_dropout['feat_p'] = config['arch']['gat']['dropout']['feat_p']
           hidden_block_dropout['attn_p'] = config['arch']['gat']['dropout']['attn_p']
        else:
           hidden_block_dropout['feat_p'] = 0
           hidden_block_dropout['attn_p'] = 0

        hidden_block = (
        config['arch']['layer_type'], config['arch']['block_type'], [config['arch']['num_hiddens'], config['arch']['num_hiddens'], config['arch']['gat']['num_heads'], \
        config['arch']['activation'], hidden_block_dropout['feat_p'], hidden_block_dropout['attn_p']],
        )

        structure.insert(-1, hidden_block)
    
    return structure


def get_agg_r_blocks(config):
    structure = [
        ('gcn', 'v', [config['data']['num_features'], config['arch']['num_hiddens'],\
        config['arch']['activation'], config['arch']['bias']]),
        ('gcn', 'v', [config['arch']['num_hiddens'], config['data']['num_classes'],\
        ('no', None), config['arch']['bias']]),
    ]

    agg_block = (
        config['arch']['layer_type'], 'v', [config['data']['num_classes'], \
        config['data']['num_classes'], ('no', None), False],
    )

    for i in range(config['arch']['num_layers'] - 2):
        structure.append(agg_block)

    return structure


def sanity_check(config):
    assert config['data']['dataset'] in config['data']['all_datasets']
    assert config['data']['implement'] in ['dgl', 'pyg']
    assert config['arch']['implement'] in ['dgl', 'pyg']
    assert config['arch']['activation'][0] in ['relu', 'lkrelu', 'elu']
    #assert config['arch']['unit'] in ['layer', 'block']
    assert config['optim']['type'] in ['adam', 'rmsprop', 'sgd']

    if not config['data']['full_sup']:
        if config['data']['dataset'] in ['cora', 'citeseer', 'pubmed']:
            assert config['data']['label_per_class'] <= config['min_num_nodes_class_total_num_nodes']\
                .get(config['data']['implement']).get(config['data']['dataset'])[0]
            assert config['data']['label_per_class'] * config['features_classes'].get(config['data']['dataset'])[1] \
                <= config['min_num_nodes_class_total_num_nodes']\
                .get(config['data']['implement']).get(config['data']['dataset'])[1] - 1500 # 1500 for val and test

    if config['data']['random_split']['use']:
        assert config['data']['random_split']['num_splits'] >= 1

    if config['optim']['gnclip']['use']:
        assert config['optim']['gnclip']['ref_window'] == -1 or config['optim']['gnclip']['ref_window'] > 0
        assert config['optim']['gnclip']['norm'] in [1, 2]

    if config['arch']['dropout']['use']:
        assert config['arch']['dropout']['p'] >= 0
        assert config['arch']['dropout']['p'] <= 1
        assert all([0 < abs(x) <= config['arch']['num_layers'] for x in config['arch']['dropout']['layer']])

    if 'r' in config['arch']['block_type'].split('_'):
        assert 'res' in config['arch']['layer_type'].split('_'), \
            'require res in block but res is not switched on'
    if config['arch']['layer_type'] in ['agg_r', 'agg_m']:
        assert config['arch']['block_type'] == 'v'
        assert not config['archg']['bn']
    if config['arch']['bn']:
        assert 'b' in config['arch']['block_type'].split('_')
    if config['arch']['nn']:
        assert 'n' in config['arch']['block_type'].split('_')
    if config['arch']['layer_type'] == 'gat':
        assert config['arch']['num_hiddens'] % config['arch']['gat']['num_heads'] == 0

def get_comment(config):
    allowed_cmt = ['num_layers', 'type', 'hd', 'wd', 'lr', 'lrd', 'bn', 'dp', 'dpl', 'dpp',\
                   'sw_r', 'sw_plr', 'sw_exloss', 'sw_lbda', 'sw_divtype', 'l1',\
                   'act', 'gnclip', 'cwndw', 'epc', 'blk_type', 'full', 'ftrans',\
                   'ent', 'sw_prt', 'nn', 'cnum', 'lbr']

    cmt_items = config['cmt_items']

    assert type(cmt_items) == list
    assert set(allowed_cmt).intersection(set(cmt_items)) == set(cmt_items)

    exp_cmt = []
    if 'num_layers' in cmt_items:
        exp_cmt.append(f"layer_{config['arch']['num_layers']}", )
    if 'type' in cmt_items:
        exp_cmt.append(f"type_{config['arch']['layer_type']}", )
    if 'hd' in cmt_items:
        exp_cmt.append(f"hidden_{config['arch']['num_hiddens']}", )
    if 'lr' in cmt_items:
        exp_cmt.append(f"lr_{config['optim']['lr']:.1e}", )
    if 'lrd' in cmt_items:
        exp_cmt.append(f"lrd_{config['lr_down_flag']}", )
    if 'bn' in cmt_items:
        exp_cmt.append(f"bn_{config['arch']['bn']}", )
    if 'dp' in cmt_items:
        exp_cmt.append(f"dp_{config['arch']['dropout']['use']}", )
    if 'dpl' in cmt_items:
        exp_cmt.append(f"dpl_{config['arch']['dropout']['layer']}", )
    if 'dpp' in cmt_items:
        exp_cmt.append(f"dpp_{config['arch']['dropout']['p']}", )
    if 'l1' in cmt_items:
        exp_cmt.append(f"l1_{config['optim']['l1_weight']}")
    if 'act' in cmt_items:
        exp_cmt.append(f"act_{config['arch']['activation'][0]}")
    if 'gnclip' in cmt_items:
        exp_cmt.append(f"gnclip_{config['optim']['gnclip']['use']}")
    if 'cwndw' in cmt_items:
        exp_cmt.append(f"gnclip_wndw_{config['optim']['gnclip']['ref_window']}")
    if 'epc' in cmt_items:
        exp_cmt.append(f"epc_{config['optim']['epoch']}")
    if 'blk_type' in cmt_items:
        exp_cmt.append(f"blk_{config['arch']['block_type']}")
    if 'full' in cmt_items:
        exp_cmt.append(f"full_{config['data']['full_sup']}")
    if 'nn' in cmt_items:
        exp_cmt.append(f"nn_{config['arch']['nn']}", )
    if 'wd' in cmt_items:
        exp_cmt.append(f"wd_{config['optim']['weight_decay']}", )
    if 'lbr' in cmt_items:
        exp_cmt.append(f"lbr_{config['data']['label_per_class']}", )
    exp_cmt = '_'.join(exp_cmt)

    return exp_cmt


