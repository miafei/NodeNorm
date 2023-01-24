from config import ex
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl

from copy import deepcopy as dcopy
from os.path import join as opjoin
from utils import load_data, save_source, check_before_pkl, resplit

from initializer import init_logger, init_seed, init_optimizer
from models import GNN
from worker import train, test
import numpy as np



@ex.automain
def main(_run, _config, _log):
    '''
    _config: dictionary; its keys and values are the variables setting in the cfg function
    _run: run object defined by Sacred, can be used to record hashable values and get some information, e.g. run id, for a run
    _log: logger object provided by Sacred, but is not very flexible, we can define loggers by oureselves
    '''
    
    config = dcopy(_config)  # We need this step because Sacred does not allow us to change _config object
                        # But sometimes we need to add some key-value pairs to config
    torch.cuda.set_device(config['gpu_id'])

    save_source(_run)  # Source code are saved by running this line
    init_seed(config['seed'])
    logger = init_logger(log_root=_run.observers[0].dir, file_name='log.txt')

    output_folder_path = opjoin(_run.observers[0].dir, config['path']['output_folder_name'])
    os.makedirs(output_folder_path, exist_ok=True)

    best_acc_list = []
    last_acc_list = []
    train_best_list = []
    train_last_list = []

    best_epoch = []

    data = load_data(config=config)
    split_iterator = range(config['data']['random_split']['num_splits']) \
                     if config['data']['random_split']['use'] \
                    else range(1)

    config['adj'] = data[0]

    for i in split_iterator:
        output_folder = opjoin(output_folder_path, str(i))
        os.makedirs(output_folder, exist_ok=True)

        if config['data']['random_split']['use']:
            data = resplit(dataset=config['data']['dataset'],
                           data=data,
                           full_sup=config['data']['full_sup'],
                           num_classes=torch.unique(data[2]).shape[0],
                           num_nodes=data[1].shape[0],
                           num_per_class=config['data']['label_per_class'],
                           )
            print(torch.sum(data[3]))

        model = GNN(config=config)

        if i == 0:
            logger.info(model)

        if config['use_gpu']:
            model.cuda()
            data = [each.cuda() if hasattr(each, 'cuda') else each for each in data] 
        
        optimizer = init_optimizer(params=model.parameters(),
                                   optim_type=config['optim']['type'],
                                   lr=config['optim']['lr'],
                                   weight_decay=config['optim']['weight_decay'],
                                   momentum=config['optim']['momemtum'])

        criterion = nn.NLLLoss()

        best_model_path = opjoin(output_folder, 'best_model.pth')
        last_model_path = opjoin(output_folder, 'last_model.pth')
        best_dict_path = opjoin(output_folder, 'best_pred_dict.pkl')
        last_dict_path = opjoin(output_folder, 'last_pred_dict.pkl')
        losses_curve_path = opjoin(output_folder, 'losses.pkl')
        accs_curve_path = opjoin(output_folder, 'accs.pkl')
        best_state_path = opjoin(output_folder, 'best_state.pkl')
        grads_path = opjoin(output_folder, 'grads.pkl')

        best_pred_dict, last_pred_dict, train_losses, train_accs, \
        val_losses, val_accs, best_state, grads, model_state = train(best_model_path,
                                                       last_model_path,
                                                       config, 
                                                       criterion, 
                                                       data, 
                                                       logger, 
                                                       model, 
                                                       optimizer
                                                       )
        last_model_state, best_model_state = model_state

        losses_dict = {
            'train': train_losses,
            'val': val_losses
            }

        accs_dict = {
            'train': train_accs,
            'val': val_accs
            }
        logger.info(f'split_seed: {i: 04d}')
        logger.info(f'Test set results on the last model:')
        last_pred_dict = test(criterion,
                              data, 
                              last_model_path,
                              last_pred_dict, 
                              logger, 
                              model,
                              last_model_state,
                              )

        logger.info(f'Test set results on the best model:')
        if config['fastmode']:
            best_pred_dict = last_pred_dict
        else:
            best_pred_dict = test(criterion,
                                  data,
                                  best_model_path,
                                  best_pred_dict,
                                  logger,
                                  model,
                                  best_model_state,
                                  )

        logger.info('\n')

        check_before_pkl(best_pred_dict)
        with open(best_dict_path, 'wb') as f:
            pkl.dump(best_pred_dict, f)

        check_before_pkl(last_pred_dict)
        with open(last_dict_path, 'wb') as f:
            pkl.dump(last_pred_dict, f)
        
        check_before_pkl(losses_dict)
        with open(losses_curve_path, 'wb') as f:
            pkl.dump(losses_dict, f)
        
        check_before_pkl(accs_dict)
        with open(accs_curve_path, 'wb') as f:
            pkl.dump(accs_dict, f)

        check_before_pkl(best_state)
        with open(best_state_path, 'wb') as f:
            pkl.dump(best_state, f)

        check_before_pkl(grads)
        with open(grads_path, 'wb') as f:
            pkl.dump(grads, f)

        best_acc_list.append(best_pred_dict['test acc'].item())
        last_acc_list.append(last_pred_dict['test acc'].item())
        train_best_list.append(best_state['train acc'].item())
        train_last_list.append(train_accs[-1].item())


    logger.info('********************* STATISTICS *********************')
    np.set_printoptions(precision=4, suppress=True)
    logger.info(f"\n"
                f"Best test acc: {best_acc_list}\n"
                f"Mean: {np.mean(best_acc_list)}\t"
                f"Std: {np.std(best_acc_list)}\n"
                )


