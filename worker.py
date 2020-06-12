import torch
from torch.nn.utils import clip_grad_norm_
from utils import accuracy, adjust_learning_rate
import numpy as np
from tqdm import tqdm
from copy import deepcopy as dcopy


def get_l1_regularization(model):
    regularization_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            regularization_loss += torch.sum(abs(param))
    return regularization_loss


def get_grads(module, param):
    sum_numel = np.zeros(2)
    if hasattr(module, param) \
       and getattr(module, param) is not None \
       and getattr(module, param).requires_grad:
        sum_numel[0] = getattr(module, param).grad.abs().sum().detach().cpu().numpy()
        sum_numel[1] = getattr(module, param).grad.numel()
    else:
        for each in module.named_children():
            name = each[0]
            sub_module = getattr(module, name)
            sum_numel += get_grads(sub_module, param)
    return sum_numel


def record_grads(w_grads, b_grads, model, epoch_idx, bias):
    grad_count = 0
    for layer in model.structure:
        layer_name = layer._get_name()
        grad_sum_numel = get_grads(layer, 'weight')
        if grad_sum_numel[1] != 0 and (layer_name.find('Batch') == -1):
            w_grad = grad_sum_numel[0] / grad_sum_numel[1]
            w_grads[grad_count][epoch_idx] = w_grad
            
            if bias:
                grad_sum_numel = get_grads(layer, 'bias')
                assert grad_sum_numel[1] != 1
                b_grad = grad_sum_numel[0] / grad_sum_numel[1]
                b_grads[grad_count][epoch_idx] = b_grad
            grad_count += 1


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    return total_norm


def get_clip_value(gn_container, ref_window):
    if ref_window == -1:
        value = sum(gn_container) / len(gn_container)
    else:
        assert len(gn_container) >= ref_window
        value = sum(gn_container[-ref_window : ]) / ref_window
    return value


def test(criterion, data, model_path, pred_dict, logger, model, model_state):
    features, labels, idx_train, idx_val, idx_test = data[1], data[2], data[3], data[4], data[5]

    ## note that also need to save predicts of train / val with last model
    with torch.no_grad():
        #model.load_state_dict(torch.load(model_path))
        model.load_state_dict(model_state)
        model.eval()
        output, node_emb = model(features, data[0])

        loss_test = criterion(output[idx_test], labels[idx_test])
        acc_test, correct_test = accuracy(output[idx_test], labels[idx_test])

        logger.info(f'loss= {loss_test.item(): .4f}\t'
                    f'accuracy= {acc_test.item(): .4f}')

        pred_dict['test score'] = output[idx_test].cpu().detach().numpy()
        pred_dict['test correct'] = correct_test.cpu().detach().numpy()
        pred_dict['test acc'] = acc_test.cpu().detach().numpy()
        
    return pred_dict


def set_pred_dict(correct_train, correct_val, idx_train, idx_val, output):
    pred_dict = {}
    pred_dict['train score'] = output[idx_train].cpu().detach().numpy()
    pred_dict['train correct'] = correct_train.cpu().detach().numpy()
    pred_dict['val score'] = output[idx_val].cpu().detach().numpy()
    pred_dict['val correct'] = correct_val.cpu().detach().numpy()
    return pred_dict


def train(best_model_path, last_model_path, config, criterion, data, logger, model, optimizer):
    best_state = {
        'val acc': 0,
        'train acc':0,
        'epoch': 0
        }
    features, labels, idx_train, idx_val, idx_test = data[1], data[2], data[3], data[4], data[5]

    train_losses = np.zeros(config['optim']['epoch'], dtype=np.float32)
    val_losses = np.zeros(config['optim']['epoch'], dtype=np.float32)
    train_accs = np.zeros(config['optim']['epoch'], dtype=np.float32)
    val_accs = np.zeros(config['optim']['epoch'], dtype=np.float32)

    ## gradient
    filter_list = ['relu', 'dropout', 'bn', 'swish', 'lkrelu', 'agg_r', 'agg_m']
    learn_layers = [each for each in config['arch']['structure'] if not each[0] in filter_list]
    w_grads = np.zeros((len(learn_layers), config['optim']['epoch']), dtype=np.float32)
    b_grads = np.zeros((len(learn_layers), config['optim']['epoch']), dtype=np.float32)
    gn_container = []

    best_model = None

    for epoch in tqdm(range(config['optim']['epoch']), bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
        adjust_learning_rate(optimizer=optimizer,
                             epoch=epoch,
                             lr_down_epoch_list=config['optim']['down_list'],
                             logger=logger)
        model.train()
        adj = data[0]
        output, node_emb = model(x=features,
                                 graph=adj,
                                 labels=labels,
                                 idx_train=idx_train,
                                 )
        loss_train = criterion(output[idx_train], labels[idx_train])
        acc_train, correct_train = accuracy(output[idx_train], labels[idx_train])
        optimizer.zero_grad()
        l1_regularization_loss = get_l1_regularization(model)
        loss_train += config['optim']['l1_weight'] * l1_regularization_loss
        loss_train.backward()
        if config['optim']['gnclip']['use'] and epoch > max(config['optim']['gnclip']['ref_window'], 0):
            clip_value = get_clip_value(gn_container, config['optim']['gnclip']['ref_window'])
            clip_grad_norm_(model.parameters(), clip_value, norm_type=config['optim']['gnclip']['norm'])

        clipped_grad_norm = get_grad_norm(model.parameters(), norm_type=config['optim']['gnclip']['norm'])
        gn_container.append(clipped_grad_norm)
        if config['record_grad']:
            record_grads(w_grads=w_grads,
                         b_grads=b_grads,
                         model=model,
                         epoch_idx=epoch,
                         bias=config['arch']['bias'])
        optimizer.step()
        train_losses[epoch] = loss_train.item()
        train_accs[epoch] = acc_train.item()

        if not config['fastmode'] or (config['fastmode'] and epoch == config['optim']['epoch'] - 1):
            model.eval()
            with torch.no_grad():
                output, _ = model(features, adj)

            loss_val = criterion(output[idx_val], labels[idx_val])
            acc_val, correct_val = accuracy(output[idx_val], labels[idx_val])

            val_losses[epoch] = loss_val.item()
            val_accs[epoch] = acc_val.item()

            if acc_val > best_state['val acc']:
                best_state['val acc'] = acc_val
                best_state['epoch'] = epoch
                best_state['train acc'] = acc_train
                best_model = dcopy(model.state_dict())
                best_pred_dict = set_pred_dict(correct_train, correct_val, idx_train, idx_val, output)

    last_pred_dict = set_pred_dict(correct_train, correct_val, idx_train, idx_val, output)

    if config['save_model']:
        torch.save(model.state_dict(), last_model_path)
        torch.save(best_model, best_model_path)

    best_state['val acc'] = best_state['val acc'].cpu().detach().numpy()
    best_state['train acc'] = best_state['train acc'].cpu().detach().numpy()

    return best_pred_dict, last_pred_dict, train_losses, train_accs, val_losses, val_accs, best_state, (w_grads, b_grads), (model.state_dict(), best_model)


