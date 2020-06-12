import logging
import torch
import os
import random
import torch.optim as optim
import numpy as np


def init_logger(log_root, file_name):
    # if print on screen at the same time (when print is not used)
    logpath = os.path.join(log_root, file_name)
    logger = logging.getLogger('train')  
    logger.setLevel(logging.INFO)  

    fh = logging.FileHandler(logpath, mode='a') 
    fh.setLevel(logging.INFO)  
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s ')
    fh.setFormatter(formatter)

    logger.addHandler(fh)  
    return logger
    

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_optimizer(params, optim_type, lr, weight_decay, momentum):
    if optim_type == 'adam':
        optimizer = optim.Adam(params,
                               lr=lr,
                               weight_decay=weight_decay)
    elif optim_type == 'rmsprop':
        optimizer = optim.RMSprop(params,
                                  lr=lr,
                                  weight_decay=weight_decay)
    elif optim_type == 'sgd':
        optimizer = optim.SGD(params,
                              lr=lr,
                              weight_decay=weight_decay,
                              momentum=momentum)
    else:
        raise NotImplementedError 
    return optimizer
