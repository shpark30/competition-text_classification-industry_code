import os
import logging
from pathlib import Path
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import statistics
from collections import defaultdict

def vote(tensor, dim=1):
    max_idx = torch.argmax(tensor, dim, keepdim=True)
    device = max_idx.get_device()
    if device == -1:
        device = 'cpu'
    one_hot = torch.FloatTensor(tensor.shape)
    one_hot = one_hot.to(device)
    one_hot.zero_()
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot

def get_optimizer(optimizer_type, model, lr, betas, weight_decay, eps=1e-08, amsgrad=False):
        if optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        elif optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        elif optimizer_type == 'RAdam':
            optimizer = torch.optim.RAdam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        else:
            print(f'{optimizer_type} is not available')
            raise 
        return optimizer

def create_logger(log_path, name='torch', file_name='train.log', fmt='%(asctime)s | %(message)s'):
    logger = logging.getLogger(name)
    # Check handler exists
    if len(logger.handlers) > 0:
        return logger # Logger already exists
    formatter = logging.Formatter(fmt)
    logger.setLevel(logging.INFO)
    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)
    logger.addHandler(stream_hander)
    file_handler = logging.FileHandler(os.path.join(log_path, file_name))
    logger.addHandler(file_handler)
    file_handler.setFormatter(formatter)
    return logger

class Evaluator():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.loss_sum = 0
        self.loss = 0 # mean
        self.acc = 0
        self.macro_pc = 0
        self.macro_rc = 0
        self.macro_f1 = 0
        self.predictions=[]
        self.labels=[]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, label, loss=None):
        if loss is not None:
            self.loss_sum += loss
        self.predictions += pred
        self.labels += label
        batch_confusion_matrix = confusion_matrix(label, pred, labels=list(range(self.num_classes)))
        self.confusion_matrix += batch_confusion_matrix

    def compute(self):
        self.loss = self.loss_sum / len(self.predictions)
        self.class_scores = defaultdict(list)

        self.acc = np.diagonal(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        for c in range(self.num_classes):
            num_pred = self.confusion_matrix[:, c].sum()
            num_true = self.confusion_matrix[c].sum()
            TP = self.confusion_matrix[c, c]
            PC = TP/num_pred if num_pred != 0 else 0 # TP / (TP+FP)
            RC = TP/num_true if num_true != 0 else 0  # TP / (TP+FN)
            F1 = 2 * PC * RC / (PC + RC) if (PC + RC) != 0 else 0 # (2 * PC * RC) / (PC + RC)
            self.class_scores['class_id'].append(c)
            self.class_scores['precision'].append(PC)
            self.class_scores['recall'].append(RC)
            self.class_scores['f1score'].append(F1)
        self.macro_pc = statistics.mean(self.class_scores['precision'])
        self.macro_rc = statistics.mean(self.class_scores['recall'])
        self.macro_f1 = statistics.mean(self.class_scores['f1score'])

def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
        
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def save_performance_graph(summary_path, save_path):
    data = pd.read_csv(summary_path, encoding='cp949')
    best_acc_epoch = data.loc[data['valid loss'].tolist().index(data['valid loss'].min()), 'epoch']
    best_loss_epoch = data.loc[data['valid acc'].tolist().index(data['valid acc'].max()), 'epoch']

    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(3,1,1)
    ax1.set_title('Train & Valid Loss Graph', fontsize=25)
    ax1.plot(data['epoch'], data['train loss'], label='train')
    ax1.plot(data['epoch'], data['valid loss'], label='valid')
    ax1.axvline(best_acc_epoch, color='gray', linestyle='--', linewidth=2,
               label=f'best_acc_epoch: {best_acc_epoch}')
    ax1.axvline(best_loss_epoch, color='lightgray', linestyle='--', linewidth=2,
               label=f'best_loss_epoch: {best_loss_epoch}')
    ax1.set_xlabel('epoch', size=20)
    ax1.set_ylabel('loss', size=20)
    ax1.tick_params (axis = 'x', labelsize =15)
    ax1.tick_params (axis = 'y', labelsize =15)
    ax1.legend(loc='best', fontsize=20, frameon=True, shadow=True)
    
    ax2 = fig.add_subplot(3,1,2)
    ax2.set_title('Train Performance', size=25)
    ax2.plot(data['epoch'], data['train acc'], label='accuracy')
    ax2.plot(data['epoch'], data['train pc'], label='precision')
    ax2.plot(data['epoch'], data['train rc'], label='recall')
    ax2.plot(data['epoch'], data['train f1'], label='f1score')
    ax2.axvline(best_acc_epoch, color='gray', linestyle='--', linewidth=2,
               label=f'best_acc_epoch: {best_acc_epoch}')
    ax2.axvline(best_loss_epoch, color='lightgray', linestyle='--', linewidth=2,
               label=f'best_loss_epoch: {best_loss_epoch}')
    ax2.set_xlabel('epoch', size=20)
    ax2.set_ylabel('loss', size=20)
    ax2.tick_params (axis = 'x', labelsize =15)
    ax2.tick_params (axis = 'y', labelsize =15)
    ax2.legend(loc='best', fontsize=20, frameon=True, shadow=True)
    
    ax3 = fig.add_subplot(3,1,3)
    ax3.set_title('Valid Performance', size=25)
    ax3.plot(data['epoch'], data['valid acc'], label='accuracy')
    ax3.plot(data['epoch'], data['valid pc'], label='precision')
    ax3.plot(data['epoch'], data['valid rc'], label='recall')
    ax3.plot(data['epoch'], data['valid f1'], label='f1score')
    ax3.axvline(best_acc_epoch, color='gray', linestyle='--', linewidth=2,
               label=f'best_acc_epoch: {best_acc_epoch}')
    ax3.axvline(best_loss_epoch, color='lightgray', linestyle='--', linewidth=2,
               label=f'best_loss_epoch: {best_loss_epoch}')
    ax3.set_xlabel('epoch', size=20)
    ax3.set_ylabel('loss', size=20)
    ax3.tick_params (axis = 'x', labelsize =15)
    ax3.tick_params (axis = 'y', labelsize =15)
    ax3.legend(loc='best', fontsize=20, frameon=True, shadow=True)

    plt.savefig(save_path, dpi=50)
    plt.close(fig)
        
        
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(
        description='Training Disease Recognition in Pet CT')
    # parser.add_argument('root', metavar='DIR',
    #                     help='path to data')
    parser.add_argument('--num_test', default=100000, type=int,
                        help='the number of test data')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N',
                        help='mini-batch size (default: 16)'
                             '[kobert] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                             '[kogpt2] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                             '[kogpt3] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50')
    import json
    args=parser.parse_args()
    with open('./config.json', 'w') as outfile:
        json.dump(args.__dict__, outfile, indent=4)