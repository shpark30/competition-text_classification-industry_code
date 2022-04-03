import os
import logging
from pathlib import Path
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

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

def num2code(num, digits=0):
    """ int타입의 데이터를 일정한 자릿수(digits)의 코드값으로 변환 """
    num = str(num)
    code = '0'*(digits-len(num)) + num
    return code

def save_performance_graph(summary_path, save_path):
    data = pd.read_csv(summary_path, encoding='cp949')
    best_acc_epoch = data.loc[data['valid loss'].tolist().index(data['valid loss'].min()), 'epoch']
    best_loss_epoch = data.loc[data['accuracy'].tolist().index(data['accuracy'].max()), 'epoch']

    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(2,1,1)
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

    ax2 = fig.add_subplot(2,1,2)
    ax2.set_title('Model Performance', size=25)
    ax2.plot(data['epoch'], data['accuracy'], label='accuracy')
    ax2.plot(data['epoch'], data['precision'], label='precision')
    ax2.plot(data['epoch'], data['recall'], label='recall')
    ax2.plot(data['epoch'], data['f1score'], label='f1score')
    ax2.axvline(best_acc_epoch, color='gray', linestyle='--', linewidth=2,
               label=f'best_acc_epoch: {best_acc_epoch}')
    ax2.axvline(best_loss_epoch, color='lightgray', linestyle='--', linewidth=2,
               label=f'best_loss_epoch: {best_loss_epoch}')
    ax2.set_xlabel('epoch', size=20)
    ax2.set_ylabel('loss', size=20)
    ax2.tick_params (axis = 'x', labelsize =15)
    ax2.tick_params (axis = 'y', labelsize =15)
    ax2.legend(loc='best', fontsize=20, frameon=True, shadow=True)

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