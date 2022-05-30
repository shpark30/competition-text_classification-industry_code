import os
import logging
import json
import statistics
import csv
from typing import Optional, Union, List
import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm
from collections import defaultdict, Counter
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pathlib
from pathlib import Path
import argparse

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import transformers
from transformers.optimization import get_scheduler

print('pytorch ver:', torch.__version__)
print('transformers ver:', transformers.__version__)
from loss import CrossEntropy, FocalCrossEntropy, label2target, get_loss
from utils import create_logger, create_directory, increment_path, save_performance_graph, Evaluator, get_optimizer
from dataset import preprocess
from load import load_backbone_tokenizer, load_model, load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(torch.cuda.device_count())))
os.environ["CUDA_LAUNCH_BLOCKING"] = ",".join(map(str, range(torch.cuda.device_count())))

def main(args):
    global best_acc
    global best_loss
    
    # load data
    try:
        data = pd.read_csv(args.root, sep='|', encoding='cp949')
    except:
        data = pd.read_csv(args.root, sep='|', encoding='utf-8')
        
    # preprocess
    train_data, test_data, cat2id, id2cat = preprocess(data, num_test=args.num_test, upsample=args.upsample, minimum=args.minimum, target=args.target, seed=args.seed)
    
    # load backbone, tokenizer
    backbone, tokenizer = load_backbone_tokenizer(args.model, max_len=args.max_len)
    
    # load model
    num_classes = len(cat2id.keys())
    model = load_model(args.model, backbone, num_classes, num_layers=args.n_layers, dr_rate=args.dr_rate, bias=args.bias_off, batchnorm=args.batchnorm, layernorm=args.layernorm)
    model = model.to(args.device)
    
    # load dataset, dataloader
    train_set = load_dataset(args.model, train_data['text'].tolist(), train_data['label'].tolist(), tokenizer, max_len=50)
    test_set = load_dataset(args.model, test_data['text'].tolist(), test_data['label'].tolist(), tokenizer, max_len=50)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=True, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers,
                             shuffle=False, pin_memory=False)
    
    logger.info(f'# train data: {len(train_set)}')
    logger.info(f'# test  data: {len(test_set)}')
    
    # save cat2id, id2cat
    with open(args.project / 'cat2id.json', 'w', encoding='cp949') as f:
        json.dump(cat2id, f, indent=4)
    with open(args.project / 'id2cat.json', 'w', encoding='cp949') as f:
        json.dump(id2cat, f, indent=4)
    
    # optimizer
    betas=(args.beta1, args.beta2)
    optimizer = get_optimizer(optimizer_type=args.optimizer, model=model, lr=args.lr, betas=betas,
                              weight_decay=args.weight_decay, eps=args.epsilon, amsgrad=args.amsgrad)
    
    # lr-scheduler
    max_iter = len(train_loader) * args.epochs
    num_warmup_steps=int(max_iter * args.warmup_ratio)
    scheduler = get_scheduler(name=args.lr_scheduler, optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_iter)
                    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.device is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(args.resume, map_location=args.device)
            # build model
            model.classifier.load_state_dict(checkpoint['state_dict'])
            # build optimizer
            optimizer.load_state_dict(checkpoint['optimizer'])
            # build scheduler
            scheduler.load_state_dict(checkpoint['scheduler'])
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['acc']
            best_acc = checkpoint['loss']
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            print('start epoch: {}'.format(args.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # loss function
    criterion = get_loss(args.loss)
            
    # train
    logger.info(f"Start Training {args.model}")
    for epoch in range(args.start_epoch, args.epochs):
        # epoch train
        eval_train = train(model, train_loader, optimizer, criterion, scheduler, args.device)
    
        # epoch validation
        eval_valid = valid(model, test_loader, criterion, args.device)
        
        # logging scores
        logger.info(f'{args.model} Epoch {epoch} Result')
        logger.info(f'\ttrain | loss: {eval_train.loss}\tacc: {round(eval_train.acc, 6)}\tpc: {round(eval_train.macro_pc, 6)}\trc: {round(eval_train.macro_rc, 6)}\tf1: {round(eval_train.macro_f1, 6)}')
        logger.info(f'\tvalid | loss: {eval_valid.loss}\tacc: {round(eval_valid.acc, 6)}\tpc: {round(eval_valid.macro_pc, 6)}\trc: {round(eval_valid.macro_rc, 6)}\tf1: {round(eval_valid.macro_f1, 6)}')
        
        # save scores
        if epoch==args.start_epoch:
            # summary.csv
            with open(args.project / 'summary.csv', 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(['epoch', 'train loss', 'train acc', 'train pc', 'train rc', 'train f1',
                                          'valid loss', 'valid acc', 'valid pc', 'valid rc', 'valid f1'])
            # base frame for precisions, recalls and f1scores
            class_id = list(set(train_loader.dataset.label))
            num_train_data, num_valid_data = [0] * len(class_id), [0] * len(class_id)
            for c_id, n in dict(Counter(train_loader.dataset.label)).items():
                num_train_data[c_id] = n
            for c_id, n in dict(Counter(test_loader.dataset.label)).items():
                num_valid_data[c_id] = n
            history_train = defaultdict(lambda: pd.DataFrame({
                    'class_id': class_id,
                    'class': list(map(lambda x: ''.join(id2cat[x]), class_id)),
                    '# train data' : num_train_data,
                    '# valid data' : num_valid_data
                }))
            history_valid = defaultdict(lambda: pd.DataFrame({
                'class_id': class_id,
                'class': list(map(lambda x: ''.join(id2cat[x]), class_id)),
                '# train data' : num_train_data,
                '# valid data' : num_valid_data
            }))
            
        # add new line to summary.csv
        with open(args.project / 'summary.csv', 'a', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([epoch, eval_train.loss, eval_train.acc, eval_train.macro_pc, eval_train.macro_rc, eval_train.macro_f1,
                                    eval_valid.loss, eval_valid.acc, eval_valid.macro_pc, eval_valid.macro_rc, eval_valid.macro_f1])
            
        # add new column(epoch) to precision.csv, recall.csv and f1score.csv
        for metric, values in eval_train.class_scores.items():
            if metric != 'class_id':
                history_train[metric][f'epoch {epoch}'] = 0
                for c_id, v in zip(eval_valid.class_scores['class_id'], values):
                    r = history_train[metric][history_train[metric]['class_id']==c_id][f'epoch {epoch}'].index
                    history_train[metric].loc[r, f'epoch {epoch}'] = v
                history_train[metric].to_csv(args.project / f'{metric}_train.csv', encoding='utf-8-sig', index=False)

        # add new column(epoch) to precision.csv, recall.csv and f1score.csv
        for metric, values in eval_valid.class_scores.items():
            if metric != 'class_id':
                history_valid[metric][f'epoch {epoch}'] = 0
                for c_id, v in zip(eval_valid.class_scores['class_id'], values):
                    r = history_valid[metric][history_valid[metric]['class_id']==c_id][f'epoch {epoch}'].index
                    history_valid[metric].loc[r, f'epoch {epoch}'] = v
                history_valid[metric].to_csv(args.project / f'{metric}_valid.csv', encoding='utf-8-sig', index=False)
            
        # save performance graph
        save_performance_graph(args.project / 'summary.csv', args.project / 'performance.png')
        
        # model save
        if  best_loss is None or eval_valid.loss < best_loss: 
            print(f'Validation loss got better {best_loss} --> {eval_valid.loss}.  Saving model ...')
            torch.save({'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best acc': eval_valid.acc if best_acc is None or eval_valid.acc > best_acc else best_acc,
                        'best loss': eval_valid.loss if best_loss is None or eval_valid.loss < best_loss else eval_valid.loss,
                        'epoch': epoch,
                        },
                       args.project / 'weights' / 'best_loss.pth.tar')
            best_loss = eval_valid.loss
            
            # save valid predictions
            pred_frame = pd.DataFrame({
                "doc": test_loader.dataset.doc,
                "category": list(map(lambda x: ''.join(id2cat[x]), test_loader.dataset.label)),
                "predictions": list(map(lambda x: ''.join(id2cat[x]), eval_valid.predictions))
            })
            pred_frame.to_csv(args.project / 'best_loss_predictions.csv', encoding='utf-8-sig', index=False)
            patience = 0
        else:
            logger.info(f'patience {patience} --> {patience+1}')
            patience += 1
        
        if patience >= args.patience:
            logger.info('Early Stop!')
            break
        
def train(model, train_loader, optimizer, criterion, scheduler, device):
    eval_train = Evaluator(model.num_classes)
    model.train()
    
    for inputs, label in tqdm(train_loader, total=len(train_loader)):
        for k, v in inputs.items():
            inputs[k] = v.to(device, non_blocking=True)
        output = model(**inputs)
        target = label2target(output, label).to(device, non_blocking=True)
        loss = criterion(output, target)
        
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()  # Update learning rate
        
        # update score
        pred = output.argmax(1)
        eval_train.update(pred.tolist(), label.tolist(), loss=float(loss)*len(label))
    eval_train.compute()
    return eval_train
        
    
def valid(model, valid_loader, criterion, device):
    eval_valid = Evaluator(model.num_classes)
    
    model.eval()
    with torch.no_grad():
        for inputs, label in tqdm(valid_loader, total=len(valid_loader)):
            for k, v in inputs.items():
                inputs[k] = v.to(device, non_blocking=True)
            output = model(**inputs)
            target = label2target(output, label).to(device, non_blocking=True)
            loss = criterion(output, target)
            
            # update score
            pred = output.argmax(1)
            eval_valid.update(pred.tolist(), label.tolist(), loss=float(loss)*len(label))
    eval_valid.compute()
            
    return eval_valid


def get_args():
    FILE = Path(__file__).resolve()
    DATA = FILE.parents[1]
    ROOT = FILE.parents[0]  # root directory
    save_dir = increment_path(Path(ROOT) / 'runs'/ 'train' / 'exp')

    # Dataset
    parser=argparse.ArgumentParser(
            description='Training Disease Recognition in Pet CT')
    parser.add_argument('--root', default=DATA / 'data' / '1. 실습용자료_final.txt', type=str,
                        help='data format should be txt, sep="|"')
    parser.add_argument('--project', default=save_dir, type=str)
    parser.add_argument('--num-test', default=100000, type=int,
                        help='the number of test data')
    parser.add_argument('--upsample', default='', type=str,
                        help='"shuffle", "reproduce", "random"')
    parser.add_argument('--minimum', default=500, type=int,
                        help='(upsample) setting the minimum number of data of each categories')
    parser.add_argument('--target', default='S', type=str,
                        help='target')

    # DataLoader
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                        help='mini-batch size (default: 16)'
                             '[kobert] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                             '[kogpt2] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                             '[kogpt3] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50')

    # Model
    parser.add_argument('-m', '--model', default='kobert', type=str,
                        help='Model to train. Available models are ["kobert", "mlbert", "bert", "albert", "asbart", "kogpt2", "kogpt3", "electra", "funnel"]. default is "kobert".')
    parser.add_argument('--n-layers', default=1, type=int,
                        help='')
    parser.add_argument('-bn', '--batchnorm', action='store_true', help='')
    parser.add_argument('-ln', '--layernorm', action='store_true', help='')
    parser.add_argument('--dr-rate', default=None, type=float,
                        help='')
    parser.add_argument('--bias-off', action='store_false',
                        help='')

    # Train setting
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--patience', default=10, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')

    # Loss
    parser.add_argument('--loss', default='FCE', type=str,
                        help='Loss function. Availabel loss functions are ["CE", "FCE", "ICE"] . default is Focal Cross Entropy(FCE).')

    # Learning rate
    parser.add_argument('-lr', '--learning-rate', default=0.02, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-scheduler', default='cosine_with_restarts',
                        type=str, help='Available schedulers are "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup')
    parser.add_argument('--warmup-ratio', default=0.01, type=float, help='lr-scheduler')


    # Optimizer
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='default is AdamW')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='momentum1 in Adam')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='momentum2 in Adam')
    parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight_decay')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-8)
    parser.add_argument('--amsgrad', action='store_true')

    # Single GPU Train
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use. "cpu", "cuda", "cuda:0", "cuda:1"')

    parser.add_argument('--seed', default=5986, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--max-len', default=50, type=int,
                        help='max sequence length to cut or pad')


    args=parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()
    
    create_directory(args.project / 'weights')
    create_directory(args.project)
    logger = create_logger(args.project, file_name='log.txt')

    # save config
    with open(args.project / 'config.json', 'w', encoding='cp949') as f:
        arg_dict = {k: (str(v) if type(v)==pathlib.PosixPath else v) for k, v in args.__dict__.items()}
        json.dump(arg_dict, f, indent=4)

    print('output path:', args.project)

    best_acc = None
    best_loss = None

    main(args)