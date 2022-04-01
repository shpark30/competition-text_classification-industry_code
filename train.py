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
os.environ["TOKENIZERS_PARALLELISM"] = "false" # https://github.com/pytorch/pytorch/issues/57273
import pathlib
from pathlib import Path
import argparse

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import transformers # kakao kogpt requires transformers version 4.12.0
from transformers.optimization import get_scheduler

import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

print('pytorch ver:', torch.__version__)
print('transformers ver:', transformers.__version__)

from loss import CrossEntropy, FocalCrossEntropy, label2target
from utils import create_logger, create_directory, increment_path, save_performance_graph
from dataset import preprocess, train_test_split, KOGPT2ClassifyDataset, KOGPT3ClassifyDataset, KOBERTClassifyDataset, EnsembleDataset
from network import KOGPT2Classifier, KOGPT3Classifier, KOBERTClassifier, EnsembleClassifier

FILE = Path(__file__).resolve()
DATA = FILE.parents[2]
ROOT = FILE.parents[0]  # root directory
save_dir = increment_path(Path(ROOT) / 'runs'/ 'train' / 'exp')
    
# Dataset
parser=argparse.ArgumentParser(
        description='Training Disease Recognition in Pet CT')
# parser.add_argument('root', metavar='DIR',
#                     help='path to data')
parser.add_argument('--root', default=DATA / 'data' / '1. 실습용자료_중복제거_hsp.txt', type=str,
                    help='data format should be txt, sep="|"')
parser.add_argument('--project', default=save_dir, type=str)
parser.add_argument('--num-test', default=100000, type=int,
                    help='the number of test data')
parser.add_argument('--upsample', action='store_true',
                    help='')
parser.add_argument('--target', default='S', type=str,
                    help='target')
# parser.add_argument('--num_test_ratio', default=0.1, type=float,
#                     help='a ratio of test data')

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
                    help='Model to train. Available models are ["kobert", "kogpt2", "kogpt3", "ensemble"]. default is "kogpt3".')

# Train setting
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

# Loss
parser.add_argument('--loss', default='FCE', type=str,
                    help='Loss function. Availabel loss functions are . default is Focal Cross Entropy(FCE).')

# Learning rate
parser.add_argument('-lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-scheduler', default='cosine_with_restarts',
                    type=str, help='Available schedulers are "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup')
parser.add_argument('--warmup-step', default=1000, type=int, help='lr-scheduler')


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

parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training.')
parser.add_argument('--max-len', default=50, type=int,
                    help='max sequence length to cut or pad')


args=parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(torch.cuda.device_count())))
os.environ["CUDA_LAUNCH_BLOCKING"] = ",".join(map(str, range(torch.cuda.device_count())))

create_directory(args.project / 'weights')
create_directory(args.project)
logger = create_logger(args.project, file_name='log.txt')

# save config
with open(args.project / 'config.json', 'w', encoding='cp949') as f:
    arg_dict = {k: (str(v) if type(v)==pathlib.PosixPath else v) for k, v in args.__dict__.items()}
    json.dump(arg_dict, f, indent=4)

print('output path:', args.project)

best_score = None

def main(args):
    global best_score
    global best_epoch
    
    # preprocess data
    (model, train_set, test_set), cat2id, id2cat = get_model_dataset(args.model, args.root, args.num_test, args.upsample, args.target, args.max_len, args.seed)
    model = model.to(args.device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=True, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers,
                             shuffle=False, pin_memory=False)
    
    logger.info(f'# train data: {len(train_set)}')
    logger.info(f'# test  data: {len(test_set)}')
    
    with open(args.project / 'cat2id.json', 'w', encoding='cp949') as f:
        json.dump(cat2id, f, indent=4)
    with open(args.project / 'id2cat.json', 'w', encoding='cp949') as f:
        json.dump(id2cat, f, indent=4)
    
    # optimizer
    betas=(args.beta1, args.beta2)
    optimizer = get_optimizer(optimizer_type=args.optimizer, model=model, lr=args.lr, betas=betas,
                              weight_decay=args.weight_decay, eps=args.epsilon, amsgrad=args.amsgrad)

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
            model.load_state_dict(checkpoint['state_dict'])
            # build optimizer
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']
            best_score = checkpoint['best_score']
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            print('start epoch: {}'.format(args.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    # lr-scheduler
    t_total = len(train_loader) * args.epochs
    scheduler = get_scheduler(name=args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.warmup_step, num_training_steps=t_total)

    # loss function
    criterion = get_loss(args.loss)
            
    # train
    for epoch in range(args.start_epoch, args.epochs):
        # epoch train
        train_loss = train(model, train_loader, optimizer, criterion, scheduler, args.device)
    
        # epoch validation
        predictions, valid_loss, acc, class_scores = valid(model, test_loader, criterion, args.device)
        
        # logging scores
        macro_pc = statistics.mean(class_scores['precision'])
        macro_rc = statistics.mean(class_scores['recall'])
        macro_f1 = statistics.mean(class_scores['f1score'])
        logger.info(f'Epoch {epoch} Result')
        logger.info(f'\ttrain loss: {train_loss}\tvalid_loss: {valid_loss}')
        logger.info(f'\tacc: {round(acc, 6)}\tpc: {round(macro_pc, 6)}\trc: {round(macro_rc, 6)}\tf1: {round(macro_f1, 6)}')
        
        # save scores
        if epoch==args.start_epoch:
            # summary.csv
            with open(args.project / 'summary.csv', 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(['epoch', 'train loss', 'valid loss', 'accuracy', 'precision', 'recall', 'f1score'])
            # base frame for precisions, recalls and f1scores
            class_id = list(set(train_loader.dataset.label))
            num_train_data, num_valid_data = [0] * len(class_id), [0] * len(class_id)
            for c_id, n in dict(Counter(train_loader.dataset.label)).items():
                num_train_data[c_id] = n
            for c_id, n in dict(Counter(test_loader.dataset.label)).items():
                num_valid_data[c_id] = n
            history_frame = defaultdict(lambda: pd.DataFrame({
                'class_id': class_id,
                'class': list(map(lambda x: ''.join(id2cat[x]), class_id)),
                '# train data' : num_train_data,
                '# valid data' : num_valid_data
            }))
            
        # add new line to summary.csv
        with open(args.project / 'summary.csv', 'a', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([epoch, train_loss, valid_loss, acc, macro_pc, macro_rc, macro_f1])
            
        # add new column(epoch) to precision.csv, recall.csv and f1score.csv
        for metric, values in class_scores.items():
            if metric != 'class_id':
                history_frame[metric][f'epoch {epoch}'] = 0
                for c_id, v in zip(class_scores['class_id'], values):
                    r = history_frame[metric][history_frame[metric]['class_id']==c_id][f'epoch {epoch}'].index
                    history_frame[metric].loc[r, f'epoch {epoch}'] = v
                history_frame[metric].to_csv(args.project / f'{metric}.csv', encoding='utf-8', index=False)
            
        # save performance graph
        save_performance_graph(args.project / 'summary.csv', args.project / 'performance.png')
        
        # model save
        epoch_score = acc
        is_best = best_score is None or epoch_score > best_score
        torch.save({'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best score': acc if is_best else best_score,
                    'epoch': epoch},
                   args.project / 'weights' / 'checkpoint.pth.tar')
        
        if is_best: 
            print(f'Validation score got better {best_score} --> {epoch_score}.  Saving model ...')
            shutil.copyfile(args.project / 'weights' / 'checkpoint.pth.tar', args.project / 'weights' / 'best.pth.tar')
            best_score = epoch_score
            
            # save valid predictions
            pred_frame = pd.DataFrame({
                "doc": test_loader.dataset.doc,
                "category": list(map(lambda x: ''.join(id2cat[x]), test_loader.dataset.label)),
                "predictions": list(map(lambda x: ''.join(id2cat[x]), predictions))
            })
            pred_frame.to_csv(args.project / 'best_model_predictions.csv', encoding='utf-8-sig', index=False)
            
            
def train(model, train_loader, optimizer, criterion, scheduler, device):
    train_loss = 0
    model.train()
#     with torch.autograd.detect_anomaly():
    for (input_ids, attention_mask, token_type_ids, label) in tqdm(train_loader, total=len(train_loader)):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        token_type_ids = token_type_ids.to(device, non_blocking=True)

        # forward propagation
        output = model(input_ids, attention_mask, token_type_ids)
        target = label2target(output, label).to(device, non_blocking=True)
        loss = criterion(output, target)
        
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule

        train_loss += float(loss)*len(label)
    train_loss /= len(train_loader.dataset)
    return train_loss
        
    
def valid(model, valid_loader, criterion, device):
    valid_loss = 0
    class_scores = defaultdict(list)
    predictions = []
    valid_confusion_matrix = np.zeros((model.num_classes, model.num_classes), dtype=np.int64)
    
    model.eval()
    with torch.no_grad():
        for (input_ids, attention_mask, token_type_ids, label) in tqdm(valid_loader, total=len(valid_loader)):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            token_type_ids = token_type_ids.to(device, non_blocking=True)

            # forward propagation
            output = model(input_ids, attention_mask, token_type_ids)
            target = label2target(output, label).to(device, non_blocking=True)
            loss = criterion(output, target)
            valid_loss += loss*len(label)
                         
            
            # get confusion matrix
            pred = torch.argmax(output, 1).cpu()
            valid_confusion_matrix += confusion_matrix(label, pred, labels=list(range(model.num_classes)))
            predictions += pred.tolist()
            
        valid_loss /= len(valid_loader.dataset)
        acc = np.diagonal(valid_confusion_matrix).sum() / valid_confusion_matrix.sum()
        for c in range(len(valid_confusion_matrix)):
            num_pred = valid_confusion_matrix[:, c].sum()
            num_true = valid_confusion_matrix[c].sum()
            TP = valid_confusion_matrix[c, c]
            FP = num_true - TP
            FN = num_pred - TP
            PC = TP/num_pred if num_pred != 0 else 0 # TP / (TP+FP)
            RC = TP/num_true if num_true != 0 else 0  # TP / (TP+FN)
            F1 = 2 * PC * RC / (PC + RC) if PC + RC != 0 else 0 # (2 * PC * RC) / (PC + RC)
            class_scores['class_id'].append(c)
            class_scores['precision'].append(PC)
            class_scores['recall'].append(RC)
            class_scores['f1score'].append(F1)
            
    return predictions, float(valid_loss), acc, class_scores


def get_model_dataset(model_type, root, num_test, upsample, target, max_len, seed):
    def _get_kobert_model_dataset(num_classes, doc_train, label_train, doc_test, label_test, max_len):
        kobert, vocab = get_pytorch_kobert_model()
        tokenizer_path = get_tokenizer()
        tokenizer = nlp.data.BERTSPTokenizer(tokenizer_path, vocab, lower=False)
        transform = nlp.data.BERTSentenceTransform(
                    tokenizer, max_seq_length=max_len, pad=True, pair=False) 
        
        train_set = KOBERTClassifyDataset(doc_train, label_train, transform)
        test_set = KOBERTClassifyDataset(doc_test, label_test, transform)
        
        model = KOBERTClassifier(kobert, num_classes=num_classes)
        return model, train_set, test_set
    
    def _get_kogpt2_model_dataset(num_classes, doc_train, label_train, doc_test, label_test, max_len):
        train_set = KOGPT2ClassifyDataset(doc_train, label_train, max_len=max_len, padding='max_length', truncation=True)
        test_set = KOGPT2ClassifyDataset(doc_test, label_test, max_len=max_len, padding='max_length', truncation=True)
        
        model = KOGPT2Classifier(num_classes=num_classes, pad_token_id = train_set.tokenizer.eos_token_id)
        return model, train_set, test_set
    
    def _get_kogpt3_model_dataset(num_classes, doc_train, label_train, doc_test, label_test, max_len):
        train_set = KOGPT3ClassifyDataset(doc_train, label_train, max_len=max_len, padding='max_length', truncation=True)
        test_set = KOGPT3ClassifyDataset(doc_test, label_test, max_len=max_len, padding='max_length', truncation=True)
        
        model = KOGPT3Classifier(num_classes=num_classes, pad_token_id = train_set.tokenizer.eos_token_id)
        return model, train_set, test_set
    
    def _get_ensemble_model_dataset(num_classes, doc_train, label_train, doc_test, label_test, max_len):
        kobert, vocab = get_pytorch_kobert_model()
        tokenizer_path = get_tokenizer()
        tokenizer = nlp.data.BERTSPTokenizer(tokenizer_path, vocab, lower=False)
        transform = nlp.data.BERTSentenceTransform(
                    tokenizer, max_seq_length=max_len, pad=True, pair=False) 
        
        train_set = EnsembleDataset(doc_train, label_train, kobert_tokenizer=transform, max_len=max_len, padding='max_length', truncation=True)
        test_set = EnsembleDataset(doc_test, label_test, kobert_tokenizer=transform, max_len=max_len, padding='max_length', truncation=True)
        
        kobert = KOBERTClassifier(kobert, num_classes=num_classes)
        kogpt2 = KOGPT2Classifier(num_classes=num_classes, pad_token_id = train_set.kogpt_tokenizer.eos_token_id)
        model = EnsembleClassifier(kogpt2, kobert, num_classes=num_classes)
        return model, train_set, test_set
    
    try:
        data = pd.read_csv(root, sep='|', encoding='euc-kr')
    except:
        data = pd.read_csv(root, sep='|', encoding='utf-8')
        
    test_ratio = num_test/len(data)
    train, test, cat2id, id2cat = preprocess(data, test_ratio=test_ratio, upsample=upsample, target=target, seed=seed)
    doc_train, doc_test, label_train, label_test = train['text'].tolist(), test['text'].tolist(), train['label'].tolist(), test['label'].tolist()
#     doc_train, label_train, doc_test, label_test = train_test_split(doc, label, test_ratio=test_ratio, seed=seed)
    num_classes = len(cat2id.keys())
    
    if model_type=='kobert':
        return _get_kobert_model_dataset(num_classes, doc_train, label_train, doc_test, label_test, max_len), cat2id, id2cat
    elif model_type=='kogpt2':
        return _get_kogpt2_model_dataset(num_classes, doc_train, label_train, doc_test, label_test, max_len), cat2id, id2cat
    elif model_type=='kogpt3':
        return _get_kogpt3_model_dataset(num_classes, doc_train, label_train, doc_test, label_test, max_len), cat2id, id2cat
    elif model_type=='ensemble':
        return _get_ensemble_model_dataset(num_classes, doc_train, label_train, doc_test, label_test, max_len), cat2id, id2cat
    else:
        raise
        
def get_optimizer(optimizer_type, model, lr, betas, weight_decay, eps=1e-08, amsgrad=False):
    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    elif optimizer_type == 'AdamP':
        optimizer = torch.optim.AdamP(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    elif optimizer_type == 'RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
#     elif optimizer_type == 'SGD':
#         optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
#     elif optimizer_type == 'SGDW':
#         optimizer = torch.optim.SGDW(model.parameters(), lr=args.lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
#     elif optimizer_type == 'SGDP':
#         optimizer = torch.optim.SGDP(model.parameters(), lr=args.lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    else:
        raise
    return optimizer

def get_lr_scheduler(schduler_type, **kwargs):
    if schduler_type == 'warmup_cosine_restart':
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(**kwargs) #num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    else:
        raise
    return scheduler

def get_loss(loss_type, **kwargs):
    if loss_type == 'CE':
        criterion = CrossEntropy(**kwargs)
    elif loss_type == 'FCE':
        criterion = FocalCrossEntropy(**kwargs)
    else:
        raise
    return criterion

if __name__=='__main__':
    main(args)
    