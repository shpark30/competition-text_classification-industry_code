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

def main(args):
    checkpoint = torch.load(os.path.join(args.exp_path, 'weights/best.pth.tar'), map_location=args.device)
    with open(os.path.join(args.exp_path, 'config.json'), 'r') as f:
        checkpoint_args = json.load(f)
    with open(os.path.join(args.exp_path, 'id2cat.json'), 'r') as f:
        id2cat = json.load(f)
    with open(os.path.join(args.exp_path, 'cat2id.json'), 'r') as f:
        cat2id = json.load(f)
        
    (model, train_set, test_set), cat2id, id2cat = get_model_dataset(checkpoint_args['model'], args.root, checkpoint_args['num_test'], upsample='', target='S', max_len=checkpoint_args['max_len'], seed=checkpoint_args['seed'])
    train_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=False)
    valid_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=False)
    model = model.to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    
    betas = (checkpoint_args['beta1'], checkpoint_args['beta2'])
    optimizer = get_optimizer(optimizer_type=checkpoint_args['optimizer'], model=model,
                              lr=checkpoint_args['lr'], betas=betas,
                              weight_decay=checkpoint_args['weight_decay'], eps=1e-08, amsgrad=False)
#     optimizer.load_state_dict(checkpoint['optimizer']) # build optimizer
    t_total = len(train_loader) * args.additional_epochs
    scheduler = get_scheduler(name=checkpoint_args['lr_scheduler'], optimizer=optimizer, num_warmup_steps=1000, num_training_steps=t_total)
    
    criterion = get_loss(loss_type='FCE')
    
    best_score = None
    for epoch in range(args.additional_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, scheduler, args.device)
#         predictions, valid_loss, acc, class_scores = valid(model, valid_loader, criterion, args.device)
        
        # logging scores
        macro_pc = statistics.mean(class_scores['precision'])
        macro_rc = statistics.mean(class_scores['recall'])
        macro_f1 = statistics.mean(class_scores['f1score'])
        print(f'Epoch {epoch} Result')
        print(f'\ttrain loss: {train_loss}\tvalid_loss: {valid_loss}')
        print(f'\tacc: {round(acc, 6)}\tpc: {round(macro_pc, 6)}\trc: {round(macro_rc, 6)}\tf1: {round(macro_f1, 6)}')
        print(f'\t{epoch} epoch train loss: {train_loss}')
        torch.save({'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best score': checkpoint['best score'],
                        'epoch': checkpoint['epoch'],
                        'additional_epoch': epoch
                       },
                        os.path.join(args.exp_path, 'weights', 'additional.pth.tar'))
        print('save model')
#         if best_score is None or valid_loss < best_score:
#             torch.save({'state_dict': model.state_dict(),
#                         'optimizer': optimizer.state_dict(),
#                         'scheduler': scheduler.state_dict(),
#                         'best score': checkpoint['best score'],
#                         'epoch': checkpoint['epoch'],
#                         'additional_epoch': epoch
#                        },
#                         os.path.join(args.exp_path, 'weights', 'additional.pth.tar'))
#             print('save model')
#             best_score = valid_loss
#         else:
#             continue
                
                
    
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
        
                   
    train, test, cat2id, id2cat = preprocess(data, num_test=num_test, upsample=upsample, target=target, seed=seed)
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
    parser=argparse.ArgumentParser(
        description='')

    parser.add_argument('--root', default= '../../data/1. 실습용자료.txt', type=str,
                        help='data format should be txt, sep="|"')
    parser.add_argument('--exp-path', default='./runs/train/exp14', type=str,
                       help='path of a directory which contains the "weights" folder and id2cat.json')
    parser.add_argument('--additional-epochs', default=5, type=int, metavar='N',
                        help='')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N',
                        help='mini-batch size (default: 16)'
                             '[kobert] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                             '[kogpt2] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                             '[kogpt3] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50')
    
    parser.add_argument('--device', default='cuda:1', type=str)
    args=parser.parse_args()
    
    main(args)