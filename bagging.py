import os
import logging
import json
import statistics
import csv
from copy import copy, deepcopy
from typing import Optional, Union, List
import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm
from collections import defaultdict, Counter, OrderedDict
import pathlib
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import random

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers.optimization import get_scheduler

import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from load import *
from loss import CrossEntropy, FocalCrossEntropy, label2target, get_loss
from utils import vote, create_logger, create_directory, increment_path, save_performance_graph, get_optimizer, Evaluator
from dataset import train_test_split, num2code, concat_text, bootstrap, upsample_corpus, EnsembleDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false" # https://github.com/pytorch/pytorch/issues/57273
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'


def main(args):
    # save config
    create_directory(args.project)
    print('output_path:', args.project)
    with open(args.project / 'config.json', 'w', encoding='utf-8-sig') as f:
        arg_dict = {k: (str(v) if type(v)==pathlib.PosixPath else v) for k, v in args.__dict__.items()}
        json.dump(arg_dict, f, indent=4)
    # create logger
    logger = create_logger(args.project, name=f'ensemble', file_name='log.txt')
    
    # load data
    data = pd.read_csv(args.root, sep='|', encoding='cp949')
    
    # 대-중-소 이어붙이기
    data['digit_2'] = data['digit_2'].apply(lambda x:  num2code(x, 2))
    data['digit_3'] = data['digit_3'].apply(lambda x:  num2code(x, 3))
    data['digit'] = data['digit_1'] + data['digit_2'] + data['digit_3']
    
    # 중복 행 제거
    data = data.drop_duplicates(subset=['text_obj', 'text_mthd', 'text_deal', 'digit'], keep='first')
    
    # 결측치 채우기
    data = data.fillna('')
    
    # 레이블 인코딩
    unique_digit = data['digit'].sort_values().drop_duplicates().tolist()
    cat2id, id2cat = {}, {}
    for i, cat in enumerate(unique_digit):
        cat2id[cat] = i
        id2cat[i] = cat
    data['label'] = data['digit'].apply(lambda x: cat2id[x])
    
    # Train Test Split
    test_ratio = args.num_test/len(data)
    train, test = train_test_split(data, test_ratio=test_ratio, seed=5986)
    logger.info(f'# base model train data: {args.num_samples}')
    logger.info(f'# ensemble test data: {len(test)}')
    
    # Sub Sampling
    sub_data_list, oob = bootstrap(train, args.estimators,
                              dist=args.sample_dist, num_samples=args.num_samples,
                              seed=args.seed)
#     test = pd.concat([test, oob]).reset_index(drop=True)
    logger.info(f'sub data 수: {len(sub_data_list)}')
    logger.info(f'sub data 카테고리 수: {[(i, len(sub_data["label"].unique())) for i, sub_data in enumerate(sub_data_list)]}')
    logger.info(f'out of bag(oob) will be used to test. new_test = test({len(test)-len(oob)}) + oob({len(oob)}) = {len(test)}')
    
    # Save sub datasets
    for i, sub_data in enumerate(sub_data_list):
        sub_data.to_csv(args.project / f'sub_data{i}.csv', index=False, encoding='utf-8-sig')
    # Save test dataset
    test.to_csv(args.project / 'test_data.csv', index=False, encoding='utf-8-sig')
    
    # Column Selection
    sub_data_list = [sub_data[['label', 'text_obj', 'text_mthd', 'text_deal']]
                  for sub_data in sub_data_list]
    test = test[['label', 'text_obj', 'text_mthd', 'text_deal']]
    
    # Sub train-valid split
    sub_train_list, sub_valid_list = zip(*map(lambda sub_data: train_test_split(
                                                    sub_data, test_ratio=args.num_sub_valid/len(sub_data), seed=args.seed),
                                              sub_data_list))
    
    # Upsampling
    logger.info(f'sub train 수(before upsample): {[len(sub_train) for sub_train in sub_train_list]}')
    if args.upsample != '':
        sub_train_list = list(map(
            lambda sub_train: upsample_corpus(sub_train, minimum=args.minimum, method=args.upsample, seed=args.seed),
            sub_train_list
        ))
    logger.info(f'sub train 수(after upsample): {[len(sub_train) for sub_train in sub_train_list]}')
    
    
    # text 이어붙이기
    sub_train_list = [concat_text(sub_train) for sub_train in sub_train_list]
    sub_valid_list = [concat_text(sub_valid) for sub_valid in sub_valid_list]
    test = concat_text(test)
    
    # Load Backbone, Data Loaders
    backbones = {}
    tokenizers = {}
    train_loaders = []
    valid_loaders = []
    for model_type, (num, i) in args.num_base_models.items():
        if num:
            backbone, tokenizer = load_backbone_tokenizer(model_type, max_len=args.max_len)
            backbones[model_type] = backbone
            tokenizers[f'{model_type}_tokenizer'] = tokenizer
            for sub_train, sub_valid in zip(sub_train_list[i-num:i], sub_valid_list[i-num:i]):
                train_set = load_dataset(model_type, sub_train['text'].tolist(), sub_train['label'].tolist(),
                                         tokenizer, max_len=args.max_len)
                valid_set = load_dataset(model_type, sub_valid['text'].tolist(), sub_valid['label'].tolist(),
                                         tokenizer, max_len=args.max_len)
                train_loaders.append(DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                                      shuffle=True, pin_memory=False))
                valid_loaders.append(DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.workers,
                                     shuffle=False, pin_memory=False))
    for model_type, (num, i) in args.num_base_models.items():
        logger.info(f'{model_type} base model {num}개')
    
    # Train!
    base_models = []
    
    num_classes = len(cat2id)
    for e in range(args.estimators):
        patience = 0 # early stop patience
        model_args = copy(args) # config for {e}th model
        model_args.project = args.project / f'model{e}' # runs/train/exp00/model{e}
        create_directory(model_args.project / 'weights') # runs/train/exp00/model{e}/weights
        
        
        # dataset
        train_loader = train_loaders[e]
        valid_loader = valid_loaders[e]
        logger.info(f'# train data: {len(train_loader.dataset)}')
        logger.info(f'# valid data: {len(valid_loader.dataset)}')

        # build model
        for model_type, (num, acm_num) in args.num_base_models.items():
            if e <= acm_num-1:
                model_args.model = model_type
                break
            else:
                continue
        model = load_model(model_type=model_args.model,
                           backbone=copy(backbones[model_args.model]),
                           num_classes=num_classes,
                           num_layers=args.n_layers, 
                           dr_rate=args.dr_rate,
                           bias=True,
                           batchnorm=args.batchnorm,
                           layernorm=args.layernorm)
        
        model = model.to(args.device)

        # save config
        with open(model_args.project / 'config.json', 'w', encoding='utf-8-sig') as f:
            arg_dict = {k: (str(v) if type(v)==pathlib.PosixPath else v) for k, v in model_args.__dict__.items()}
            json.dump(arg_dict, f, indent=4)

        # optimizer
        optimizer = get_optimizer(args.optimizer, model, args.lr,
                                  (args.beta1, args.beta2), args.weight_decay, eps=1e-08, amsgrad=False)

        # lr scheduler
        max_iter = len(train_loader) * args.epochs
        num_warmup_step = (max_iter * args.warmup_ratio)
        scheduler = get_scheduler(name=args.lr_scheduler, optimizer=optimizer, 
                                  num_warmup_steps=num_warmup_step, num_training_steps=max_iter)

        if args.loss != 'mix':
            criterion = get_loss(args.loss)
        else:
            n_models = eval(f'args.n_{model_args.model}')
            
            before_num_acm = 0
            for model_type, (n, num_acm) in args.num_base_models.items():
                if model_type == model_args.model:
                    ith = n-(num_acm-e)
            if ith%2==1:
                criterion = get_loss('FCE')
            else:
                criterion = get_loss('ICE')
        
        logger.info(f'Start Training Model{e} {model_args.model}')
        best_acc = None
        best_loss = None
        for epoch in range(args.epochs):
            eval_train = Evaluator(num_classes)
            eval_valid = Evaluator(num_classes)
            
            # train
            model.train()
            for inputs, label in tqdm(train_loader, total=len(train_loader)):
                for k, v in inputs.items():
                    inputs[k] = v.to(args.device, non_blocking=True).long()
                    
                # forward
                try:
                    output = model(**inputs)
                except:
                    import pdb
                    pdb.set_trace()
                    output = model(**inputs)
                target = label2target(output, label).to(args.device, non_blocking=True)
                loss = criterion(output, target, softmax=True)

                # backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                
                # update score
                pred = output.argmax(1)
                eval_train.update(pred.tolist(), label.tolist(), loss=float(loss)*len(label))
            
            # validation
            model.eval()
            with torch.no_grad():
                for inputs, label in tqdm(valid_loader, total=len(valid_loader)):
                    for k, v in inputs.items():
                        inputs[k] = v.to(args.device, non_blocking=True).long()
                        
                    # forward
                    output = model(**inputs)
                    target = label2target(output, label).to(args.device, non_blocking=True)
                    loss = criterion(output, target, softmax=True)

                    # update score
                    pred = torch.argmax(output, 1).cpu()
                    eval_valid.update(pred.tolist(), label.tolist(), loss=float(loss)*len(label))
                    
            eval_train.compute()
            eval_valid.compute()
            
            # logging scores
            logger.info(f'Model {e}/{args.estimators-1} {model_args.model} | Epoch {epoch} Result')
            logger.info(f'\ttrain | loss: {eval_train.loss}\tacc: {round(eval_train.acc, 6)}\tpc: {round(eval_train.macro_pc, 6)}\trc: {round(eval_train.macro_rc, 6)}\tf1: {round(eval_train.macro_f1, 6)}')
            logger.info(f'\tvalid | loss: {eval_valid.loss}\tacc: {round(eval_valid.acc, 6)}\tpc: {round(eval_valid.macro_pc, 6)}\trc: {round(eval_valid.macro_rc, 6)}\tf1: {round(eval_valid.macro_f1, 6)}')

            # save scores
            if epoch==0:
                # summary.csv
                with open(model_args.project / 'summary.csv', 'w', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow(['epoch', 'train loss', 'train acc', 'train pc', 'train rc', 'train f1',
                                          'valid loss', 'valid acc', 'valid pc', 'valid rc', 'valid f1'])
                # base frame for precisions, recalls and f1scores
                class_id = list(set(train_loader.dataset.label))
                num_train_data, num_valid_data = [0] * len(class_id), [0] * len(class_id)
                for c_id, n in dict(Counter(train_loader.dataset.label)).items():
                    num_train_data[c_id] = n
                for c_id, n in dict(Counter(valid_loader.dataset.label)).items():
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
            with open(model_args.project / 'summary.csv', 'a', newline='') as f:
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
                    history_train[metric].to_csv(model_args.project / f'{metric}_train.csv', encoding='utf-8-sig', index=False)
            
            # add new column(epoch) to precision.csv, recall.csv and f1score.csv
            for metric, values in eval_valid.class_scores.items():
                if metric != 'class_id':
                    history_valid[metric][f'epoch {epoch}'] = 0
                    for c_id, v in zip(eval_valid.class_scores['class_id'], values):
                        r = history_valid[metric][history_valid[metric]['class_id']==c_id][f'epoch {epoch}'].index
                        history_valid[metric].loc[r, f'epoch {epoch}'] = v
                    history_valid[metric].to_csv(model_args.project / f'{metric}_valid.csv', encoding='utf-8-sig', index=False)

            # save performance graph
            save_performance_graph(model_args.project / 'summary.csv', model_args.project / 'performance.png')

            # model save
#             if best_acc is None or eval_valid.acc > best_acc: 
#                 logger.info(f'Validation accuracy got better {best_acc} --> {eval_valid.acc}.  Saving model ...')
#                 shutil.copyfile(model_args.project / 'weights' / 'checkpoint.pth.tar',
#                                 model_args.project / 'weights' / 'best_acc.pth.tar')
#                 best_acc = eval_valid.acc

#                 # save valid predictions
#                 pred_frame = pd.DataFrame({
#                     "doc": valid_loader.dataset.doc,
#                     "category": list(map(lambda x: ''.join(id2cat[x]), valid_loader.dataset.label)),
#                     "predictions": list(map(lambda x: ''.join(id2cat[x]), eval_valid.predictions))
#                 })
#                 pred_frame.to_csv(model_args.project / 'best_acc_predictions.csv', encoding='utf-8-sig', index=False)
#                 del pred_frame
                
            if best_loss is None or eval_valid.loss <= best_loss:
                logger.info(f'Validation loss got better {best_loss} --> {eval_valid.loss}.  Saving model ...')
                torch.save({'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'best acc': eval_valid.acc if best_acc is None or eval_valid.acc > best_acc else best_acc,
                            'best loss': eval_valid.loss if best_loss is None or eval_valid.loss <= best_loss else best_loss,
                            'epoch': epoch},
                           model_args.project / 'weights' / 'best_loss.pth.tar')
#                 shutil.copyfile(model_args.project / 'weights' / 'checkpoint.pth.tar',
#                                 model_args.project / 'weights' / 'best_loss.pth.tar')
                best_loss = eval_valid.loss

                # save valid predictions
                pred_frame = pd.DataFrame({
                    "doc": valid_loader.dataset.doc,
                    "category": list(map(lambda x: ''.join(id2cat[x]), valid_loader.dataset.label)),
                    "predictions": list(map(lambda x: ''.join(id2cat[x]), eval_valid.predictions))
                })
                pred_frame.to_csv(model_args.project / 'best_loss_predictions.csv', encoding='utf-8-sig', index=False)
                patience = 0
                del pred_frame
            else:
                logger.info(f'patience {patience} --> {patience+1}')
                patience += 1

            if patience >= args.patience:
                logger.info('Early Stop!')
                break
        model = model.to('cpu')
        del model
#         base_models.append(deepcopy(model.classifier))
        
    
    # Ensemble
    # Load Base Models
    base_models = defaultdict(list)
    for e in range(args.estimators):
        model_path = args.project / f'model{e}' / 'weights' / 'best_loss.pth.tar'
        config_path = args.project / f'model{e}' / 'config.json'
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            model_args = json.load(f)
        model = load_model(model_type=model_args['model'],
                           backbone=copy(backbones[model_args['model']]),
                           num_classes=num_classes,
                           num_layers=model_args['n_layers'], 
                           dr_rate=model_args['dr_rate'],
                           bias=True,
                           batchnorm=model_args['batchnorm'],
                           layernorm=model_args['layernorm'])
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
        except:
            import pdb
            pdb.set_trace()
            model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
        model.eval()
        base_models[model_args['model']].append(copy(model))
        
    
    # Build Model
#     ensemble = EnsembleClassifier(num_classes, )
#                                   kobert=kobert if args.n_kobert else None,
#                                   kogpt2=kogpt2 if args.n_kogpt2 else None,
#                                   kobert_classifiers=base_models[:args.n_kobert],
#                                   kogpt2_classifiers=base_models[args.n_kobert:args.n_kobert+args.n_kogpt2])
    
    # Ensemble Data Loader
    test_set = EnsembleDataset(test['text'].tolist(), test['label'].tolist(), **tokenizers,
#                                kobert_tokenizer=kobert_transform if args.n_kobert else None,
#                                kogpt2_tokenizer=kogpt2_tokenizer if args.n_kogpt2 else None,
                               #mlbert_tokenizer=None, # mlbert_tokenizer if args.n_mlbert else None,
                               max_len=args.max_len)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers,
                             shuffle=False, pin_memory=False)
    
    # Test Ensemble
    eval_mean = Evaluator(num_classes)
    eval_vote = Evaluator(num_classes)
    eval_base_models = [Evaluator(num_classes) for e in range(args.estimators)]
    
    logger.info('Start Test Ensemble')
    with torch.no_grad():
        for inputs, label in tqdm(test_loader, total=len(test_loader)):
            # forward propagation
            base_model_outputs = []
            for model_type, models in base_models.items():
                for model in models:
                    model.to(args.device)
                    _inputs = {}
                    _inputs['input_ids'] = inputs[model_type][:, 0].to(args.device, non_blocking=True).long()
                    _inputs['attention_mask'] = inputs[model_type][:, 1].to(args.device, non_blocking=True).long()
                    if 'bart' not in model_type:
                        _inputs['token_type_ids'] = inputs[model_type][:, 2].to(args.device, non_blocking=True).long()
                    output = model(**_inputs)
                    pred = F.softmax(output, dim=1)
                    base_model_outputs.append(pred.cpu())
            
            base_model_outputs = torch.stack(base_model_outputs)
#             base_model_outputs = ensemble(**inputs) # (n_model, n_batch, num_classes)
            ensemble_output_mean = torch.mean(base_model_outputs, dim=0)
            ensemble_output_vote = vote(base_model_outputs, dim=2)
            target = label2target(base_model_outputs[0], label)
            
            # update
            for i, base_model_output in enumerate(base_model_outputs):
                loss = criterion(base_model_output.cpu(), target, softmax=False)
                pred_base_model = base_model_output.argmax(1).cpu()
                eval_base_models[i].update(pred_base_model.tolist(), label.tolist(), loss=float(loss)*len(label))
            
            # ensemble result
            ensemble_loss = criterion(ensemble_output_mean, target, softmax=False)
            pred_mean = ensemble_output_mean.argmax(1).cpu()
            pred_vote = ensemble_output_vote.sum(0).argmax(1).cpu()
            eval_mean.update(pred_mean.tolist(), label.tolist(), loss=float(ensemble_loss)*len(label))
            eval_vote.update(pred_vote.tolist(), label.tolist())
            
        # compute acc, pc, rc, f1
        for eval_base_model in eval_base_models:
            eval_base_model.compute()
        eval_mean.compute()
        eval_vote.compute()
        
    # logging scores
        # ensemble
    logger.info(f'mean agg| loss: {round(eval_mean.loss, 6)}\tacc: {round(eval_mean.acc, 6)}\tpc: {round(eval_mean.macro_pc, 6)}\trc: {round(eval_mean.macro_rc, 6)}\tf1: {round(eval_mean.macro_f1, 6)}')
    logger.info(f'vote agg| loss: -\tacc: {round(eval_vote.acc, 6)}\tpc: {round(eval_vote.macro_pc, 6)}\trc: {round(eval_vote.macro_rc, 6)}\tf1: {round(eval_vote.macro_f1, 6)}')
        # base model
    for e, eval_base_model in enumerate(eval_base_models):
        logger.info(f'model{e} | \tacc: {round(eval_base_model.acc, 6)}\tpc: {round(eval_base_model.macro_pc, 6)}\trc: {round(eval_base_model.macro_rc, 6)}\tf1: {round(eval_base_model.macro_f1, 6)}')

    # save summary.csv
    with open(args.project / 'test_summary.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['agg_fn', 'test_loss', 'accuracy', 'precision', 'recall', 'f1score'])
        wr.writerow(['mean', eval_mean.loss, eval_mean.acc, eval_mean.macro_pc, eval_mean.macro_rc, eval_mean.macro_f1])
        wr.writerow(['vote', '-', eval_vote.acc, eval_vote.macro_pc, eval_vote.macro_rc, eval_vote.macro_f1])
        for e, eval_base_model in enumerate(eval_base_models):
            wr.writerow([f'model{e}', eval_base_model.loss, eval_base_model.acc, eval_base_model.macro_pc, eval_base_model.macro_rc, eval_base_model.macro_f1])
            
    # save precision.csv, recall.csv and f1score.csv
    class_id = list(set(id2cat.keys()))
    num_test_data = [0] * len(class_id)
    for c_id, n in dict(Counter(test_loader.dataset.label)).items():
        num_test_data[c_id] = n

    history_frame = pd.DataFrame({
        'class_id': class_id,
        'class': [''.join(id2cat[x]) for x in class_id],
        '# test data' : num_test_data
    })
    for metric, values in eval_mean.class_scores.items():
        if metric != 'class_id':
            history_frame[metric] = 0
            for c_id, v in zip(eval_mean.class_scores['class_id'], values):
                r = history_frame[history_frame['class_id']==c_id][metric].index
                history_frame.loc[r, metric] = v
            history_frame.to_csv(args.project / f'test_result_mean.csv', encoding='utf-8-sig', index=False)

    history_frame = pd.DataFrame({
        'class_id': class_id,
        'class': [''.join(id2cat[x]) for x in class_id],
        '# test data' : num_test_data
    })
    for metric, values in eval_vote.class_scores.items():
        if metric != 'class_id':
            history_frame[metric] = 0
            for c_id, v in zip(eval_vote.class_scores['class_id'], values):
                r = history_frame[history_frame['class_id']==c_id][metric].index
                history_frame.loc[r, metric] = v
            history_frame.to_csv(args.project / f'test_result_vote.csv', encoding='utf-8-sig', index=False)

    # save valid predictions
    pred_frame = pd.DataFrame({
        "doc": test_loader.dataset.doc,
        "category": list(map(lambda x: ''.join(id2cat[x]), test_loader.dataset.label)),
        "pred_mean": list(map(lambda x: ''.join(id2cat[x]), eval_mean.predictions)),
        "pred_vote": list(map(lambda x: ''.join(id2cat[x]), eval_vote.predictions))
    })

    pred_frame.to_csv(args.project / 'predictions.csv', encoding='utf-8-sig', index=False)
    
    
def get_args():
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]
    DATA = FILE.parents[1] / 'data'
    save_dir = increment_path(Path(ROOT) / 'runs'/ 'train' / 'exp')
    
    parser=argparse.ArgumentParser(
        description='Training Disease Recognition in Pet CT')
    parser.add_argument('--root', default=DATA / '1. 실습용자료_final.txt', type=str,
                    help='data format should be txt, sep="|"')
    parser.add_argument('--project', default=save_dir, type=str)
    
    # Data Preprocess
    parser.add_argument('--num-test', default=50000, type=int,
                    help='the number of test data')
    parser.add_argument('--upsample', default='', type=str,
                    help='"shuffle", "uniform", "random"')
    parser.add_argument('--minimum', default=100, type=int,
                    help='(upsample) setting the minimum number of data of each categories')
    parser.add_argument('--num-samples', default=150000, type=int,
                        help='the number of sampled data.'
                           'one sub dataset contains {num-samples} data')
    parser.add_argument('--num-sub-valid', default=50000, type=int,
                        help='the valid ratio of num-samples will be sub valid data')
    parser.add_argument('--sample-dist', default='same', type=str,
                        help='class distributions of sampled sub datasets.'
                            '"same": same as the distribution of the mother dataset'
                            '"random": random distribution not considering mother dataset')
    
    # Data Loader
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N',
                    help='mini-batch size (default: 512)'
                         '[kobert] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                         '[kogpt2] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                         '[kogpt3] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50')
    
    # Model
    parser.add_argument('-m', '--model', default='ensemble', type=str,
                        help='Model to train. Available models are ensemble.')
    parser.add_argument('--n-kobert', default=0, type=int,
                        help='a number of kobert base models')
    parser.add_argument('--n-bert', default=0, type=int,
                        help='a number of kobert base models')
    parser.add_argument('--n-mlbert', default=0, type=int,
                        help='a number of multi-lingual bert base models')
    parser.add_argument('--n-albert', default=0, type=int,
                        help='a number of multi-lingual bert base models')
    parser.add_argument('--n-kobart', default=0, type=int,
                        help='a number of multi-lingual bert base models')
    parser.add_argument('--n-asbart', default=0, type=int,
                        help='a number of multi-lingual bert base models')
    parser.add_argument('--n-kogpt2', default=0, type=int,
                        help='a number of kogpt2 base models')
    parser.add_argument('--n-kogpt3', default=0, type=int,
                        help='a number of kogpt2 base models')
    parser.add_argument('--n-electra', default=0, type=int,
                        help='a number of kogpt2 base models')
    parser.add_argument('--n-funnel', default=0, type=int,
                        help='a number of kogpt2 base models')
    
#     parser.add_argument('--n-hanbert', default=0, type=int,
#                         help='a number of hanbert base models')
#     parser.add_argument('--n-dstbert', default=0, type=int,
#                         help='a number of dstbert base models')
#     parser.add_argument('--n-skobert', default=0, type=int,
#                         help='a number of skobert base models')
    
    # n layers
    parser.add_argument('--dr-rate', default=None, type=float,
                        help='')
    parser.add_argument('--n-layers', default=1, type=int,
                        help='a number of layers to be stacked upen kobert language model.')
    parser.add_argument('--layernorm', action='store_true',
                        help='a number of layers to be stacked upen kobert language model.')
    parser.add_argument('--batchnorm', action='store_true',
                        help='a number of layers to be stacked upen kobert language model.')
    
#     parser.add_argument('--kobert-layers', default=1, type=int,
#                         help='a number of layers to be stacked upen kobert language model.')
#     parser.add_argument('--kogpt2-layers', default=1, type=int,
#                         help='a number of layers to be stacked upen kogpt2 language model.')
#     parser.add_argument('--mlbert-layers', default=1, type=int,
#                         help='a number of layers to be stacked upen mlbert language model.')
#     parser.add_argument('--hanbert-layers', default=1, type=int,
#                         help='a number of layers to be stacked upen hanbert language model.')
#     parser.add_argument('--dstbert-layers', default=1, type=int,
#                         help='a number of layers to be stacked upen dstbert language model.')
#     parser.add_argument('--skobert-layers', default=1, type=int,
#                         help='a number of layers to be stacked upen skobert language model.')
#     # lyaer normalization
#     parser.add_argument('--kobert-ln', action='store_true',
#                         help='use layer normalization in a classifier.')
#     parser.add_argument('--kogpt2-ln', action='store_true',
#                         help='use layer normalization in a classifier.')
#     parser.add_argument('--mlbert-ln', action='store_true',
#                         help='use layer normalization in a classifier.')
#     parser.add_argument('--hanbert-ln', action='store_true',
#                         help='use layer normalization in a classifier.')
#     parser.add_argument('--dstbert-ln', action='store_true',
#                         help='use layer normalization in a classifier.')
#     parser.add_argument('--skobert-ln', action='store_true',
#                         help='use layer normalization in a classifier.')
#     # batch normalization
#     parser.add_argument('--kobert-bn', action='store_true',
#                         help='use batch normalization in a classifier.')
#     parser.add_argument('--kogpt2-bn', action='store_true',
#                         help='use batch normalization in a classifier.')
#     parser.add_argument('--mlbert-bn', action='store_true',
#                         help='use batch normalization in a classifier.')
#     parser.add_argument('--hanbert-bn', action='store_true',
#                         help='use batch normalization in a classifier.')
#     parser.add_argument('--dstbert-bn', action='store_true',
#                         help='use batch normalization in a classifier.')
#     parser.add_argument('--skobert-bn', action='store_true',
#                         help='use batch normalization in a classifier.')
    
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--patience', default=10, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
#     parser.add_argument('--additional-epochs', default=5, type=int, metavar='N',
#                         help='additional train epochs')              
#     parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                             help='path to latest checkpoint (default: none)')
                
    # Loss
    parser.add_argument('--loss', default='FCE', type=str,
                        help='Loss function. Availabel loss functions are ["CE", "FCE", "ICE", "mix"]. default is Focal Cross Entropy(FCE).')

    # Learning rate
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-scheduler', default='cosine_with_restarts',
                        type=str, help='Available schedulers are "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup')
    parser.add_argument('--warmup-ratio', default=0.02, type=int, help='lr-scheduler')

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='default is AdamW')
    parser.add_argument('--beta1', type=float, default=0.9,
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
    args.estimators = sum([args.n_kobert,args.n_bert,args.n_mlbert,args.n_albert,
                           args.n_kobart,args.n_asbart,
                           args.n_kogpt2,args.n_kogpt3,
                           args.n_electra,args.n_funnel])
    num_base_models = OrderedDict()
    acc_num=0
    for model_type, num in zip(['kobert', 'bert', 'mlbert', 'albert',
                                'kobart', 'asbart',
                                'kogpt2', 'kogpt3',
                                'electra', 'funnel'],
                               [args.n_kobert,args.n_bert,args.n_mlbert,args.n_albert,
                                args.n_kobart,args.n_asbart,
                                args.n_kogpt2,args.n_kogpt3,
                                args.n_electra,args.n_funnel]):
        acc_num+=num
        num_base_models[model_type] = (num, acc_num)
    args.num_base_models = num_base_models
    for model_type, (num, acm_num) in args.num_base_models.items():
        print(model_type, (num, acm_num))
        
        
#     args.num_base_models = OrderedDict({name: num for name, num in zip(['kobert', 'bert', 'mlbert', 'albert',
#                                                                         'kobart', 'asbart',
#                                                                         'kogpt2', 'kogpt3',
#                                                                         'electra', 'funnel'],
#                                                                        [args.n_kobert,args.n_bert,args.n_mlbert,args.n_albert,
#                                                                         args.n_kobart,args.n_asbart,
#                                                                         args.n_kogpt2,args.n_kogpt3,
#                                                                         args.n_electra,args.n_funnel])})
    assert args.estimators > 0, 'need at least one base model'
    return args
    
if __name__=='__main__':
    args = get_args()
    main(args)