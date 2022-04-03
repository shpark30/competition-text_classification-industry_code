import os
import logging
import json
import statistics
import csv
from copy import copy
from typing import Optional, Union, List
import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm
from collections import defaultdict, Counter
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

from transformers import BertModel, GPTJForCausalLM, GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from transformers.optimization import get_scheduler

import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from loss import CrossEntropy, FocalCrossEntropy, label2target
from utils import create_logger, create_directory, increment_path, save_performance_graph, num2code
from dataset import train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false" # https://github.com/pytorch/pytorch/issues/57273
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
os.environ["CUDA_LAUNCH_BLOCKING"] = '0,1'


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
    data = pd.read_csv(args.root, sep='|', encoding='utf-8')
    
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
    train, test = train_test_split(data, test_ratio=test_ratio, seed=args.seed)
    logger.info(f'# base model train data: {args.num_samples}')
    logger.info('# ensemble test data: {len(test)}')
    
    # Sub Sampling
    def bootstrap(data, estimators, dist='same', num_samples=100000, min_cat_data=2, seed=42):
        def _sampling_same(data, seed, total_samples, min_cat_data):
            n = min_cat_data
            e = total_samples/len(data)
            random.seed(seed)
            sub_data = pd.DataFrame()
            for lb in data['label'].unique().tolist():
                data_lb = data[data['label']==lb].copy()
                seed_lb = random.randint(0, 1000)
                if len(data_lb) <= n:
                    n_sampled = len(data_lb)
                else:
                    a, b = (100*e-n)/(100-n), 100*n*(1-e)/(100-n)
                    n_sampled = int(a*len(data_lb) + b)
                sub_data = pd.concat([sub_data, data_lb.sample(n=n_sampled, random_state=seed_lb)])
                sub_data = sub_data.reset_index(drop=True)
            return sub_data
        
        def _sampling_random(data, seed, total_samples, min_cat_data, cut_std=1000):
            dist = data['label'].value_counts().to_frame().reset_index()
            dist.columns = ['label', 'cnt']
            def rand_num(max_range, min_range=3, seed=42):
                random.seed(seed)
                if max_range <= min_range:
                    min_range = max_range
                assert max_range >= min_range
                return random.randint(min_range, max_range)
            i = 0
            while True:
                dist['n_samples'] = dist.apply(lambda x: rand_num(x[1], min_cat_data, seed=seed+x[0]+i), axis=1)
                sum_samples = sum(dist['n_samples'])
                i += 1
                if sum_samples < total_samples:
                    continue
                elif sum_samples > total_samples:
                    save_num = total_samples - dist[dist['n_samples'] < cut_std]['n_samples'].sum()
                    to_cut_df_dist = dist[dist['n_samples'] >= cut_std]
                    to_cut_df = pd.DataFrame()
                    for lb, n in zip(*[to_cut_df_dist['label'].tolist(), to_cut_df_dist['n_samples'].tolist()]):
                        to_cut_df = pd.concat([to_cut_df, pd.DataFrame({'label': [lb]*n})])
                    to_cut_df = to_cut_df.reset_index(drop=True)
                    adj_cut_df = to_cut_df.sample(n=save_num, random_state=42)
                    adj_cut_df_dist = adj_cut_df['label'].value_counts().to_frame().reset_index()
                    adj_cut_df_dist.columns = ['label', 'cnt']
                    for lb, n in zip(*[adj_cut_df_dist['label'].tolist(), adj_cut_df_dist['cnt'].tolist()]):
                        r = dist[dist['label']==lb].index[0]
                        dist.loc[r, 'n_samples'] = n
                    assert dist['n_samples'].sum() == total_samples
                    break
                else: # sum_samples == total_samples
                    break
            sub_data = pd.DataFrame()
            for lb in data['label'].unique().tolist():
                n_sampled = int(dist[dist['label']==lb]['n_samples'])
                data_lb = data[data['label']==lb].copy()
                seed_lb = random.randint(0, 1000)
                sub_data = pd.concat([sub_data, data_lb.sample(n=n_sampled, random_state=seed_lb)])
                sub_data = sub_data.reset_index(drop=True)
            return sub_data
        
        sub_data_list = [] # 생성된 서브데이터를 담을 리스트
        print("BootStrapping!")
        for estimator in tqdm(range(args.estimators), total=args.estimators): # args.estimators 만큼 서브데이터 생성
            random.seed(seed*estimator)
            seed_i = random.randint(0, 1000)
            if dist=='same':
                sub_data = _sampling_same(data, seed_i, num_samples, min_cat_data)
            elif dist=='random':
                sub_data = _sampling_random(data, seed_i, num_samples, min_cat_data)
            sub_data_list.append(sub_data)
        print('sub data 수:', len(sub_data_list))
        print('sub data 카테고리 수:',
              [(i, len(sub_data['label'].unique())) for i, sub_data in enumerate(sub_data_list)])
        return sub_data_list
    
    sub_data_list = bootstrap(train, args.estimators,
                              dist=args.sample_dist, num_samples=args.num_samples,
                              seed=args.seed)
    
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
                                                    sub_data, test_ratio=args.sub_valid_ratio, seed=args.seed),
                                              sub_data_list))
    
    # Upsampling
    def upsample_corpus(df, minimum=500, method='uniform', seed=42):
        random.seed(seed)
        labels = df['label'].unique().tolist()
        upsampled = pd.DataFrame()
        for lb in labels:
            temp_df = df[df['label']==lb].copy()
            n = 0
            while True:
                n+=1
                if n==50:
                    import pdb
                    pdb.set_trace()
                if len(temp_df)>=minimum:
                    break

                if method=='random':
                    n_sample = minimum-len(temp_df)
                    sample = temp_df.sample(n=n_sample, replace=True, random_state=seed)
                    not_empty_cond = sample[['text_obj','text_mthd','text_deal']].applymap(
                        lambda x: len(x)!=0).apply(any, axis=1)
                    if not not_empty_cond.all():
                        sample = sample[not_empty_cond].reset_index(drop=True)
                    temp_df = pd.concat([temp_df, sample])
                elif method=='uniform':
                    n_rep = minimum//len(temp_df)
                    n_sample = minimum%len(temp_df)
                    sample = temp_df.sample(n=n_sample, random_state=seed)
                    not_empty_cond = sample[['text_obj','text_mthd','text_deal']].applymap(
                        lambda x: len(x)!=0).apply(any, axis=1)
                    if not not_empty_cond.all():
                        sample = sample[not_empty_cond].reset_index(drop=True)
                    temp_df = pd.concat([temp_df for _ in range(n_rep)]+[sample])
                elif method=='shuffle':
                    s1=random.randrange(0, 1000) # seed 1
                    s2=random.randrange(0, 1000) # seed 2
                    s3=random.randrange(0, 1000) # seed 3
                    n_sample = minimum-len(temp_df)
                    sample = pd.concat([
                        temp_df['text_obj'].sample(n=n_sample, replace=True, 
                                                   random_state=s1).reset_index(drop=True),
                        temp_df['text_mthd'].sample(n=n_sample, replace=True, 
                                                    random_state=s2).reset_index(drop=True),
                        temp_df['text_deal'].sample(n=n_sample, replace=True, 
                                                    random_state=s3).reset_index(drop=True)
                    ], axis=1)
                    sample['label'] = lb
                    sample=sample[['label', 'text_obj','text_mthd','text_deal']]
                    not_empty_cond = sample[['text_obj','text_mthd','text_deal']].applymap(
                        lambda x: len(x)!=0).apply(any, axis=1)
                    if not not_empty_cond.all():
                        sample = sample[not_empty_cond].reset_index(drop=True)
                    temp_df=pd.concat([temp_df, sample])
            upsampled = pd.concat([upsampled, temp_df])
        upsampled = upsampled.reset_index(drop=True)
        return upsampled
    
    logger.info(f'sub train 수(before upsample): {[len(sub_train) for sub_train in sub_train_list]}')
    sub_train_list = list(map(
        lambda sub_train: upsample_corpus(sub_train, minimum=args.minimum, method=args.upsample, seed=args.seed),
        sub_train_list
    ))
    logger.info(f'sub train 수(after upsample): {[len(sub_train) for sub_train in sub_train_list]}')
    
    
    # text 이어붙이기
    def concat_text(data):
        data['text'] = data[['text_obj', 'text_mthd', 'text_deal']].apply(
                lambda text_tuple: ' '.join(text_tuple), axis=1)
        return data

    sub_train_list = [concat_text(sub_train) for sub_train in sub_train_list]
    sub_valid_list = [concat_text(sub_valid) for sub_valid in sub_valid_list]
    test = concat_text(test)
    
    # Data Loaders
    train_loaders = []
    valid_loaders = []
    if args.model == 'ensemble':
        n_bert = int(args.estimators/2)
        n_gpt = args.estimators - n_bert
    elif args.model == 'ensemble-kobert':
        n_bert = args.estimators
        n_gpt = 0
    elif args.model == 'ensemble-kogpt2':
        n_bert = 0
        n_gpt = args.estimators
    else:
        import pdb
        pdb.set_trace()
#         raise f'{args.model} is not available for --model'

    logger.info(f'bert base model {n_bert}개')
    logger.info(f'gpt base model {n_gpt}개')
    
        # Kobert Data Loaders
    class KOBERTClassifyDataset(Dataset):
        def __init__(self, doc, label, tokenizer):
            super(KOBERTClassifyDataset, self).__init__()
            self.doc = doc
            self.tokenizer = tokenizer
            self.tokenized = [self.tokenizer([d]) for d in doc] # numpy.array
            self.label = label

        def gen_attention_mask(self, token_ids, valid_length):
            attention_mask = np.zeros_like(token_ids)
            attention_mask[:valid_length] = 1
            return attention_mask

        def __getitem__(self, i):
            token_ids = self.tokenized[i][0]
            valid_length = self.tokenized[i][1]
            token_type_ids = self.tokenized[i][2]
            attention_mask = self.gen_attention_mask(token_ids, valid_length)
            return (token_ids, # numpy.array
                    attention_mask, # numpy.array
                    token_type_ids, # numpy.array
                    self.label[i]) # int scalar
            # numpy array will be changed to torch.tensor via DataLoader

        def __len__(self):
            return (len(self.label))
    
    bert, bert_vocab = get_pytorch_kobert_model()
    bert_tokenizer_path = get_tokenizer()
    bert_tokenizer = nlp.data.BERTSPTokenizer(bert_tokenizer_path, bert_vocab, lower=False)
    bert_transform = nlp.data.BERTSentenceTransform(
                bert_tokenizer, max_seq_length=args.max_len, pad=True, pair=False) 

    for sub_train, sub_valid in tqdm(zip(sub_train_list[:n_bert], sub_valid_list[:n_bert]),
                                    total=n_bert):
        train_set = KOBERTClassifyDataset(sub_train['text'].tolist(),
                                         sub_train['label'].tolist(),
                                         bert_transform)
        valid_set = KOBERTClassifyDataset(sub_valid['text'].tolist(),
                                         sub_valid['label'].tolist(),
                                         bert_transform)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                                  shuffle=True, pin_memory=False)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.workers,
                                 shuffle=False, pin_memory=False)
        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)
            
        # gpt Data Loaders
    class KOGPT2ClassifyDataset(Dataset):
        def __init__(self, doc, label, tokenizer, max_len):
            super(KOGPT2ClassifyDataset, self).__init__()
            self.doc = doc
            self.label = label
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.tokenized = self.tokenizer(doc, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')

        def __len__(self):
            return len(self.label)

        def __getitem__(self, idx):
            return (self.tokenized.input_ids[idx],
                    self.tokenized.attention_mask[idx],
                    self.tokenized.token_type_ids[idx],
                    self.label[idx])
    
    gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            'skt/kogpt2-base-v2',
            bos_token='</s>', eos_token='</s>', unk_token='<unk>',
            pad_token='<pad>', mask_token='<mask>')
    for sub_train, sub_valid in tqdm(zip(sub_train_list[-n_gpt:], sub_valid_list[-n_gpt:]),
                                    total=n_gpt):
        train_set = KOGPT2ClassifyDataset(sub_train['text'].tolist(),
                                         sub_train['label'].tolist(),
                                         gpt_tokenizer, args.max_len)
        valid_set = KOGPT2ClassifyDataset(sub_valid['text'].tolist(),
                                         sub_valid['label'].tolist(),
                                         gpt_tokenizer, args.max_len)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                                  shuffle=True, pin_memory=False)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.workers,
                                 shuffle=False, pin_memory=False)
        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)
            
            
    # Build Base Models
    class KOBERTClassifier(nn.Module):
        def __init__(self, bert, num_classes, hidden_size = 4026, dr_rate=None, params=None):
            super(KOBERTClassifier, self).__init__()
            self.bert = bert
            self.num_classes = num_classes
            self.dr_rate = dr_rate

    #         self.classifier = nn.Linear(hidden_size , num_classes)
            self.classifier = nn.Sequential(nn.Linear(768, hidden_size, bias=True),
                                            nn.ReLU(),
                                            nn.Linear(hidden_size, num_classes, bias=True))
            if dr_rate:
                self.dropout = nn.Dropout(p=dr_rate)

        def forward(self, token_ids, attention_mask, token_type_ids):
            _, pooler = self.bert(input_ids=token_ids.long(),
                                  token_type_ids=token_type_ids.long(),
                                  attention_mask=attention_mask.float())
            if self.dr_rate:
                pooler = self.dropout(pooler)
            return self.classifier(pooler)

    class KOGPT2Classifier(nn.Module):
        def __init__(self, gpt, num_classes, hidden_size=4026, freeze_gpt=True, dr_rate=None):
            super(KOGPT2Classifier, self).__init__()
            self.gpt = gpt
            self.num_classes = num_classes
            self.hidden_size = hidden_size
            self.freeze_gpt = freeze_gpt
            self.dr_rate = dr_rate

            # classifier
            self.classifier = nn.Sequential(nn.Linear(768, hidden_size, bias=True, dtype=torch.float32),
                                            nn.ReLU(),
                                            nn.Linear(hidden_size, num_classes, bias=True, dtype=torch.float32))

            if self.dr_rate:
                self.dropout = nn.Dropout(p=dr_rate)

        def forward(self, token_ids, attention_mask, token_type_ids):
            # transformer decoder output
            # size : (b, n_dec_seq, n_hidden)
            dec_output = self.gpt.transformer(input_ids=token_ids,
                                      token_type_ids=token_type_ids,
                                      attention_mask = attention_mask)

            # language model output
            # size : (b, n_dec_seq, n_dec_vocab)
            logits_lm = self.gpt.lm_head(dec_output.last_hidden_state)

            # classifier output
            # size : (b, n_hidden)
            dec_outputs = dec_output.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
            # size : (b, num_classes)
            logits_cls = self.classifier(dec_outputs)

    #         return logits_lm[:, :-1, :].contiguous(), logits_cls, dec_output.attentions
            return logits_cls
    
    if n_gpt:
        gpt = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path='skt/kogpt2-base-v2',
            pad_token_id=gpt_tokenizer.eos_token_id, torch_dtype='auto',
            low_cpu_mem_usage=True)

        for child in gpt.children():
            for param in child.parameters():
                param.requires_grad = False
    
    def get_optimizer(optimizer_type, model, lr, betas, weight_decay, eps=1e-08, amsgrad=False):
        if optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        elif optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        elif optimizer_type == 'RAdam':
            optimizer = torch.optim.RAdam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        else:
            raise
        return optimizer

    def get_loss(loss_type, **kwargs):
        if loss_type == 'CE':
            criterion = CrossEntropy(**kwargs)
        elif loss_type == 'FCE':
            criterion = FocalCrossEntropy(**kwargs)
        else:
            raise
        return criterion
    
    # Train!
    base_models= []
    num_classes = len(cat2id)
    for e in range(args.estimators):
        patience = 0 # early stop patience
        model_args = copy(args) # config for {e}th model
        model_args.project = args.project / f'model{e}' # runs/train/exp00/model{e}
        create_directory(model_args.project / 'weights') # runs/train/exp00/model{e}/weights
        logger.info(f'Start Training Model{e}')
        
        # dataset
        train_loader = train_loaders[e]
        valid_loader = valid_loaders[e]
        logger.info(f'# train data: {len(train_loader.dataset)}')
        logger.info(f'# valid data: {len(valid_loader.dataset)}')

        # build model
        if e < n_bert:
            model_args.model = 'kobert'
            model = KOBERTClassifier(bert=bert, num_classes=num_classes)
        else:
            model_args.model = 'kogpt2'
            model = KOGPT2Classifier(gpt=gpt, num_classes=num_classes)
        model = model.to(args.device)

        # save config
        with open(model_args.project / 'config.json', 'w', encoding='utf-8-sig') as f:
            arg_dict = {k: (str(v) if type(v)==pathlib.PosixPath else v) for k, v in model_args.__dict__.items()}
            json.dump(arg_dict, f, indent=4)

        # optimizer
        optimizer = get_optimizer(args.optimizer, model, args.lr,
                                  (args.beta1, args.beta2), args.weight_decay, eps=1e-08, amsgrad=False)

        # lr scheduler
        t_total = len(train_loader) * args.epochs
        scheduler = get_scheduler(name=args.lr_scheduler, optimizer=optimizer, 
                                  num_warmup_steps=args.warmup_step, num_training_steps=t_total)\

        criterion = get_loss(args.loss)

        best_acc = None
        best_loss = None
        for epoch in range(args.epochs):
            # train
            train_loss = 0
            model.train()
            for (input_ids, attention_mask, token_type_ids, label) in tqdm(train_loader, total=len(train_loader)):
                input_ids = input_ids.to(args.device, non_blocking=True)
                attention_mask = attention_mask.to(args.device, non_blocking=True)
                token_type_ids = token_type_ids.to(args.device, non_blocking=True)

                # forward propagation
                output = model(input_ids, attention_mask, token_type_ids)
                target = label2target(output, label).to(args.device, non_blocking=True)
                loss = criterion(output, target)

                # backward propagation
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                train_loss += float(loss)*len(label)
            train_loss /= len(train_loader.dataset)

            # validation
            valid_loss = 0
            class_scores = defaultdict(list)
            predictions = []
            valid_confusion_matrix = np.zeros((model.num_classes, model.num_classes), dtype=np.int64)

            model.eval()
            with torch.no_grad():
                for (input_ids, attention_mask, token_type_ids, label) in tqdm(valid_loader, total=len(valid_loader)):
                    input_ids = input_ids.to(args.device, non_blocking=True)
                    attention_mask = attention_mask.to(args.device, non_blocking=True)
                    token_type_ids = token_type_ids.to(args.device, non_blocking=True)

                    # forward propagation
                    output = model(input_ids, attention_mask, token_type_ids)
                    target = label2target(output, label).to(args.device, non_blocking=True)
                    loss = criterion(output, target)
                    valid_loss += float(loss)*len(label)


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

            # logging scores
            macro_pc = statistics.mean(class_scores['precision'])
            macro_rc = statistics.mean(class_scores['recall'])
            macro_f1 = statistics.mean(class_scores['f1score'])
            logger.info(f'Epoch {epoch} Result')
            logger.info(f'\ttrain loss: {train_loss}\tvalid_loss: {valid_loss}')
            logger.info(f'\tacc: {round(acc, 6)}\tpc: {round(macro_pc, 6)}\trc: {round(macro_rc, 6)}\tf1: {round(macro_f1, 6)}')

            # save scores
            if epoch==0:
                # summary.csv
                with open(model_args.project / 'summary.csv', 'w', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow(['epoch', 'train loss', 'valid loss', 'accuracy', 'precision', 'recall', 'f1score'])
                # base frame for precisions, recalls and f1scores
                class_id = list(set(train_loader.dataset.label))
                num_train_data, num_valid_data = [0] * len(class_id), [0] * len(class_id)
                for c_id, n in dict(Counter(train_loader.dataset.label)).items():
                    num_train_data[c_id] = n
                for c_id, n in dict(Counter(valid_loader.dataset.label)).items():
                    num_valid_data[c_id] = n
                history_frame = defaultdict(lambda: pd.DataFrame({
                    'class_id': class_id,
                    'class': list(map(lambda x: ''.join(id2cat[x]), class_id)),
                    '# train data' : num_train_data,
                    '# valid data' : num_valid_data
                }))

            # add new line to summary.csv
            with open(model_args.project / 'summary.csv', 'a', newline='') as f:
                wr = csv.writer(f)
                wr.writerow([epoch, train_loss, valid_loss, acc, macro_pc, macro_rc, macro_f1])

            # add new column(epoch) to precision.csv, recall.csv and f1score.csv
            for metric, values in class_scores.items():
                if metric != 'class_id':
                    history_frame[metric][f'epoch {epoch}'] = 0
                    for c_id, v in zip(class_scores['class_id'], values):
                        r = history_frame[metric][history_frame[metric]['class_id']==c_id][f'epoch {epoch}'].index
                        history_frame[metric].loc[r, f'epoch {epoch}'] = v
                    history_frame[metric].to_csv(model_args.project / f'{metric}.csv', encoding='utf-8-sig', index=False)

            # save performance graph
            save_performance_graph(model_args.project / 'summary.csv', model_args.project / 'performance.png')

            # model save
            epoch_score = acc
            torch.save({'state_dict': model.classifier.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best acc': acc if best_acc is None or acc > best_acc else best_acc,
                        'best loss': valid_loss if best_loss is None or valid_loss <= best_loss else best_loss,
                        'epoch': epoch},
                       model_args.project / 'weights' / 'checkpoint.pth.tar')

            if best_acc is None or acc > best_acc: 
                logger.info(f'Validation accuracy got better {best_acc} --> {acc}.  Saving model ...')
                shutil.copyfile(model_args.project / 'weights' / 'checkpoint.pth.tar',
                                model_args.project / 'weights' / 'best_acc.pth.tar')
                best_acc = acc

                # save valid predictions
                pred_frame = pd.DataFrame({
                    "doc": valid_loader.dataset.doc,
                    "category": list(map(lambda x: ''.join(id2cat[x]), valid_loader.dataset.label)),
                    "predictions": list(map(lambda x: ''.join(id2cat[x]), predictions))
                })
                pred_frame.to_csv(model_args.project / 'best_acc_predictions.csv', encoding='utf-8-sig', index=False)

            if best_loss is None or valid_loss <= best_loss:
                logger.info(f'Validation loss got better {best_loss} --> {valid_loss}.  Saving model ...')
                shutil.copyfile(model_args.project / 'weights' / 'checkpoint.pth.tar',
                                model_args.project / 'weights' / 'best_loss.pth.tar')
                best_loss = valid_loss

                # save valid predictions
                pred_frame = pd.DataFrame({
                    "doc": valid_loader.dataset.doc,
                    "category": list(map(lambda x: ''.join(id2cat[x]), valid_loader.dataset.label)),
                    "predictions": list(map(lambda x: ''.join(id2cat[x]), predictions))
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
            del class_scores, predictions
        base_models.append(copy(model.classifier))
        del history_frame
        
    # Build Ensemble Model
    class EnsembleClassifier(nn.Module):
        def __init__(self, num_classes,
                     bert=None,
                     gpt=None,
                     bert_classifiers=[],
                     gpt_classifiers=[],
                     aggregation='mean_softmax',
                     stacking=None
                    ):
            super(EnsembleClassifier, self).__init__()
            assert (bert is not None) == (len(bert_classifiers)!=0),\
                'expect both of bert and bert_classifiers, but get one of them'
            assert (gpt is not None) == (len(gpt_classifiers)!=0),\
                'expect both of gpt and gpt_classifiers, but get one of them'
            self.num_classes=num_classes
            self.bert=bert
            self.gpt=gpt
            self.bert_classifiers=bert_classifiers
            self.gpt_classifiers=gpt_classifiers
            self.aggregation=aggregation
            if aggregation=='mean_softmax':
                self.aggregation_fn=self.mean_softmax
            elif aggregation=='sum_argmax':
                self.aggregation_fn=self.sum_argmax

            for child in self.bert.children():
                for param in child.parameters():
                    param.requires_grad = False

            for child in self.gpt.children():
                for param in child.parameters():
                    param.requires_grad = False

        def mean_softmax(self, tensor, dim=1):
            return tensor.mean(dim)

        def sum_argmax(self, tensor, dim=1):
            max_idx = torch.argmax(tensor, dim, keepdim=True)
            one_hot = torch.FloatTensor(tensor.shape)
            one_hot.zero_()
            one_hot.scatter_(dim, max_idx, 1)
            return one_hot


        def forward(self, token_ids, attention_mask, token_type_ids):
            output = []
            # bert output
            if self.bert:
                _, pooler = self.bert(input_ids=token_ids.long()[:,0].clone(),
                                      token_type_ids=token_type_ids.long()[:,0].clone(),
                                      attention_mask=attention_mask.float()[:,0].clone())
                output += [
                    F.softmax(classifier(pooler), dim=1) for classifier in self.bert_classifiers
                ]

            if self.gpt:
                dec_output = self.gpt.transformer(input_ids=token_ids[:,1],
                                          token_type_ids=token_type_ids[:,1],
                                          attention_mask = attention_mask[:,1])
                dec_outputs = dec_output.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
                output +=  [
                    F.softmax(classifier(pooler), dim=1) for classifier in self.gpt_classifiers
                ]

            output = torch.cat([op.unsqueeze(1) for op in output], dim=1)
            aggregated = self.aggregation_fn(output)
            return aggregated

    bert_classifiers = base_models[:n_bert]
    gpt_classfiers = base_models[n_bert:]
    ensemble = EnsembleClassifier(num_classes,
                                 bert=bert if n_bert else None,
                                 gpt=gpt if n_gpt else None,
                                 bert_classifiers=bert_classifiers,
                                 gpt_classifiers=gpt_classifiers,
                                 aggregation='mean_softmax',
                                 stacking=None)
    
    # Ensemble Data Loader
    class EnsembleDataset(Dataset):
        def __init__(self, doc, label, kobert_tokenizer=None, kogpt_tokenizer=None,
                     max_len=50, padding='max_length', truncation=True):
            super(EnsembleDataset, self).__init__()
    #         assert (kobert_tokenizer==None) and (kogpt_tokenizer==None),\
    #                 'expect at least one of kobert and kogpt tokenizer, but get neither'
            self.doc = doc
            self.label = label
            self.kobert_tokenizer = kobert_tokenizer
            self.kogpt_tokenizer = kogpt_tokenizer
            self.kobert_tokenized = [self.kobert_tokenizer([d]) for d in doc]
            self.kogpt_toknized = self.kogpt_tokenizer(doc, padding=padding, max_length=max_len, truncation=truncation, return_tensors='pt')

        def gen_attention_mask(self, token_ids, valid_length):
            attention_mask = np.zeros_like(token_ids)
            attention_mask[:valid_length] = 1
            return attention_mask

        def __getitem__(self, idx):
            kobert_token_ids, kobert_valid_length, kobert_token_type_ids = self.kobert_tokenized[idx]
            kobert_attention_mask = self.gen_attention_mask(kobert_token_ids, kobert_valid_length)

            kogpt_token_ids = self.kogpt_toknized.input_ids[idx]
            kogpt_attention_mask = self.kogpt_toknized.attention_mask[idx]
            kogpt_token_type_ids = self.kogpt_toknized.token_type_ids[idx]

            return (torch.cat([torch.from_numpy(kobert_token_ids).unsqueeze(0), kogpt_token_ids.unsqueeze(0)], dim=0), # token_ids
                    torch.cat([torch.from_numpy(kobert_attention_mask).unsqueeze(0), kogpt_attention_mask.unsqueeze(0)], dim=0), # attention_mask
                    torch.cat([torch.from_numpy(kobert_token_type_ids).unsqueeze(0), kogpt_token_type_ids.unsqueeze(0)], dim=0), # token_type_ids
                    self.label[idx]) # int scalar

        def __len__(self):
            return (len(self.label))

    test_set = EnsembleDataset(test['text'].tolist(), test['label'].tolist(),
                              kobert_tokenizer=bert_transform, kogpt_tokenizer=gpt_tokenizer,
                              max_len=args.max_len)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers,
                             shuffle=False, pin_memory=False)
        
    
    # Test Ensemble
    logger.info('Start Test Ensemble')

    ensemble = ensemble.to(args.device)
    ensemble.eval()
    test_loss = 0
    class_scores = defaultdict(list)
    predictions = []
    test_confusion_matrix = np.zeros((ensemble.num_classes, ensemble.num_classes), dtype=np.int64)

    ensemble.eval()
    with torch.no_grad():
        for (input_ids, attention_mask, token_type_ids, label) in tqdm(test_loader, total=len(test_loader)):
            input_ids = input_ids.to(args.device, non_blocking=True)
            attention_mask = attention_mask.to(args.device, non_blocking=True)
            token_type_ids = token_type_ids.to(args.device, non_blocking=True)

            # forward propagation
            output = ensemble(input_ids, attention_mask, token_type_ids)
            target = label2target(output, label).to(args.device, non_blocking=True)
            loss = criterion(output, target)
            test_loss += float(loss)*len(label)


            # get confusion matrix
            pred = torch.argmax(output, 1).cpu()
            test_confusion_matrix += confusion_matrix(label, pred, labels=list(range(ensemble.num_classes)))
            predictions += pred.tolist()

        test_loss /= len(test_loader.dataset)
        acc = np.diagonal(test_confusion_matrix).sum() / test_confusion_matrix.sum()
        for c in range(len(test_confusion_matrix)):
            num_pred = test_confusion_matrix[:, c].sum()
            num_true = test_confusion_matrix[c].sum()
            TP = test_confusion_matrix[c, c]
            FP = num_true - TP
            FN = num_pred - TP
            PC = TP/num_pred if num_pred != 0 else 0 # TP / (TP+FP)
            RC = TP/num_true if num_true != 0 else 0  # TP / (TP+FN)
            F1 = 2 * PC * RC / (PC + RC) if PC + RC != 0 else 0 # (2 * PC * RC) / (PC + RC)
            class_scores['class_id'].append(c)
            class_scores['precision'].append(PC)
            class_scores['recall'].append(RC)
            class_scores['f1score'].append(F1)

    # logging scores
    macro_pc = statistics.mean(class_scores['precision'])
    macro_rc = statistics.mean(class_scores['recall'])
    macro_f1 = statistics.mean(class_scores['f1score'])
    logger.info(f'\ttest_loss: {valid_loss}')
    logger.info(f'\tacc: {round(acc, 6)}\tpc: {round(macro_pc, 6)}\trc: {round(macro_rc, 6)}\tf1: {round(macro_f1, 6)}')

    # summary.csv
    with open(args.project / 'test_result.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['test_loss', 'accuracy', 'precision', 'recall', 'f1score'])
    # base frame for precisions, recalls and f1scores
    class_id = list(set(train_loader.dataset.label))
    num_test_data = [0] * len(class_id)
    for c_id, n in dict(Counter(test_loader.dataset.label)).items():
        num_test_data[c_id] = n
    history_frame = pd.DataFrame({
        'class_id': class_id,
        'class': list(map(lambda x: ''.join(id2cat[x]), class_id)),
        '# test data' : num_test_data
    })

    # add new line to summary.csv
    with open(args.project / 'test_result.csv', 'a', newline='') as f:
        wr = csv.writer(f)
        wr.writerow([test_loss, acc, macro_pc, macro_rc, macro_f1])

    # add new column(epoch) to precision.csv, recall.csv and f1score.csv
    for metric, values in class_scores.items():
        if metric != 'class_id':
            history_frame[metric] = 0
            for c_id, v in zip(class_scores['class_id'], values):
                r = history_frame[history_frame['class_id']==c_id][metric].index
                history_frame[metric].loc[r, metric] = v
            history_fram.to_csv(args.project / f'test_result_verbose.csv', encoding='utf-8-sig', index=False)

    # save valid predictions
    pred_frame = pd.DataFrame({
        "doc": test_loader.dataset.doc,
        "category": list(map(lambda x: ''.join(id2cat[x]), test_loader.dataset.label)),
        "predictions": list(map(lambda x: ''.join(id2cat[x]), predictions))
    })
    pred_frame.to_csv(args.project / 'predictions.csv', encoding='utf-8-sig', index=False)
    
    
def get_args():
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]
    DATA = FILE.parents[1] / 'data'
    save_dir = increment_path(Path(ROOT) / 'runs'/ 'train' / 'exp')
    
    parser=argparse.ArgumentParser(
        description='Training Disease Recognition in Pet CT')
    parser.add_argument('--root', default=DATA / '1. 실습용자료_hsp2.txt', type=str,
                    help='data format should be txt, sep="|"')
    parser.add_argument('--project', default=save_dir, type=str)
    
    # Data
    parser.add_argument('--num-test', default=50000, type=int,
                    help='the number of test data')
    parser.add_argument('--upsample', default='shuffle', type=str,
                    help='"shuffle", "reproduce", "random"')
    parser.add_argument('--minimum', default=100, type=int,
                    help='(upsample) setting the minimum number of data of each categories')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N',
                    help='mini-batch size (default: 512)'
                         '[kobert] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                         '[kogpt2] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                         '[kogpt3] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50')

    parser.add_argument('-m', '--model', default='ensemble', type=str,
                        help='Model to train. Available models are ["ensemble", "ensemble-kobert", "ensemble-kogpt2"].'
                            'default is "ensemble".')
    parser.add_argument('--estimators', default=10, type=int,
                        help='a number of base models')
    parser.add_argument('--num-samples', default=150000, type=int,
                        help='the number of sampled data.'
                           'one sub dataset contains {num-samples} data')
    parser.add_argument('--sub-valid-ratio', default=0.1, type=float,
                        help='the valid ratio of num-samples will be sub valid data')
    parser.add_argument('--sample-dist', default='same', type=str,
                        help='class distributions of sampled sub datasets.'
                            '"same": same as the distribution of the mother dataset'
                            '"random": random distribution not considering mother dataset')
    
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
                        help='Loss function. Availabel loss functions are . default is Focal Cross Entropy(FCE).')

    # Learning rate
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float,
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
    return args
    
if __name__=='__main__':
    args = get_args()
    main(args)