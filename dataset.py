import pandas as pd
import numpy as np
import random
from typing import Optional, Union, List
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

def num2code(num, digits=None):
    """ int타입의 데이터를 일정한 자릿수(digits)의 코드값으로 변환 """
    num = str(num)
    code = '0'*(digits-len(num)) + num
    return code

def concat_text(data):
    data['text'] = data[['text_obj', 'text_mthd', 'text_deal']].apply(
            lambda text_tuple: ' '.join(text_tuple), axis=1)
    return data

def bootstrap(data, estimators, dist='same', num_samples=100000, min_cat_data=2, seed=42):
    """
    dataframe 형식의 데이터를 받아 estimators 만큼의 서브 데이터셋(List)과 샘플링되지 않은 데이터oob(pd.DataFrame)을 return한다.
    sampling 수는 num_samples로 지정하고, sample 방식은 'same', 'random'이 있다.
        - same : 샘플링한 서브 데이터셋과 원본 data의 카테고리별 데이터 수 분포를 같게 한다.
        - random : 원본 data의 카테고리별 데이터 수 분포와 상관없이 카테고리별로 랜덤한 수의 데이터를 추출한다.
    """
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
                a, b = e, n-n*e
                n_sampled = int(a*len(data_lb) + b)
            if n_sampled < 0 :
                import pdb
                pdb.set_trace()
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
            sum_samples = dist['n_samples'].sum()
            i += 1
            if sum_samples < total_samples:
                continue
            elif sum_samples > total_samples:
                cut_samples = dist[dist['n_samples'] < cut_std]['n_samples'].sum()
                if cut_samples <= total_samples:
                    save_num = total_samples - cut_samples
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
                    assert sum_samples != total_samples, f'sum_samples should be same as total_samples, {sum_samples} != {total_samples}.'
                    break
                else: # cut_samples > total_samples
                    cut_std -= 100

                    continue
            else: # sum_samples == total_samples
                break
        if cut_std < 1000:
            print(f'decrease cut_std since cut_samples > total_samples. {1000} -->> {cut_std}')
        sub_data = pd.DataFrame()
        for lb in data['label'].unique().tolist():
            n_sampled = int(dist[dist['label']==lb]['n_samples'])
            data_lb = data[data['label']==lb].copy()
            seed_lb = random.randint(0, 1000)
            sub_data = pd.concat([sub_data, data_lb.sample(n=n_sampled, random_state=seed_lb)])
            sub_data = sub_data.reset_index(drop=True)
        return sub_data

    print("BootStrapping!")
    sub_data_list = [] # 생성된 서브데이터를 담을 리스트
    for estimator in tqdm(range(estimators), total=estimators): # estimators 만큼 서브데이터 생성
        random.seed(seed*estimator)
        seed_i = random.randint(0, 1000)
        if dist=='same':
            sub_data = _sampling_same(data, seed_i, num_samples, min_cat_data)
        elif dist=='random':
            sub_data = _sampling_random(data, seed_i, num_samples, min_cat_data)
        sub_data_list.append(sub_data)
    sampled = pd.concat([sub_data for sub_data in sub_data_list]).drop_duplicates()
    oob = pd.concat([data, sampled]).drop_duplicates(keep=False) # out of bagging
    return sub_data_list, oob

def upsample_corpus(df, minimum=500, method='uniform', seed=42):
    """
    minimum으로 정한 수보다 데이터가 적은 카테고리에 대해 upsample을 수행한다.
    upsample 방식은 uniform, random, shuffle이 있다.
        - uniform : 대상 카테고리의 모든 데이터를 동일한 수만큼 복사한다.
        - random : 대상 카테고리의 데이터를 minimum 수만큼 복원 추출한다.
        
    """
    random.seed(seed)
    labels = df['label'].unique().tolist()
    upsampled = pd.DataFrame()
    for lb in labels:
        temp_df = df[df['label']==lb].copy()
        n = 0
        while True:
            n+=1
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

def train_test_split(frame, test_ratio=0.1, seed=42):
    """
    frame의 분포에 따라 train, test셋으로 나눈다. test셋의 비율은 test_ratio로 정한다.
    """
    train, test = pd.DataFrame(), pd.DataFrame()
    for lb in frame['label'].unique():
        # (lb) 카테고리의 데이터 순서 섞기
        temp_df_l = frame[frame['label']==lb].sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # 데이터 나누기
        if len(temp_df_l) < 3:
            # 데이터 수가 3개 미만인 카테고리는 모두 훈련 데이터로 활용한다. 
            train = pd.concat([train, temp_df_l])
        else:
            # train과 test를 slice할 index 구하기
            if len(temp_df_l) <= 5:
                slice_idx = 1
            elif len(temp_df_l) <= 10:
                slice_idx = 2
            elif len(temp_df_l) < 100:
                a, b = 8/90, 10/9
                slice_idx = int(a*len(temp_df_l) + b)
            else: # len(ttemp) >= 100
                slice_idx = int(len(temp_df_l)*test_ratio)
                
            # train, test 나누기
            train = pd.concat([train, temp_df_l.iloc[slice_idx:, :]])
            test = pd.concat([test, temp_df_l.iloc[:slice_idx, :]])
            
    return train, test

def preprocess(frame, num_test=100000, upsample='', minimum=500, clean_fn=None, target='S', seed=42):
        """
        input
            - frame : 'digit_1', 'digit_2', 'digit_3', text_obj', 'text_mthd', 'text_deal' 컬럼을 포함하는 데이터프레임
        Result
            - train[pd.DataFrame] : 학습 데이터 셋
            - test[pd.DataFrame] : 시험 데이터 셋
            - cat2id[Dict] : 소분류(str)를 key로, id(int)를 값으로 가지는 사전 객체
            - id2cat[Dict] : id(int)를 key로, tuple(대분류, 중분류, 소분류)를 값으로 가지는 사전 객체
        """
        
        # clean text
        if clean_fn: # text 전처리 함수
            frame[['text_obj', 'text_mthd', 'text_deal']] = frame[['text_obj', 'text_mthd', 'text_deal']].apply(clean_fn)
            
        # 결측치 공백('')으로 채우기
        frame = frame.fillna('') 
        
        # 중복제거
        frame = frame.drop_duplicates(['digit_1', 'digit_2', 'digit_3', 'text_obj', 'text_mthd', 'text_deal'], keep='first')
        
        # labeling
        frame['digit_2'] = frame['digit_2'].apply(lambda x: num2code(x, 2)) # 중분류를 2자리 코드값으로 변환
        frame['digit_3'] = frame['digit_3'].apply(lambda x: num2code(x, 3)) # 소분류를 3자리 코드값으로 변환
        frame['digit'] = frame["digit_1"] + frame["digit_2"] + frame["digit_3"]
        unique_digit = frame['digit'].sort_values().drop_duplicates().tolist()
        
        cat2id, id2cat = {}, {}
        for i, cat in enumerate(unique_digit):
            cat2id[cat] = i
            id2cat[i] = cat
        frame['label'] = frame['digit'].apply(lambda x: cat2id[x])
                
        # train-test split
        test_ratio = num_test/len(frame)
        train, test = train_test_split(frame, test_ratio=test_ratio, seed=seed)
        
        # upsample
        if upsample != '':
            train = upsample_corpus(train, minimum=minimum, method=upsample, seed=seed)
        
        # input : 'text' column
        # target: 'label' column
        train['text'] = train[['text_obj', 'text_mthd', 'text_deal']].apply(lambda x: ' '.join(x), axis=1)
        test['text'] = test[['text_obj', 'text_mthd', 'text_deal']].apply(lambda x: ' '.join(x), axis=1)
        train = train[['text', 'label']]
        test = test[['text', 'label']]

        return train, test, cat2id, id2cat

    
class EnsembleDataset(Dataset):
    def __init__(self, doc, label,
                 kobert_tokenizer=None, bert_tokenizer=None, albert_tokenizer=None, mlbert_tokenizer=None,
                 kobart_tokenizer=None, asbart_tokenizer=None,
                 kogpt2_tokenizer=None, kogpt3_tokenizer=None, electra_tokenizer=None, funnel_tokenizer=None,
                 max_len=50, padding='max_length', truncation=True):
        super(EnsembleDataset, self).__init__()
        self.doc = doc
        self.label = label
        self.kobert_tokenizer = kobert_tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.mlbert_tokenizer = mlbert_tokenizer
        self.albert_tokenizer = albert_tokenizer
        self.kobart_tokenizer = kobart_tokenizer
        self.asbart_tokenizer = asbart_tokenizer
        self.kogpt2_tokenizer = kogpt2_tokenizer
        self.kogpt3_tokenizer = kogpt3_tokenizer
        self.electra_tokenizer = electra_tokenizer
        self.funnel_tokenizer = funnel_tokenizer
        
        self.kobert_tokenized = [] if self.kobert_tokenizer is None else [self.kobert_tokenizer([d]) for d in doc]
        self.bert_tokenized   = [] if self.bert_tokenizer is None else self.bert_tokenizer(doc, padding=padding, max_length=max_len, truncation=truncation, return_tensors='pt')
        self.mlbert_tokenized   = [] if self.mlbert_tokenizer is None else self.mlbert_tokenizer(doc, padding=padding, max_length=max_len, truncation=truncation, return_tensors='pt')
        self.albert_tokenized = [] if self.albert_tokenizer is None else self.albert_tokenizer(doc, padding=padding, max_length=max_len, truncation=truncation, return_tensors='pt')
        self.kobart_tokenized = [] if self.kobart_tokenizer is None else self.kobart_tokenizer(doc, padding=padding, max_length=max_len, truncation=truncation, return_tensors='pt')
        self.asbart_tokenized = [] if self.asbart_tokenizer is None else self.asbart_tokenizer(doc, padding=padding, max_length=max_len, truncation=truncation, return_tensors='pt')
        self.kogpt2_tokenized = [] if self.kogpt2_tokenizer is None else self.kogpt2_tokenizer(doc, padding=padding, max_length=max_len, truncation=truncation, return_tensors='pt')
        self.kogpt3_tokenized = [] if self.kogpt3_tokenizer is None else self.kogpt3_tokenizer(doc, padding=padding, max_length=max_len, truncation=truncation, return_tensors='pt')
        self.electra_tokenized= [] if self.electra_tokenizer is None else self.electra_tokenizer(doc, padding=padding, max_length=max_len, truncation=truncation, return_tensors='pt')
        self.funnel_tokenized = [] if self.funnel_tokenizer is None else self.funnel_tokenizer(doc, padding=padding, max_length=max_len, truncation=truncation, return_tensors='pt')
        
    def __len__(self):
        return (len(self.label))
    
    def gen_attention_mask(self, input_ids, valid_length):
        attention_mask = np.zeros_like(input_ids)
        attention_mask[:valid_length] = 1
        return attention_mask

    def __getitem__(self, idx):
        inputs = {}
        if self.kobert_tokenizer:
            kobert_input_ids, kobert_valid_length, kobert_token_type_ids = self.kobert_tokenized[idx]
            kobert_attention_mask = self.gen_attention_mask(kobert_input_ids, kobert_valid_length)
            inputs['kobert'] = torch.stack([torch.from_numpy(kobert_input_ids),  # 50
                                           torch.from_numpy(kobert_attention_mask), 
                                           torch.from_numpy(kobert_token_type_ids)]) 
        if self.bert_tokenizer:
            inputs['bert'] = torch.stack([self.bert_tokenized.input_ids[idx],
                                         self.bert_tokenized.attention_mask[idx],
                                         self.bert_tokenized.token_type_ids[idx]])
        if self.albert_tokenizer:
            inputs['albert'] = torch.stack([self.albert_tokenized.input_ids[idx],
                                           self.albert_tokenized.attention_mask[idx],
                                           self.albert_tokenized.token_type_ids[idx]])
        if self.mlbert_tokenizer:
            inputs['mlbert'] = torch.stack([self.mlbert_tokenized.input_ids[idx],
                                           self.mlbert_tokenized.attention_mask[idx],
                                           self.mlbert_tokenized.token_type_ids[idx]])
        if self.kobart_tokenizer:
            inputs['kobart'] = torch.stack([self.kobart_tokenized.input_ids[idx],
                                           self.kobart_tokenized.attention_mask[idx]])
        if self.asbart_tokenizer:
            inputs['asbart'] = torch.stack([self.asbart_tokenized.input_ids[idx],
                                           self.asbart_tokenized.attention_mask[idx]])
        if self.kogpt2_tokenizer:
            inputs['kogpt2'] = torch.stack([self.kogpt2_tokenized.input_ids[idx],
                                           self.kogpt2_tokenized.attention_mask[idx],
                                           self.kogpt2_tokenized.token_type_ids[idx]])
        if self.kogpt3_tokenizer:
            inputs['kogpt3'] = torch.stack([self.kogpt3_tokenized.input_ids[idx],
                                           self.kogpt3_tokenized.attention_mask[idx],
                                           self.kogpt3_tokenized.token_type_ids[idx]])
        if self.electra_tokenizer:
            inputs['electra'] = torch.stack([self.electra_tokenized.input_ids[idx],
                                           self.electra_tokenized.attention_mask[idx],
                                           self.electra_tokenized.token_type_ids[idx]])
        if self.funnel_tokenizer:
            inputs['funnel'] = torch.stack([self.funnel_tokenized.input_ids[idx],
                                           self.funnel_tokenized.attention_mask[idx],
                                           self.funnel_tokenized.token_type_ids[idx]])
        return inputs, self.label[idx]
        
        
class KobertClassifyDataset(Dataset):
    def __init__(self, doc, label, tokenizer):
        super(KobertClassifyDataset, self).__init__()
        self.doc = doc
        self.tokenizer = tokenizer
        self.tokenized = [self.tokenizer([d]) for d in doc] # numpy.array
        self.label = label
    
    def gen_attention_mask(self, input_ids, valid_length):
        attention_mask = np.zeros_like(input_ids)
        attention_mask[:valid_length] = 1
        return attention_mask
    
    def __getitem__(self, i):
        input_ids = self.tokenized[i][0]
        valid_length = self.tokenized[i][1]
        token_type_ids = self.tokenized[i][2]
        attention_mask = self.gen_attention_mask(input_ids, valid_length)
        return ({'input_ids': input_ids, # numpy.array
                 'attention_mask': attention_mask, # numpy.array
                 'token_type_ids': token_type_ids}, # numpy.array
                self.label[i]) # int scalar

    def __len__(self):
        return (len(self.label))
    
class ClassifyDataset(Dataset):
    def __init__(self, doc: List, label: List, tokenizer, max_len=50, padding='max_length', truncation=True):
        super(ClassifyDataset, self).__init__()
        self.doc = doc
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.padding = padding
        self.truncation = truncation
        self.tokenized = self.tokenizer(doc, padding='max_length', max_length=50, truncation=True, return_tensors='pt')
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return ({'input_ids': self.tokenized.input_ids[idx],
                 'attention_mask': self.tokenized.attention_mask[idx],
                 'token_type_ids': self.tokenized.token_type_ids[idx]},
                 self.label[idx])
    
class BartClassifyDataset(Dataset):
    def __init__(self, doc: List, label: List, tokenizer, max_len=50, padding='max_length', truncation=True):
        super(BartClassifyDataset, self).__init__()
        self.doc = doc
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.padding = padding
        self.truncation = truncation
        self.tokenized = self.tokenizer(doc, padding='max_length', max_length=50, truncation=True, return_tensors='pt')
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return ({'input_ids': self.tokenized.input_ids[idx],
                'attention_mask': self.tokenized.attention_mask[idx]},
                self.label[idx])