#시도 
import pandas as pd
import numpy as np
from typing import Optional, Union, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer

def num2code(num, digits=None):
    """ int타입의 데이터를 일정한 자릿수(digits)의 코드값으로 변환 """
    num = str(num)
    code = '0'*(digits-len(num)) + num
    return code

def preprocess(frame, clean_fn=None):
        """
        input
            - frame : '대분류', '중분류', '소분류', text_obj', 'text_mthd', 'text_deal' 컬럼을 포함하는 데이터프레임
        Result
            - doc[List] : 데이터 별로 'text_obj', 'text_mthd', 'text_deal'을 연결한 list
            - label[List] : 데이터 별로 category id를 부여한 list
            - cat2id[Dict] : 소분류(str)를 key로, id(int)를 값으로 가지는 사전 객체
            - id2cat[Dict] : id(int)를 key로, tuple(대분류, 중분류, 소분류)를 값으로 가지는 사전 객체
        """
        
        # 1. doc
        frame[['text_obj', 'text_mthd', 'text_deal']] = frame[['text_obj', 'text_mthd', 'text_deal']].fillna('')
        if clean_fn: # text 전처리 함수
            frame[['text_obj', 'text_mthd', 'text_deal']] = frame[['text_obj', 'text_mthd', 'text_deal']].apply(clean_fn)
        doc: pd.Series = frame[['text_obj', 'text_mthd', 'text_deal']].apply(lambda x: ' '.join(x), axis=1)
        
        # 2. label
        frame['digit_2'] = frame['digit_2'].apply(lambda x: num2code(x, 2)) # 중분류를 2자리 코드값으로 변환
        frame['digit_3'] = frame['digit_3'].apply(lambda x: num2code(x, 3)) # 소분류를 3자리 코드값으로 변환
        
            # 훈련 데이터에 있는 유니크한 분류값
        unique_categories = frame[['digit_1', 'digit_2', 'digit_3']].drop_duplicates().apply(lambda x: tuple(x), axis=1).sort_values().to_list()
        
            # 카테고리 별 아이디 부여, 아이디로 카테고리 값 반환
        cat2id, id2cat = {}, {}
        for i, cat in enumerate(unique_categories):
            cat2id[cat[-1]] = i
            id2cat[i] = cat
            
        label: pd.Series = frame['digit_3'].apply(lambda x: cat2id[x])
        return doc.to_list(), label.to_list(), cat2id, id2cat
    

def train_test_split(doc, label, test_ratio=0.1, seed=42):
    """
    temp_df 
    """
    temp_df = pd.DataFrame({'doc': doc, 'label': label})
    train, test = pd.DataFrame(), pd.DataFrame()
    for l in temp_df.label.unique():
        temp_df_l = temp_df[temp_df.label==l].sample(frac=1, random_state=seed).reset_index(drop=True)
        if len(temp_df_l) < 3:
            train = pd.concat([train, temp_df_l])
        else:
            if len(temp_df_l) <= 5:
                slice_idx = 1
            elif len(temp_df_l) <= 10:
                slice_idx = 2
            elif len(temp_df_l) < 100:
                a, b = 8/90, 10/9
                slice_idx = int(a*len(temp_df_l) + b)
            else: # len(ttemp) >= 100
                slice_idx = int(len(temp_df_l)*test_ratio)
            train = pd.concat([train, temp_df_l.iloc[slice_idx:, :]])
            test = pd.concat([test, temp_df_l.iloc[:slice_idx, :]])
    return train['doc'].to_list(), train['label'].to_list(), test['doc'].to_list(), test['label'].to_list()
            
class EnsembleDataset(Dataset):
    def __init__(self, doc, label, kobert_tokenizer, max_len=50, padding='max_length', truncation=True):
        super(EnsembleDataset, self).__init__()
        self.doc = doc
        
        self.kobert_tokenizer = kobert_tokenizer
        self.kogpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
                                'skt/kogpt2-base-v2',
                                bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                pad_token='<pad>', mask_token='<mask>')
        self.kobert_tokenized = [self.kobert_tokenizer([d]) for d in doc] # numpy.array
        self.kogpt_toknized = self.kogpt_tokenizer(doc, padding=padding, max_length=max_len, truncation=truncation, return_tensors='pt')
        self.label = label
        
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = np.zeros_like(token_ids)
        attention_mask[:valid_length] = 1
        return attention_mask
    
    def __getitem__(self, idx):
        kobert_token_ids = self.kobert_tokenized[idx][0]
        kobert_valid_length = self.kobert_tokenized[idx][1]
        kobert_token_type_ids = self.kobert_tokenized[idx][2]
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
        
class KOGPT3ClassifyDataset(Dataset):
    def __init__(self, doc: List, label: List, max_len=50, padding='max_length', truncation=True):
        super(KOGPT3ClassifyDataset, self).__init__()
        self.doc = doc
        self.label = label
        self.max_len = max_len
        self.padding = padding
        self.truncation = truncation
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path='kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
            bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
        )
        self.tokenized = self.tokenizer(doc, padding='max_length', max_length=50, truncation=True, return_tensors='pt')
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return (self.tokenized.input_ids[idx],
                self.tokenized.attention_mask[idx],
                self.tokenized.token_type_ids[idx],
                self.label[idx])
    
class KOGPT2ClassifyDataset(Dataset):
    def __init__(self, doc: List, label: List, max_len=50, padding='max_length', truncation=True):
        super(KOGPT2ClassifyDataset, self).__init__()
        self.doc = doc
        self.label = label
        self.max_len = max_len
        self.padding = padding
        self.truncation = truncation
        
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                        'skt/kogpt2-base-v2',
                        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                        pad_token='<pad>', mask_token='<mask>')
    
        self.tokenized = self.tokenizer(doc, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return (self.tokenized.input_ids[idx],
                self.tokenized.attention_mask[idx],
                self.tokenized.token_type_ids[idx],
                self.label[idx])
