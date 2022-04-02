import os
import argparse
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from network import *
from dataset import *
from utils import create_directory, increment_path

import torch
from torch.utils.data import DataLoader

import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

os.environ["TOKENIZERS_PARALLELISM"] = "false" # https://github.com/pytorch/pytorch/issues/57273
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(torch.cuda.device_count())))
os.environ["CUDA_LAUNCH_BLOCKING"] = ",".join(map(str, range(torch.cuda.device_count())))

def main(args):
    data = pd.read_csv(args.root, sep='|', encoding='cp949')
    doc_id, doc = data.index.tolist(), data[['text_obj', 'text_mthd', 'text_deal']].fillna('')
    doc = doc.apply(lambda x: ' '.join(x), axis=1).tolist()
    
    checkpoint = torch.load(os.path.join(args.exp_path, 'weights/additional2.pth.tar'), map_location=args.device)
    with open(os.path.join(args.exp_path, 'config.json'), 'r') as f:
        checkpoint_args = json.load(f)
    with open(os.path.join(args.exp_path, 'id2cat.json'), 'r') as f:
        id2cat = json.load(f)
    
    if checkpoint_args['model'] == 'kobert':
        kobert, vocab = get_pytorch_kobert_model()
        tokenizer_path = get_tokenizer()
        tokenizer = nlp.data.BERTSPTokenizer(tokenizer_path, vocab, lower=False)
        transform = nlp.data.BERTSentenceTransform(
                    tokenizer, max_seq_length=checkpoint_args['max_len'], pad=True, pair=False) 
        model = KOBERTClassifier(bert=kobert, num_classes=len(id2cat))
        model = model.to(args.device)
        dataset = KOBERTClassifyDataset(doc, doc_id, transform)
    elif checkpoint_args['model'] == 'kogpt2':
        dataset = KOGPT2ClassifyDataset(doc, doc_id, max_len=checkpoint_args['max_len'], padding='max_length', truncation=True)
        model = KOGPT2Classifier(num_classes=len(id2cat), pad_token_id = dataset.tokenizer.eos_token_id)
    elif checkpoint_args['model'] == 'kogpt3':
        dataset = KOGPT3ClassifyDataset(doc, doc_id, max_len=checkpoint_args['max_len'], padding='max_length', truncation=True)
        model = KOGPT3Classifier(num_classes=len(id2cat), pad_token_id = dataset.tokenizer.eos_token_id)
    else:
        raise
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=False)
    
    model = model.to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, doc_id in tqdm(dataloader, total=len(dataloader)):
            input_ids = input_ids.to(args.device, non_blocking=True)
            attention_mask = attention_mask.to(args.device, non_blocking=True)
            token_type_ids = attention_mask.to(args.device, non_blocking=True)
            
            output = model(input_ids, attention_mask, token_type_ids)
            output = torch.argmax(output, 1)
            
            output_cat = list(map(lambda x: id2cat[str(x)], output.cpu().tolist()))
            for r, digit in zip(doc_id.tolist(), output_cat):
                data.loc[r, ['digit_1', 'digit_2', 'digit_3']] = digit
    
    create_directory(args.project)
    data.to_csv(args.project / 'submit.csv', encoding='utf-8')
                
if __name__=='__main__':
    FILE = Path(__file__).resolve()
    DATA = FILE.parents[2]
    ROOT = FILE.parents[0]  # root directory
    save_dir = increment_path(Path(ROOT) / 'runs' / 'inference' / 'exp')
    
    parser=argparse.ArgumentParser(
        description='')

    parser.add_argument('--root', default= DATA / 'data' / '2. 모델개발용자료.txt', type=str,
                        help='data format should be txt, sep="|"')
    parser.add_argument('--exp-path', required=True, type=str,
                       help='path of a directory which contains the "weights" folder and id2cat.json')
    parser.add_argument('--project', default=save_dir, type=str)
    
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                        help='mini-batch size (default: 16)'
                             '[kobert] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                             '[kogpt2] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                             '[kogpt3] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50')
    
    parser.add_argument('--device', default='cuda', type=str)
    args=parser.parse_args()
    
    main(args)