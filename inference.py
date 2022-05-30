import os
import argparse
import json
import pandas as pd
from tqdm import tqdm
import pathlib
from pathlib import Path

from network import *
from dataset import *
from load import *
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
    # data load
    try:
        data = pd.read_csv(args.root, sep='|', encoding='cp949')
    except:
        data = pd.read_csv(args.root, sep='|', encoding='utf-8-sig')
    doc_id, doc = data.index.tolist(), data[['text_obj', 'text_mthd', 'text_deal']].fillna('')
    doc = doc.apply(lambda x: ' '.join(x), axis=1).tolist()
    
    checkpoint = torch.load(os.path.join(args.exp_path, 'weights/best_loss.pth.tar'), map_location=args.device)
    with open(os.path.join(args.exp_path, 'config.json'), 'r') as f:
        checkpoint_args = json.load(f)
    with open(os.path.join(args.exp_path, 'id2cat.json'), 'r') as f:
        id2cat = json.load(f)
    num_classes = len(id2cat)
    
    backbone, tokenizer = load_backbone_tokenizer()
    backbone, tokenizer = load_backbone_tokenizer(checkpoint_args['model'], max_len=checkpoint_args['max_len'])
    model = load_model(checkpoint_args['model'], backbone, num_classes, num_layers=checkpoint_args['n_layers'],
                       dr_rate=checkpoint_args['dr_rate'], bias=checkpoint_args['bias_off'],
                       batchnorm=checkpoint_args['batchnorm'], layernorm=checkpoint_args['layernorm'])
    
    dataset = load_dataset(checkpoint_args['model'], doc, doc_id, tokenizer, max_len=checkpoint_args['max_len'])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=False)
    
    model = model.to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    with torch.no_grad():
        for inputs, doc_id in tqdm(dataloader, total=len(dataloader)):
            for k, v in inputs.items():
                inputs[k] = v.to(args.device, non_blocking=True)
            output = model(**inputs)
            output = torch.argmax(output, 1)
            
            output_cat = list(map(lambda x: id2cat[str(x)], output.cpu().tolist()))
            for r, digit in zip(doc_id.tolist(), output_cat):
                if isinstance(digit, list):
                    data.loc[r, ['digit_1', 'digit_2', 'digit_3']] = digit
                else:
                    data.loc[r, ['digit_1', 'digit_2', 'digit_3']] = [digit[0], digit[1:3], digit[3:]]
    
    create_directory(args.project)
    # save config
    with open(args.project / 'config.json', 'w', encoding='cp949') as f:
        arg_dict = {k: (str(v) if type(v)==pathlib.PosixPath else v) for k, v in args.__dict__.items()}
        json.dump(arg_dict, f, indent=4)
    data.to_csv(args.project / 'submit.csv', encoding='utf-8-sig')
                
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
    parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N',
                        help='mini-batch size (default: 16)'
                             '[kobert] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                             '[kogpt2] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                             '[kogpt3] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50')
    
    parser.add_argument('--device', default='cuda', type=str)
    args=parser.parse_args()
    
    main(args)