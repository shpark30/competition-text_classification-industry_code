import torch
import os
import json
from collections import defaultdict

from network2 import *
from dataset2 import *

from transformers import BertModel, GPT2LMHeadModel, ElectraModel, BertModel, AlbertModel, FunnelModel, BertForSequenceClassification, BartModel
from transformers import PreTrainedTokenizerFast, AutoTokenizer, BertTokenizerFast, ElectraTokenizerFast, FunnelTokenizerFast, BertTokenizer
from asian_bart import AsianBartTokenizer, AsianBartForConditionalGeneration
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobart import get_kobart_tokenizer, get_pytorch_kobart_model

def load_backbone_tokenizer(model_type, max_len=50):
    if model_type=='kobert':
        kobert, vocab = get_pytorch_kobert_model()
        tokenizer_path = get_tokenizer()
        tokenizer = nlp.data.BERTSPTokenizer(tokenizer_path, vocab, lower=False)
        transform = nlp.data.BERTSentenceTransform(
                    tokenizer, max_seq_length=max_len, pad=True, pair=False)
        return kobert, transform
    elif model_type=='mlbert':
        mlbert = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased")
        mlbert = mlbert.bert
        mlbert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        return mlbert, mlbert_tokenizer
    elif model_type=='bert':
        bert_tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
        bert = BertModel.from_pretrained("kykim/bert-kor-base")
        return bert, bert_tokenizer
    elif model_type=='albert':
        albert = AlbertModel.from_pretrained("kykim/albert-kor-base")
        albert_tokenizer = BertTokenizerFast.from_pretrained("kykim/albert-kor-base")
        return albert, albert_tokenizer
    elif model_type=='kobart':
        kobart_tokenizer = get_kobart_tokenizer()
        kobart = BartModel.from_pretrained(get_pytorch_kobart_model())
        return kobart, kobart_tokenizer
    elif model_type=='asbart':
        asbart = AsianBartForConditionalGeneration.from_pretrained("hyunwoongko/asian-bart-ecjk")
        asbart = asbart.model
        asbart_tokenizer = AsianBartTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")
        return asbart, asbart_tokenizer
    elif model_type=='kogpt2':
        kogpt2 = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path='skt/kogpt2-base-v2')
        kogpt2_tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
                                                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                                    pad_token='<pad>', mask_token='<mask>')
        return kogpt2, kogpt2_tokenizer
    elif model_type=='kogpt3':
        kogpt3_tokenizer = BertTokenizerFast.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")
        kogpt3 = GPT2LMHeadModel.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")
        return kogpt3, kogpt3_tokenizer
    elif model_type=='electra':
        electra_tokenizer = ElectraTokenizerFast.from_pretrained("kykim/electra-kor-base")
        electra = ElectraModel.from_pretrained("kykim/electra-kor-base")
        return electra, electra_tokenizer
    elif model_type=='funnel':
        funnel_tokenizer = FunnelTokenizerFast.from_pretrained("kykim/funnel-kor-base")
        funnel = FunnelModel.from_pretrained("kykim/funnel-kor-base")
        return funnel, funnel_tokenizer
    else:
        raise f'invalid model type, {model_type}'
        
def load_model(model_type, backbone, num_classes, num_layers=1, dr_rate=None, bias=True, batchnorm=False, layernorm=False):
    if model_type=='kobert':
        model = KobertClassifier(backbone, num_classes=num_classes, num_layers=num_layers, dr_rate=dr_rate, bias=bias, batchnorm=batchnorm, layernorm=layernorm)
    elif model_type=='mlbert':
        model = BertClassifier(backbone, num_classes=num_classes, num_layers=num_layers, dr_rate=dr_rate, bias=bias, batchnorm=batchnorm, layernorm=layernorm)
    elif model_type=='bert':
        model = BertClassifier(backbone, num_classes=num_classes, num_layers=num_layers, dr_rate=dr_rate, bias=bias, batchnorm=batchnorm, layernorm=layernorm)
    elif model_type=='albert':
        model = AlbertClassifier(albert=backbone, num_classes=num_classes, num_layers=num_layers, dr_rate=dr_rate, bias=bias, batchnorm=batchnorm, layernorm=layernorm)
    elif model_type=='kobart':
        model = BartClassifier(linear_input_size=768, bart=backbone, num_classes=num_classes, num_layers=num_layers, dr_rate=dr_rate, bias=bias, batchnorm=batchnorm, layernorm=layernorm)
    elif model_type=='asbart':
        model = BartClassifier(bart=backbone, num_classes=num_classes, num_layers=num_layers, dr_rate=dr_rate, bias=bias, batchnorm=batchnorm, layernorm=layernorm)
    elif model_type=='kogpt2':
        model = KogptClassifier(kogpt=backbone, num_classes=num_classes, num_layers=num_layers, dr_rate=dr_rate, bias=bias, batchnorm=batchnorm, layernorm=layernorm)
    elif model_type=='kogpt3':
        model = KogptClassifier(kogpt=backbone, num_classes=num_classes, num_layers=num_layers, dr_rate=dr_rate, bias=bias, batchnorm=batchnorm, layernorm=layernorm)
    elif model_type=='electra':
        model = ElectraClassifier(electra=backbone, num_classes=num_classes, num_layers=num_layers, dr_rate=dr_rate, bias=bias, batchnorm=batchnorm, layernorm=layernorm)
    elif model_type=='funnel':
        model = FunnelClassifier(funnel=backbone, num_classes=num_classes, num_layers=num_layers, dr_rate=dr_rate, bias=bias, batchnorm=batchnorm, layernorm=layernorm)
    else:
        raise f'invalid model type, {model_type}'
    return model

def load_dataset(model_type, text, label, tokenizer, max_len=50):
    if model_type=='kobert':
        dataset = KobertClassifyDataset(text, label, tokenizer)
    elif model_type=='mlbert':
        dataset = ClassifyDataset(text, label, tokenizer, max_len=max_len, padding='max_length', truncation=True)
    elif model_type=='bert':
        dataset = ClassifyDataset(text, label, tokenizer, max_len=max_len, padding='max_length', truncation=True)
    elif model_type=='albert':
        dataset = ClassifyDataset(text, label, tokenizer, max_len=max_len, padding='max_length', truncation=True)
    elif model_type=='kobart':
        dataset = BartClassifyDataset(text, label, tokenizer, max_len=max_len, padding='max_length', truncation=True)
    elif model_type=='asbart':
        dataset = BartClassifyDataset(text, label, tokenizer, max_len=max_len, padding='max_length', truncation=True)
    elif model_type=='kogpt2':
        dataset = ClassifyDataset(text, label, tokenizer, max_len=max_len, padding='max_length', truncation=True)
    elif model_type=='kogpt3':
        dataset = ClassifyDataset(text, label, tokenizer, max_len=max_len, padding='max_length', truncation=True)
    elif model_type=='electra':
        dataset = ClassifyDataset(text, label, tokenizer, max_len=max_len, padding='max_length', truncation=True)
    elif model_type=='funnel':
        dataset = ClassifyDataset(text, label, tokenizer, max_len=max_len, padding='max_length', truncation=True)
    else:
        raise f'invalid model type, {model_type}'
    return dataset

def load_backbones_tokenizers_classifiers(exp_paths, num_classes=225, device='cpu'):
    # load args and checkpoints
    checkpoints = []
    args_list = []
    for path in exp_paths:
        try:
            checkpoints.append(torch.load(os.path.join(path, 'weights/best_loss.pth.tar')))
        except:
            checkpoints.append(torch.load(os.path.join(path, 'weights/best.pth.tar')))
        with open(os.path.join(path, 'config.json'), 'r', encoding='utf-8-sig') as f:
            args_list.append(json.load(f))
    
    # load classifiers
    base_model_types = []
    classifiers = defaultdict(list)
    for args, checkpoint in zip(args_list, checkpoints):
        model_type=args['model']
        input_size = 1024 if model_type=='asbart' else 768
        classifier = get_classifier(input_size, num_classes, 4026, args['bias_off'], args['dr_rate'], args['n_layers'], args['batchnorm'], args['layernorm'])
        classifier.load_state_dict(checkpoint['state_dict'])
        classifier = classifier.to(device)
        base_model_types.append(model_type)
        classifiers[model_type+'_classifiers'].append(classifier)
        
    # load backbones and tokenizers
    backbones = {}
    tokenizers = {}
    for i, model_type in enumerate(list(set(base_model_types))):
        backbone, tokenizer = load_backbone_tokenizer(model_type, max_len=args_list[i]['max_len'])
        backbones[model_type] = backbone
        tokenizers[model_type+'_tokenizer'] = tokenizer
        
    return backbones, tokenizers, classifiers