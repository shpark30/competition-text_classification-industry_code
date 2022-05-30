import torch
import torch.nn as nn
import torch.nn.functional as F

def get_classifier(input_size, num_classes, hidden_size, bias, dr_rate, num_layers, batchnorm, layernorm):
    classifier=[]
    if num_layers == 1:
        if dr_rate:
            classifier.append(nn.Dropout(p=dr_rate))
        classifier.append(nn.Linear(input_size, num_classes, bias=bias))
    else:
        for i in range(num_layers):
            """
            (dropout)
            linear(768, hidden_size)
            (normalization)
            activation
                |
            (dropout)
            linear(hidden_size, hidden_size)
            (normalization)
            activation
                |
               ...
                |
            (dropout)
            linear((hidden_size, num_classes)
            """
            # drop out
            if dr_rate:
                classifier.append(nn.Dropout(p=dr_rate))

            # linear layer
            if i == 0: # 첫번째 층
                classifier.append(nn.Linear(input_size, hidden_size, bias=True))
            elif i != num_layers-1: # 중간 층
                classifier.append(nn.Linear(hidden_size, hidden_size, bias=True))
            else: # 마지막 층
                classifier.append(nn.Linear(hidden_size, num_classes, bias=True))

            # normalization
            if i != num_layers-1: # 마지막 층이 아니면
                if batchnorm:
                    classifier.append(nn.BatchNorm1d(hidden_size))
                if layernorm:
                    classifier.append(nn.LayerNorm(hidden_size))

            # activation
            if i != num_layers-1: # 마지막 층이 아니면
                classifier.append(nn.ReLU())
    return nn.Sequential(*classifier)

class _LMClassifier(nn.Module):
    def __init__(self,
                 num_classes, num_layers=1, linear_input_size=768, hidden_size=4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(_LMClassifier, self).__init__()
        assert not all([batchnorm, layernorm]), 'use one normalization among batchnorm and layernorm. now got both of them.' 
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.linear_input_size = linear_input_size
        self.hidden_size = hidden_size
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.freeze = freeze
        self.dr_rate = dr_rate
        self.bias = bias
        
        # lm_head : classifier
        self.classifier = get_classifier(linear_input_size, num_classes, hidden_size, bias, dr_rate, num_layers, batchnorm, layernorm)

        
class KobertClassifier(_LMClassifier):
    def __init__(self, bert,
                 num_classes, num_layers=1, linear_input_size=768, hidden_size = 4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(KobertClassifier, self).__init__(num_classes, num_layers, linear_input_size, hidden_size, activation, dr_rate, bias, batchnorm, layernorm, freeze)
        self.bert = bert

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooler = self.bert(input_ids=input_ids.long(),
                              token_type_ids=token_type_ids.long(),
                              attention_mask=attention_mask.float())
        return self.classifier(pooler)
    
class BertClassifier(_LMClassifier):
    def __init__(self, bert,
                 num_classes, num_layers=1, linear_input_size=768, hidden_size = 4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(BertClassifier, self).__init__(num_classes, num_layers, linear_input_size, hidden_size, activation, dr_rate, bias, batchnorm, layernorm, freeze)
        self.bert = bert

    def forward(self, input_ids, attention_mask, token_type_ids):
        dec_output = self.bert(input_ids=input_ids.long(),
                               token_type_ids=token_type_ids.long(),
                               attention_mask=attention_mask.float())
        return self.classifier(dec_output.pooler_output)
        

class KogptClassifier(_LMClassifier):
    """ kogpt2 or kogpt3 """
    def __init__(self, kogpt, num_classes, num_layers=1, linear_input_size=768, hidden_size=4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(KogptClassifier, self).__init__(num_classes, num_layers, linear_input_size, hidden_size, activation, dr_rate, bias, batchnorm, layernorm, freeze)
        self.kogpt = kogpt
                    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # transformer decoder output
        # size : (b, n_dec_seq, n_hidden)
        dec_output = self.kogpt.transformer(input_ids=input_ids, 
                                  token_type_ids=token_type_ids,
                                  attention_mask = attention_mask)

        # classifier output
        # size : (b, n_hidden)
        dec_outputs = dec_output.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        # size : (b, num_classes)
        logits_cls = self.classifier(dec_outputs)
        return logits_cls
    
class ElectraClassifier(_LMClassifier):
    def __init__(self, electra, num_classes, num_layers=1, linear_input_size=768, hidden_size=4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(ElectraClassifier, self).__init__(num_classes, num_layers, linear_input_size, hidden_size, activation, dr_rate, bias, batchnorm, layernorm, freeze)
        self.electra = electra
                    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # transformer decoder output
        # size : (b, n_dec_seq, n_hidden)
        dec_output = self.electra(input_ids=input_ids, 
                                 token_type_ids=token_type_ids,
                                 attention_mask = attention_mask)

        # classifier output
        # size : (b, n_hidden)
        dec_outputs = dec_output.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        # size : (b, num_classes)
        logits_cls = self.classifier(dec_outputs)
        return logits_cls
    
class AlbertClassifier(_LMClassifier):
    def __init__(self, albert, num_classes, num_layers=1, linear_input_size=768, hidden_size=4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(AlbertClassifier, self).__init__(num_classes, num_layers, linear_input_size, hidden_size, activation, dr_rate, bias, batchnorm, layernorm, freeze)
        self.albert = albert
                    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # transformer decoder output
        # size : (b, n_dec_seq, n_hidden)
        dec_output = self.albert(input_ids=input_ids, 
                                 token_type_ids=token_type_ids,
                                 attention_mask = attention_mask)

        # classifier output
        # size : (b, n_hidden)
        dec_outputs = dec_output.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        # size : (b, num_classes)
        logits_cls = self.classifier(dec_outputs)
        return logits_cls
    
class FunnelClassifier(_LMClassifier):
    def __init__(self, funnel, num_classes, num_layers=1, linear_input_size=768, hidden_size=4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(FunnelClassifier, self).__init__(num_classes, num_layers, linear_input_size, hidden_size, activation, dr_rate, bias, batchnorm, layernorm, freeze)
        self.funnel = funnel
                    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # transformer decoder output
        # size : (b, n_dec_seq, n_hidden)
        dec_output = self.funnel(input_ids=input_ids, 
                                 token_type_ids=token_type_ids,
                                 attention_mask = attention_mask)

        # classifier output
        # size : (b, n_hidden)
        dec_outputs = dec_output.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        # size : (b, num_classes)
        logits_cls = self.classifier(dec_outputs)
        return logits_cls
    
class BartClassifier(_LMClassifier):
    def __init__(self, bart, num_classes, num_layers=1, linear_input_size=1024, hidden_size=4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(BartClassifier, self).__init__(num_classes, num_layers, linear_input_size, hidden_size, activation, dr_rate, bias, batchnorm, layernorm, freeze)
        self.bart = bart
                    
    def forward(self, input_ids, attention_mask):
        # transformer decoder output
        # size : (b, n_dec_seq, n_hidden)
        dec_outputs = self.bart(input_ids=input_ids, 
                               attention_mask = attention_mask)

        # classifier output
        # size : (b, n_hidden)
        dec_outputs = dec_outputs.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        # size : (b, num_classes)
        logits_cls = self.classifier(dec_outputs)
        return logits_cls