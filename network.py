import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, GPTJForCausalLM, GPT2LMHeadModel
from kobert.pytorch_kobert import get_pytorch_kobert_model

class EnsembleClassifier(nn.Module):
    """ https://discuss.pytorch.org/t/custom-ensemble-approach/52024/8 """
    def __init__(self, kogpt2, kobert, num_classes=225, hidden_size=4026):
        super(EnsembleClassifier, self).__init__()
        self.kogpt2 = kogpt2
        self.kobert = kobert
        self.num_classes = num_classes
        
        # Create new classifier
        self.classifier = nn.Sequential(nn.Linear(768+768, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, num_classes))
        
        # Remove last linear layer
        self.kogpt2.classifier = nn.Identity()
        self.kobert.classifier = nn.Identity()
        
        for child in self.kogpt2.children():
            for param in child.parameters():
                param.requires_grad = False
                
        for child in self.kobert.children():
            for param in child.parameters():
                param.requires_grad = False
        
    def forward(self, token_ids, attention_mask, token_type_ids):
        # kogpt2 output
        x1 = self.kogpt2.gpt.transformer(input_ids=token_ids.clone()[:,1],
                                  token_type_ids=token_type_ids.clone()[:,1],
                                  attention_mask = attention_mask.clone()[:,1])
        x1 = x1.last_hidden_state[:, -1].contiguous()
        x1 = x1.view(x1.size(0), -1)
        
        # kobert output
        _, x2 = self.kobert.bert(input_ids=token_ids.long()[:,0],
                          token_type_ids=token_type_ids.long()[:,0],
                          attention_mask=attention_mask.float()[:,0])
        x2 = x2.view(x2.size(0), -1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x

    
class KOBERTClassifier(nn.Module):
    def __init__(self, bert, num_classes, hidden_size = 768, dr_rate=None, params=None):
        super(KOBERTClassifier, self).__init__()
        self.bert = bert
        self.num_classes = num_classes
        self.dr_rate = dr_rate
         
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
#     def gen_attention_mask(self, token_ids, valid_length):
#         attention_mask = torch.zeros_like(token_ids)
#         for i, v in enumerate(valid_length):
#             attention_mask[i][:v] = 1
#         return attention_mask.float()

    def forward(self, token_ids, attention_mask, token_type_ids):
        _, pooler = self.bert(input_ids=token_ids.long(),
                              token_type_ids=token_type_ids.long(),
                              attention_mask=attention_mask.float())
        if self.dr_rate:
            pooler = self.dropout(pooler)
        return self.classifier(pooler)
    
    
class _KOGPTClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size, freeze_gpt, dr_rate, pretrained_model_name_or_path,
                revision=None, pad_token_id=1, torch_dtype='auto', low_cpu_mem_usage=True):
        super(_KOGPTClassifier, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.freeze_gpt = freeze_gpt
        self.dr_rate = dr_rate
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.pad_token_id = pad_token_id
        self.torch_dtype = torch_dtype
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.gpt = None
        self.classifier = None
        
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
                    
    
class KOGPT2Classifier(_KOGPTClassifier):
    def __init__(self, num_classes, hidden_size=4026, freeze_gpt=True, dr_rate=None,
                pretrained_model_name_or_path='skt/kogpt2-base-v2', 
                pad_token_id = 1,
                torch_dtype='auto', low_cpu_mem_usage=True):
        super(KOGPT2Classifier, self).__init__(num_classes, hidden_size=hidden_size, freeze_gpt=freeze_gpt, dr_rate=dr_rate, 
                                               pretrained_model_name_or_path=pretrained_model_name_or_path,
                                               pad_token_id=pad_token_id, torch_dtype=torch_dtype, 
                                               low_cpu_mem_usage=low_cpu_mem_usage)
        
        self.gpt = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            pad_token_id=pad_token_id, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage)
            
        # classifier
        self.classifier = nn.Sequential(nn.Linear(768, hidden_size, bias=True, dtype=torch.float32),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, num_classes, bias=True, dtype=torch.float32))
        
        if self.freeze_gpt:
            for child in self.gpt.children():
                for param in child.parameters():
                    param.requires_grad = False
            
    
    
class KOGPT3Classifier(_KOGPTClassifier):
    """
    classifier layer 추가 방법 참고 사이트 : https://paul-hyun.github.io/gpt-02/
    """
    def __init__(self, hidden_size=768, num_classes=235, freeze_gpt=True, dr_rate=None,
                pretrained_model_name_or_path='kakaobrain/kogpt',
                revision='KoGPT6B-ryan1.5b-float16',
                pad_token_id = 1,
                torch_dtype='auto', low_cpu_mem_usage=True):
        super(KOGPT3Classifier, self).__init__(num_classes, hidden_size=hidden_size, freeze_gpt=freeze_gpt, dr_rate=dr_rate, 
                                               pretrained_model_name_or_path=pretrained_model_name_or_path,
                                               revision=revision, pad_token_id=pad_token_id, torch_dtype=torch_dtype, 
                                               low_cpu_mem_usage=low_cpu_mem_usage)
        
        # decoder : self.gpt3.transformer (transformers GPTModel)
        # lm : self.gpt3.lm_head (nn.Linear)
        self.gpt = GPTJForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            revision=revision, pad_token_id=pad_token_id,
            torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage
        )
        
#         self.classifier = nn.Sequential(nn.Linear(4096, hidden_size, bias=True, dtype=torch.float16),
#                                         nn.ReLU(),
#                                         nn.Linear(hidden_size, num_classes, bias=True, dtype=torch.float16))
        self.classifier = nn.Linear(4096, num_classes, bias=True, dtype=torch.float16)
        
        if self.freeze_gpt:
            for child in self.gpt.children():
                for param in child.parameters():
                    param.requires_grad = False
  