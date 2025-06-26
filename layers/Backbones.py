
import json
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import AutoModelForCausalLM, AutoTokenizer

from layers.Transformer_Decoder import Transformer_Decoder
from layers.Backbone_Models.InceptionTime import InceptionTime

import torch.nn.functional as F

class Configs:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                value = Configs(value)
            setattr(self, key, value)

# 修改模型中的所有is_causal参数为True
def set_causal(module, causal=False):
    for name, child in module.named_children():
        if hasattr(child, 'is_causal'):
            child.is_causal = causal
        set_causal(child, causal)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.):
        super(MLP, self).__init__()

        all_dims = [input_dim] + hidden_dims + [output_dim]

        self.linear_layers = nn.ModuleList()
        for i in range(len(all_dims) - 1):
            self.linear_layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            self.linear_layers.append(nn.Dropout(dropout))   
        print('self.linear_layers: ', self.linear_layers)

    def forward(self, x):
        # print('x.shape: ', x.shape)
        # exit(0)
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            if i < len(self.linear_layers) - 1:
                x = F.gelu(x)
        return x

class QwenBackbone(nn.Module):
    def __init__(self, configs):
        super(QwenBackbone, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            configs['model_root'],
        ).model

        if configs['model_layer'] is not None:
            self.model.layers = self.model.layers[:configs['model_layer']]
            
            print('Using {} layers for classifier'.format(configs['model_layer']))
            
        self.text_tokenizer = AutoTokenizer.from_pretrained(configs['model_root'], local_files_only=True)
        self.text_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # 修改模型中的所有is_causal参数为False
        set_causal(self.model, causal=False)

    def get_text_embedding(self, text_ids):
        return self.model.embed_tokens(text_ids)
        
    def forward(self, inputs):
        input_embeddings = inputs['input_embeddings']
        attention_mask = inputs['attention_mask']
        
        hidden_states = self.model(inputs_embeds=input_embeddings, attention_mask=attention_mask, \
                output_hidden_states=True)['hidden_states'][-1]

        return hidden_states

class GPT2Backbone(nn.Module):
    def __init__(self, configs):
        super(GPT2Backbone, self).__init__()
        model_root = configs['model_root']
        
        if configs['use_plm'] == 1: 
            self.model = GPT2LMHeadModel.from_pretrained(model_root)
            if configs['model_layer'] is not None:
                self.model.transformer.h = self.model.transformer.h[:configs['model_layer']]
                
                print('Using {} layers for classifier'.format(configs['model_layer']))
                # exit(0)
        else:
            from transformers import GPT2Config

            configs = GPT2Config.from_json_file('config.json')
            self.model = GPT2LMHeadModel(configs)
            # print(self.model)
            # exit(0)
            
        self.text_tokenizer = GPT2Tokenizer.from_pretrained(model_root, local_files_only=True)
        self.text_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})


        # 修改模型中的所有is_causal参数为False
        set_causal(self.model, causal=False)
        
    def get_text_embedding(self, text_ids):
        return self.model.transformer.wte(text_ids)
        
    def forward(self, inputs):
        input_embeddings = inputs['input_embeddings']
        attention_mask = inputs['attention_mask']
        
        hidden_states = self.model(inputs_embeds=input_embeddings, attention_mask=attention_mask, \
                output_hidden_states=True)['hidden_states'][-1]

        
        return hidden_states
    
class InceptionBackbone(nn.Module):
    def __init__(self, configs):
        super(InceptionBackbone, self).__init__()
        model_root = configs['model_root']
        d_model = configs['model_dim']
        
        self.model = InceptionTime(c_in=d_model, c_out=d_model)
            
        self.text_tokenizer = GPT2Tokenizer.from_pretrained(model_root, local_files_only=True)
        self.text_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.text_embedding = nn.Embedding(self.text_tokenizer.vocab_size, d_model)
        
    def get_text_embedding(self, text_ids):
        return self.text_embedding(text_ids)
        
    def forward(self, inputs):
        input_embeddings = inputs['input_embeddings']
        # attention_mask = inputs['attention_mask']

        hidden_states = self.model(input_embeddings.permute(0, 2, 1))
        hidden_states = hidden_states.permute(0, 2, 1)

        return hidden_states
 
class TransformerDecoderBackbone(nn.Module):
    def __init__(self, configs):
        super(TransformerDecoderBackbone, self).__init__()
        tokenizer_root = configs['model_root']
        d_model = configs['model_dim']
        
        from transformers import GPT2Config

        configs = GPT2Config.from_json_file('config.json')
        
        self.model = GPT2LMHeadModel(configs)
        
        self.text_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_root, local_files_only=True)
        self.text_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.text_embedding = nn.Embedding(self.text_tokenizer.vocab_size, d_model)

        self.out_dim = configs.n_embd

        # 修改模型中的所有is_causal参数为False
        set_causal(self.model, causal=False)

    def get_out_dim(self):
        return self.out_dim
        
    def get_text_embedding(self, text_ids):
        return self.text_embedding(text_ids)
        
    def forward(self, inputs):
        input_embeddings = inputs['input_embeddings']
        attention_mask = inputs['attention_mask']
        
        # exit(0)
        hidden_states = self.model(inputs_embeds=input_embeddings, attention_mask=attention_mask, \
                output_hidden_states=True)['hidden_states'][-1]
        
        # print('hidden_states: ', hidden_states.shape)
        # exit(0)
        
        return hidden_states
    
def get_backbone(backbone, configs):
    print('Init Backbone: {}'.format(backbone))
    
    if backbone == 'gpt2':
        return GPT2Backbone(configs)
    elif backbone == 'trm':
        return TransformerDecoderBackbone(configs)
    elif backbone == 'qwen':
        return QwenBackbone(configs)
    elif backbone == 'IT':
        return InceptionBackbone(configs)
    else:
        raise NotImplementedError('Backbone not implemented: {}'.format(backbone))
    
def get_projection(backbone, configs):
    
    if backbone == 'trm':
        return MLP(configs.d_model, [], output_dim=128)
    elif backbone == 'gpt2':
        return MLP(configs.d_model, [128, 256, 512], 768)
    elif backbone == 'qwen':
        return MLP(configs.d_model, [128, 256, 512], 896)
    elif backbone == 'decode_only':
        return MLP(configs.d_model, [], configs.backbone_dim)
    elif backbone == 'IT':
        return MLP(configs.d_model, [], configs.backbone_dim)
    else:
        raise NotImplementedError('Backbone not implemented: {}'.format(backbone))
        