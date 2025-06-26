import os
import copy
import torch
import math

from omegaconf import OmegaConf

from torch import nn

import torch.nn.functional as F

from layers.Backbones import get_backbone, get_projection

def init_weights_kaiming(m):
	if type(m) == nn.Linear:
		nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
		m.bias.data.fill_(0.01)


# DD-Time
class Model(nn.Module):
	def __init__(self, configs, norm:str='batch', act:str="gelu", head_type='flatten'):
		
		super().__init__()

		if configs.backbone in ['gpt2', 'qwen']:
			llm_config = OmegaConf.load(configs.cfg_path)
			
			model_config = llm_config.model[configs.backbone]
			model_root = model_config['root_dir']
		else: # Using GPT2's Tokenizer
			model_root = "/data/tingyue/LM-Base-Model/GPT2"
		
		backbone_cfgs = {
			'model_root': model_root,
			'model_dim': configs.backbone_dim,
			'model_layer': configs.n_classifier_layer,
		}
		
		self.backbone = get_backbone(configs.backbone, backbone_cfgs)

		# print(self.backbone)
		# exit(0)
		
		self.text_tokenizer_len = len(self.backbone.text_tokenizer)
		self.ts_tokenizer_len = configs.elected_n_embed
		
		self.mask_id = self.ts_tokenizer_len # without [MASK]
		
		self.tokenizer_len = self.text_tokenizer_len + self.ts_tokenizer_len + 1 # [MASK]
		
		# time series projection layers
		self.projection_layers = get_projection(configs.backbone, configs)
		self.projection_layers.apply(init_weights_kaiming)
		
		# output probability layer
		model_dim = self.backbone.get_out_dim()
		
		self.output_layer = nn.Linear(model_dim, self.ts_tokenizer_len)
  
		# print(self)
		
		self.state = None
		
	def init_ts_tokenizer(self, ts_tokenizer):
		ts_tokenizer_embed = copy.deepcopy(ts_tokenizer.get_codebook()).transpose(-1, 0) # [N, D]
		
		self.mask_embed = torch.mean(ts_tokenizer_embed, dim=0, keepdim=True)	
  
		ts_tokenizer_embed = torch.cat([ts_tokenizer_embed, self.mask_embed], dim=0)
		self.register_buffer('ts_tokenizer_embed', ts_tokenizer_embed)
		
	def _get_token_embeddings(self, text_ids, ts_ids):
		text_embeddings, ts_embeddings = None, None
		
		if text_ids is not None:
			text_embeddings = self.backbone.get_text_embedding(text_ids)
   
		ts_embeddings = self.ts_tokenizer_embed[ts_ids]
		
		return text_embeddings, ts_embeddings
	
	def set_state(self, state):
		self.state = state
  
	def forward(self, inputs, mode='train', **kwargs):   
		assert mode in ['train', 'gen_ts'], 'Invalid mode: {}'.format(mode)
  
		if mode == 'gen_ts':
			return self.gen_ts(inputs, **kwargs)
     
		text_ids, ts_ids = inputs['text_ids'], inputs['ts_ids']
  
		# print(set(ts_ids.flatten().tolist()))
  
		# exit(0)	
		
		text_embeddings, ts_embeddings = self._get_token_embeddings(text_ids, ts_ids)
				
		# project ts embeddings
		ts_embeddings = self.projection_layers(ts_embeddings)
		
		if text_embeddings is not None:
			input_embeddings = torch.cat([text_embeddings, ts_embeddings], dim=1)
		else:
			input_embeddings = ts_embeddings
			
		attention_mask = torch.ones(input_embeddings.shape[0], input_embeddings.shape[1])
			
		backbone_inputs = {
			'input_embeddings': input_embeddings, # 224 ,24, 896
			'attention_mask': attention_mask
		}

		hidden_states = self.backbone(backbone_inputs)
		
		# print(hidden_states.shape)
			
		output = self.output_layer(hidden_states)
		
		# print(output.shape)
		# exit(0)

		return output
	
	def gen_ts(self, inputs, out_token_num, time_step=8):
		text_ids, ts_ids = inputs['text_ids'], inputs['ts_ids']
		
		recover_per_step = math.ceil((1.0 / time_step) * out_token_num)
		# print(recover_per_step)

		B = ts_ids.shape[0]

		device = ts_ids.device
		
		output_ids_list = []

		sample_topk = -1 

		# generate ts directly
		if sample_topk <= 0: 
			cond_ts_ids, output_ts_ids = ts_ids[:, :-out_token_num], ts_ids[:, -out_token_num:]
			for t in range(time_step):
				with torch.no_grad():
					# print(output_ts_ids[0])
					
					current_ts_ids = torch.cat([cond_ts_ids, output_ts_ids], dim=1)
					
					forward_inputs = {'text_ids': text_ids, 'ts_ids': current_ts_ids}
					logits = self.forward(forward_inputs, mode='train')
					
					logits = logits[:, -out_token_num:, :]

					probs = F.softmax(logits, dim=-1)
					
					
					# 选取output_ts_ids中，mask位置的概率最大的token
					posi, nega = torch.tensor([1]).to(device), torch.tensor([0]).to(device)
					mask_flag = torch.where(output_ts_ids == self.mask_id, posi, nega)
					
					probs, maxp_pos = torch.max(probs, dim=-1)
					probs = probs * mask_flag

					# for i in range(probs.shape[0]):
					# 	print(probs[i])
					# 	if i == 100:
					# 		exit(0)
					# print(probs.shape)
					# exit(0)
					
					# print('probs: ', probs[0])
					
					remain_num = torch.sum(mask_flag) // B
					recover_num = min(remain_num, recover_per_step)
					
					if recover_num == 0: break
					
					# 每行选取出recover_num个最大的token
					_, topk_pos = torch.topk(probs, recover_num, dim=-1)
					index = torch.arange(B)
					for k in range(recover_num):
						k_pos = topk_pos[:, k]
						
						output_ts_ids[index, k_pos] = maxp_pos[index, k_pos]
			output_ids_list.append(output_ts_ids)
		
		# sample topk path according to the probability
		else:
			for path in range(sample_topk):
				
				cond_ts_ids, output_ts_ids = ts_ids[:, :-out_token_num], ts_ids[:, -out_token_num:]
				for t in range(time_step):
					with torch.no_grad():
						current_ts_ids = torch.cat([cond_ts_ids, output_ts_ids], dim=1)
						
						forward_inputs = {'text_ids': text_ids, 'ts_ids': current_ts_ids}
						logits = self.forward(forward_inputs, mode='train')

						logits = logits[:, -out_token_num:, :]

						probs = F.softmax(logits, dim=-1) # [B', out_token_num, ts_tokenizer_len]

						sample_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1) # [B', out_token_num, 1]
						sample_tokens = sample_tokens.view(-1, probs.shape[1], 1)

						sample_probs = torch.gather(probs, dim=-1, index=sample_tokens) # [B, out_token_num, 1]

						sample_tokens = sample_tokens.squeeze(-1)
						sample_probs = sample_probs.squeeze(-1)

						# 选取output_ts_ids中，mask位置的概率最大的token
						posi, nega = torch.tensor([1]).to(device), torch.tensor([0]).to(device)
						mask_flag = torch.where(output_ts_ids == self.mask_id, posi, nega)
						
						sample_probs = sample_probs * mask_flag

						remain_num = torch.sum(mask_flag) // B
						recover_num = min(remain_num, recover_per_step)
						
						if recover_num == 0: break
						
						# 每行选取出recover_num个最大的token
						_, topk_pos = torch.topk(sample_probs, recover_num, dim=-1)
						index = torch.arange(B)
						for k in range(recover_num):
							k_pos = topk_pos[:, k]
							
							output_ts_ids[index, k_pos] = sample_tokens[index, k_pos]

				output_ids_list.append(output_ts_ids)

		return output_ids_list