import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, constant_

from einops import rearrange
from layers.RevIN import RevIN

import numpy as np

from utils.tools import series_decomp
from layers.Season_predictor import Season_predictor

class TCN(nn.Module):
    def __init__(self, args=None, **kwargs):
        super(TCN, self).__init__()

        # load parameters info
        if args is not None:
            d_model = args.d_model
            self.embedding_size = args.d_model
            self.residual_channels = args.d_model
            self.block_num = args.block_num
            self.dilations = args.dilations * self.block_num
            self.kernel_size = args.kernel_size
            self.enabel_res_parameter = args.enable_res_parameter
            self.dropout = args.dropout
            self.device = args.device
            self.data_shape = args.data_shape
        else:
            d_model = kwargs['d_model']
            self.embedding_size = kwargs['d_model']
            self.residual_channels = kwargs['d_model']
            self.block_num = kwargs['block_num']
            self.dilations = kwargs['dilations'] * self.block_num
            self.data_shape = kwargs['data_shape']
            self.kernel_size = 3
            self.enabel_res_parameter = 1
            self.dropout = 0.1

        self.max_len = self.data_shape[0]
        #print(self.max_len)

        # residual blocks    dilations in blocks:[1,2,4,8,1,2,4,8,...]
        rb = [
            ResidualBlock_b(
                self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation,
                enable_res_parameter=self.enabel_res_parameter, dropout=self.dropout
            ) for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)

        # fully-connected layer
        # self.output = nn.Linear(self.residual_channels, self.num_class)
        self.output = nn.Linear(d_model, d_model)
        self.broadcast_head = nn.Linear(d_model, self.data_shape[1])
        
        
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def forward(self, x):
        # Residual locks
        # x in shape of [(B*T)*L*D]
        dilate_outputs = self.residual_blocks(x)
        x = dilate_outputs
        return self.output(x)


class ResidualBlock_b(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=10, dilation=None, enable_res_parameter=False, dropout=0):
        super(ResidualBlock_b, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

        self.dilation = dilation
        self.kernel_size = kernel_size

        self.enable = enable_res_parameter
        self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x):  # x: [batch_size, token_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, token_len+(self.kernel_size-1)*dilations]
        #print(f'x_pad_output:{x_pad.shape}')
        out = self.dropout1(self.conv1(x_pad).squeeze(2).permute(0, 2, 1))
        #print(f'conv2_output:{out.shape}')
        # [batch_size, token_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation)
        out2 = self.dropout2(self.conv2(out_pad).squeeze(2).permute(0, 2, 1))
        out2 = F.relu(self.ln2(out2))

        if self.enable:
            x = self.a * out2 + x
        else:
            x = out2 + x

        return x
        # return self.skipconnect(x, self.ffn)

    def conv_pad(self, x, dilation):
        """ Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        #print(f'output_pad:{inputs_pad.shape}')
        return inputs_pad
    
# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# 
class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, beta=0.25):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.beta = beta
        self.entropy_penalty = 0.1
        
        # SimVQ
        self.legacy = True
        
        self.embedding = nn.Embedding(n_embed, dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=dim**-0.5)
        for p in self.embedding.parameters():
            p.requires_grad = False
        
        self.embedding_proj = nn.Linear(dim, dim)


        self.top_elected_tokens = n_embed
        # nn.init.uniform_(embed, -1.0 / self.n_embed, 1.0 / self.n_embed)
        # self.register_buffer("embed", embed)
        # self.register_buffer("cluster_size", torch.zeros(n_embed))
        # self.register_buffer("embed_avg", embed.clone())

    def forward(self, input, is_look_back):
        flatten = input.reshape(-1, self.dim)
        # dist = (
        #         flatten.pow(2).sum(1, keepdim=True)
        #         - 2 * flatten @ self.embed
        #         + self.embed.pow(2).sum(0, keepdim=True)
        # )
        # _, embed_ind = (-dist).max(1)
        # embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        # embed_ind = embed_ind.view(*input.shape[:-1])
        # quantize = self.embed_code(embed_ind)

        quant_codebook = self.embedding_proj(self.embedding.weight)
        
        # print(flatten.shape)
        # print(quant_codebook.shape)
        
        # exit(0)

        d = torch.sum(flatten ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', flatten, rearrange(quant_codebook, 'n d -> d n'))
            
        if is_look_back == False:
            d = d[:, :self.top_elected_tokens]
            
        min_encoding_indices = torch.argmin(d, dim=1)
        z_quantize = F.embedding(min_encoding_indices, quant_codebook).view(input.shape)
        # perplexity = None
        # min_encodings = None

        diff = torch.mean((z_quantize.detach()-input) ** 2)
        commit_loss = torch.mean((z_quantize - input.detach()) ** 2)

        # compute loss for embedding
        if not self.legacy:
            diff = self.beta * diff + commit_loss
        else:
            diff = diff + self.beta * commit_loss

        # preserve gradients
        z_quantize = input + (z_quantize - input).detach()
        
        
        # print("shape of z_quantize, diff, min_encoding_indices")
        # print(z_quantize.shape)
        # print(diff.shape)
        # print(min_encoding_indices.shape)
        
        # exit(0)
        
        # print(min_encoding_indices.shape)
        
        # print('size:', *input.shape[:-1])
        
        embed_ind = min_encoding_indices.view(*input.shape[:-1])
        
        # print(embed_ind.shape)
        
        return z_quantize, diff, embed_ind, d

class Encoder(nn.Module):
    def __init__(self, feat_num, hidden_dim, block_num, data_shape, dilations=[1, 1]):
        super().__init__()
        self.input_projection = nn.Linear(feat_num, hidden_dim)
        self.blocks = TCN(args=None, d_model=hidden_dim, block_num=block_num, data_shape=data_shape,
                          dilations=dilations)

    def forward(self, input):
        #print(f'input:{input.shape}')
        output = self.input_projection(input)
        #print(f'embeding_output:{output.shape}')
        output = self.blocks(output)
        
        return output


class Decoder(nn.Module):
    def __init__(self, feat_num, hidden_dim, block_num, data_shape, dilations=[1, 1]):
        super().__init__()
        self.output_projection = nn.Linear(hidden_dim, feat_num)
        self.blocks = TCN(args=None, d_model=hidden_dim, block_num=block_num, data_shape=data_shape,
                          dilations=dilations)

    def forward(self, input):
        return self.output_projection(self.blocks(input))

class W_SimVQ_decompose(nn.Module):
    def __init__(self, configs, dim_table):
        super().__init__()
        
        # model config
        token_len = configs.token_len
        hidden_dim = configs.d_model
        n_embed = configs.n_embed
        block_num = configs.block_num
        wave_length = configs.wave_length

        enc_in = configs.enc_in
        
        data_shape = (token_len, enc_in)

        self.token_len = token_len
        self.seq_len = configs.seq_len   
        self.pred_len = configs.pred_len

        self.num_t = token_len // wave_length

        self.wave_patch = (wave_length, hidden_dim)

        # Revin config
        self.revin = configs.revin
        
        affine = configs.affine
        subtract_last = configs.subtract_last

        self.dataset_mapping = {i:dataset for i, dataset in enumerate(configs.cross_dataset)}
        
        self.enc_list = nn.ModuleList([])
        self.quantize_input_list = nn.ModuleList([])
        self.dec_L_list = nn.ModuleList([])
        self.dec_P_list = nn.ModuleList([])
        self.quantize_output_list = nn.ModuleList([])
        self.revin_layer_list = nn.ModuleList([])

        self.quantize = Quantize(hidden_dim, n_embed)

        for k, v in self.dataset_mapping.items():
            enc_in = dim_table[v]
            print(f'enc_in: {enc_in}')
            data_shape = (token_len, enc_in)
            self.enc_list.append(Encoder(enc_in, hidden_dim, block_num, data_shape))
            self.quantize_input_list.append(nn.Conv2d(1, hidden_dim, kernel_size=self.wave_patch, stride=self.wave_patch))
            self.quantize_output_list.append(nn.Conv1d(int(data_shape[0] / wave_length), data_shape[0], kernel_size=1))

            self.dec_L_list.append(Decoder(data_shape[1], hidden_dim, block_num, data_shape))
            self.dec_P_list.append(Decoder(data_shape[1], hidden_dim, block_num, data_shape))

            self.revin_layer_list.append(RevIN(enc_in, affine=affine, subtract_last=subtract_last))
        

        # decompose
        self.decomp = series_decomp(kernel_size=25)

        self.season_predictor = Season_predictor(seq_len=self.seq_len, pred_len=self.pred_len, n_domain=len(configs.cross_dataset))

    def forward(self, init_look_back, init_horizon, dataset_id):
        ### Normalize
        # must keep this order!!!
        revin_layer = self.revin_layer_list[dataset_id]
        enc_layer = self.enc_list[dataset_id]
        quantize_input_layer = self.quantize_input_list[dataset_id]
        quantize_output_layer = self.quantize_output_list[dataset_id]
        dec_L_layer = self.dec_L_list[dataset_id]
        dec_P_layer = self.dec_P_list[dataset_id]



        horizon = revin_layer(init_horizon, 'norm')
        horizon_res, horizon_trend = self.decomp(horizon)
        gt_horizon_trend = revin_layer(horizon_trend, 'denorm')
        gt_horizon_res = revin_layer(horizon_res, 'denorm')

        look_back = revin_layer(init_look_back, 'norm')
        look_back_res, look_back_trend = self.decomp(look_back)
        gt_look_back_trend = revin_layer(look_back_trend, 'denorm')
        gt_look_back_res = revin_layer(look_back_res, 'denorm')
            
        # assert look_back.shape == horizon.shape
        B, L1, C = look_back_trend.shape
        B, L2, C = horizon_trend.shape

        # Reshape to [B, N, token_len, C]
        look_back_token = torch.reshape(look_back_trend, (B, self.seq_len // self.token_len, self.token_len, C))
        horizon_token = torch.reshape(horizon_trend, (B, self.pred_len // self.token_len, self.token_len, C))

        # Reshape to [B*N, token_len, C]
        look_back_token = torch.reshape(look_back_trend, (-1, self.token_len, C)) # [B*C*num, token_len, 1]
        horizon_token = torch.reshape(horizon_trend, (-1, self.token_len, C)) # [B*C*num, token_len, 1]

        ### look_back window
        enc_L = enc_layer(look_back_token) # [B*C*num, token_len, hidden_dim]

        enc_L = enc_L.unsqueeze(1)
        quant_L = quantize_input_layer(enc_L).squeeze(-1).transpose(1, 2)
        quant_L, diff_L, id_L = self.quantize(quant_L)
        quant_L = quantize_output_layer(quant_L)
        dec_L_trend = dec_L_layer(quant_L) # [B*C*num, token_len, 1]

        ### Horizon window
        with torch.no_grad():
            enc_P = enc_layer(horizon_token)
            enc_P = enc_P.unsqueeze(1)
            quant_P = quantize_input_layer(enc_P).squeeze(-1).transpose(1, 2)
            quant_P, diff_P, id_P = self.quantize(quant_P)

            # if self.random_replace_num > 0:
            #     id_P = torch.reshape(id_P, (B*C, -1)) # [B, Num]
            #     # Randomly replace tokens in the second dimension based on random_replace_num
            #     replace_indices = torch.randint(0, id_P.shape[1], (B*C, self.random_replace_num)).to(id_P.device)
            #     random_tokens = torch.randint(0, self.quantize.n_embed, (B*C, self.random_replace_num)).to(id_P.device)
            #     id_P.scatter_(1, replace_indices, random_tokens)

            #     id_P = id_P.flatten()
            #     quant_P = self.quantize.reload_quant(id_P).view(quant_P.shape)

            quant_P = quantize_output_layer(quant_P)

        dec_P_trend = dec_P_layer(quant_P)

        # Only dec_P will have gradients
        
        # print('dec_L: ', dec_L.shape)
        # print('dec_P: ', dec_P.shape)
            

        ### Channel independent: [B*C*num, token_len] -> [B, L, C] 
        dec_L_trend = torch.reshape(dec_L_trend, (B, L1, C))
        dec_L_trend_denorm = revin_layer(dec_L_trend, 'denorm')

        # decode_P
        dec_P_trend = torch.reshape(dec_P_trend, (B, L2, C))
        dec_P_trend_denorm = revin_layer(dec_P_trend, 'denorm')

        dec_P_season = self.season_predictor(look_back_res, dataset_id=dataset_id)

        dec_P = dec_P_trend + dec_P_season
        dec_P = revin_layer(dec_P, 'denorm')

        return {
            'dec_L_trend': dec_L_trend_denorm,
            'dec_P_trend': dec_P_trend_denorm,
            'dec_P': dec_P,
            'gt_L_trend': gt_look_back_trend,
            'gt_P_trend': gt_horizon_trend,
            'gt_P': init_horizon,
            'diff_L': diff_L,
            'id_L': id_L
        }

    def get_code(self, input, is_look_back=False, dataset_id=None):
        revin_layer = self.revin_layer_list[dataset_id]
        enc_layer = self.enc_list[dataset_id]
        quantize_input_layer = self.quantize_input_list[dataset_id]
        quantize_output_layer = self.quantize_output_list[dataset_id]

        with torch.no_grad():
            input = revin_layer(input, 'norm')
            input_res, input_trend = self.decomp(input)

            B, L, C = input_trend.shape
            
            trend_token = torch.reshape(input_trend, (B, L // self.token_len, self.token_len, C))
            trend_token = torch.reshape(trend_token, (-1, self.token_len, C)) # [B*num, token_len, C]
            
            # slide each windows
            assert L % self.token_len == 0, 'input length should be divided by token length'    
            
            # print(f'normed input:{input.shape}')
            # exit(0)

            enc = enc_layer(trend_token)
            enc = enc.unsqueeze(1)
            quant = quantize_input_layer(enc).squeeze(-1).transpose(1, 2)
            _, _, window_ids, d = self.quantize(quant, is_look_back=is_look_back)

            B_num, num_t = window_ids.shape
            id = torch.reshape(window_ids, (B, -1))

            # print(f'id: {id.shape}')
            # exit(0)
            
            assert num_t == self.num_t, f'num_t {num_t} should be equal to self.num_t {self.num_t}'
        
        return id
    
    def get_name(self):
        return 'w_vqvae'
    

    def elect_codebook(self, elected_ids, rate):
        self.quantize.embedding.weight = \
            nn.Parameter(self.quantize.embedding.weight[elected_ids], requires_grad=False)
        
        print(f"Quantize Codebook's shape after election@ top {rate}%: \
            {self.quantize.embedding.weight.shape}")
        
    def resort_codebook(self, weight_dict, elect_rate):
        cnts_element = weight_dict['train_cnts_elements']
        uni_element = weight_dict['train_uni_elements']
        
        _sorted_index = np.argsort(cnts_element)[::-1]
        
        sorted_index = uni_element[_sorted_index]
        
        self.quantize.embedding.weight = \
            nn.Parameter(self.quantize.embedding.weight[sorted_index], requires_grad=False)
            
        self.n_embed = len(sorted_index)
            
        # set output mask
        cnts_element = cnts_element[_sorted_index]
        board = weight_dict['total_nums'] * (elect_rate / 100)
        out_codebook_mask = np.zeros(self.n_embed)
        out_codebook_mask[np.where(cnts_element > board)] = 1
        
        self.quantize.top_elected_tokens = int(np.sum(out_codebook_mask))
        
        print(f"Quantize Codebook's shape after resorting@ top {elect_rate}%: {self.quantize.top_elected_tokens}")
        
        return self.quantize.top_elected_tokens, self.quantize.embedding.weight[:self.quantize.top_elected_tokens]
        
    def get_codebook(self): # [n_embed, hidden_dim]
        return self.quantize.embedding.weight.transpose(0, 1)
    
    def get_embedding(self, id):
        embedding = self.quantize.embedding(id)
        return self.quantize.embedding_proj(embedding)
    
    
    def decode_ids(self, id, dataset_id=None):
        quant = self.get_embedding(id)
        quant = self.quantize_output_list[dataset_id](quant)  # 2*100*64 -> 2*5000*64
        dec = self.dec_P_list[dataset_id](quant)

        return dec
    
    def pred_season(self, look_back, eps=1e-5, dataset_id=None):
        x = look_back
        dim2reduce = tuple(range(1, x.ndim-1))
        mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + eps).detach()

        x = x - mean
        x = x / stdev

        x_res, x_trend = self.decomp(x)

        dec_P_season = self.season_predictor(x_res, dataset_id=dataset_id)
        return dec_P_season

if __name__ == '__main__':
    model = W_SimVQ_CNN_double_token_decompose_v3()
    a = torch.randn(2, 5000, 12)
    tmp = model(a)
    #print(model.get_embedding)
    print(1)
