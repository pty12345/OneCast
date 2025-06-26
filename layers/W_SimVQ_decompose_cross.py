import torch
import os
import pickle
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, constant_

from einops import rearrange
from layers.RevIN import RevIN

import numpy as np

import copy

from utils.tools import series_decomp
from layers.Season_predictor import Season_predictor

from layers.W_SimVQ_decompose import W_SimVQ_decompose 

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

class W_SimVQ_decompose_cross(nn.Module):
    def __init__(self, configs, dim_table, dataset_id_map, pretrain_map, codebook_sz_dict):
        super().__init__()

        self.configs = configs
        self.dim_table = dim_table

        self.num_t = configs.token_len // configs.wave_length

        self.pretrain_map = pretrain_map
        self.dataset_id_map = dataset_id_map
        self.codebook_sz_dict = codebook_sz_dict

        if "Cross_small" in configs.model_id:
            self.cross_list = ["Cross_FRED_Covid_NYSE", "Cross_Wike2000"]
        elif "Cross_large" in configs.model_id:
            self.cross_list = ["Cross_ETTh2_ETTm2_weather", "Cross_traffic", "Cross_CzeLan"]
        else:
            raise ValueError(f"Invalid model_id: {configs.model_id}")

        self.cross_bias = {}
        for cross_name in self.cross_list:
            self.cross_bias[cross_name] = 0
        
    def load_pretrain_VQ(self):
        elected_n_embed = 0
        self.embed_weight = []

        cross_dataset = self.configs.cross_dataset
        for cross_name in self.cross_list:
            self.configs.cross_dataset = self.pretrain_map[cross_name]

            init_n_embed = self.configs.n_embed
            self.configs.n_embed = self.codebook_sz_dict[cross_name]

            vq_model = W_SimVQ_decompose(self.configs, self.dim_table)

            vqvae_path = self.configs.vqvae_path.replace("Cross_small", cross_name).replace("Cross_large", cross_name)
            vqvae_path = vqvae_path.replace(f"emb{init_n_embed}", f"emb{self.configs.n_embed}")

            load_path = os.path.join(vqvae_path, 'model.pkl')
            assert os.path.exists(load_path), "VQVAE model not found! Should be at {}".format(load_path)
            
            vq_model.load_state_dict(torch.load(load_path))
            
            weight_load_path = os.path.join(vqvae_path, 'weight.pkl')
            weight_dict = pickle.load(open(weight_load_path, 'rb'))
            
            # Use high-frequency codebook only
            self.cross_bias[cross_name] = elected_n_embed
            _elected_n_embed, _elected_codebook_weight =\
                vq_model.resort_codebook(weight_dict, elect_rate=self.configs.elect_rate)
            
            elected_n_embed += _elected_n_embed
            self.embed_weight.append(_elected_codebook_weight)

            exec(f"self.{cross_name}_vq = vq_model")

            self.configs.n_embed = init_n_embed

        self.embed_weight = torch.cat(self.embed_weight, dim=0).to('cuda:0')
        self.configs.cross_dataset = cross_dataset
        return elected_n_embed

    def get_cross_name(self, dataname):
        dataname = dataname.split('-')[0]
        for cross_name in self.cross_list:
            if dataname in cross_name:
                return cross_name
            
        raise ValueError(f"Dataset dataname {dataname} not found in dataset_id_map")

    def get_code(self, input, is_look_back=False, dataname=None):
        cross_name = self.get_cross_name(dataname)
        vq_model = eval(f"self.{cross_name}_vq")

        is_look_back = False
        ids = vq_model.get_code(input, is_look_back, self.dataset_id_map[dataname])
        ids = ids + self.cross_bias[cross_name]
        return ids
    
    def get_name(self):
        return 'w_vqvae'
 
    def get_codebook(self): # [n_embed, hidden_dim]
        return self.embed_weight.transpose(0, 1)
    
    def get_embedding(self, id, cross_name=None):
        vq_model = eval(f"self.{cross_name}_vq")
        embedding = self.embed_weight[id]
        return vq_model.quantize.embedding_proj(embedding)
    
    def decode_ids(self, id, dataname=None):
        cross_name = self.get_cross_name(dataname)
        quant = self.get_embedding(id, cross_name)

        vq_model = eval(f"self.{cross_name}_vq")
        dataset_id = self.dataset_id_map[dataname]
        quant = vq_model.quantize_output_list[dataset_id](quant)  # 2*100*64 -> 2*5000*64
        dec = vq_model.dec_P_list[dataset_id](quant)

        return dec

    def pred_season(self, look_back, eps=1e-5, dataname=None):
        cross_name = self.get_cross_name(dataname)
        vq_model = eval(f"self.{cross_name}_vq")

        dataset_id = self.dataset_id_map[dataname]
        dec_P_season = vq_model.pred_season(look_back, eps, dataset_id)
        return dec_P_season

    def de_norm(self, decode_ts, dataname=None):
        cross_name = self.get_cross_name(dataname)
        vq_model = eval(f"self.{cross_name}_vq")
        return vq_model.revin_layer_list[self.dataset_id_map[dataname]](decode_ts, 'denorm')

if __name__ == '__main__':
    pass
