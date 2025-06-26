import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, constant_

from utils.tools import series_decomp

from einops import rearrange
from models.RevIN import RevIN
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
            p.requires_grad = True
        
        self.embedding_proj = nn.Linear(dim, dim)

        # nn.init.uniform_(embed, -1.0 / self.n_embed, 1.0 / self.n_embed)
        # self.register_buffer("embed", embed)
        # self.register_buffer("cluster_size", torch.zeros(n_embed))
        # self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
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
            
        # print(d.shape)
        # exit(0)

        # print('d: ', d.shape)
        
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
        
        return z_quantize, diff, min_encoding_indices

        # diff = (quantize.detach() - input).pow(2).mean() 
        # commit_loss = (quantize - input.detach()).pow(2).mean() #detach = stop gradient 
        # diff += commit_loss * self.beta
        # quantize = input + (quantize - input).detach() #通过常数让编码器和解码器连续

        # return quantize, diff, embed_ind  # new_x, mse with input, index

    def reload_quant(self, id):
        with torch.no_grad():
            quant_codebook = self.embedding_proj(self.embedding.weight)
            z_quantize = F.embedding(id, quant_codebook)
        return z_quantize

    def embed_code(self, embed_id):
        embedding = F.embedding(embed_id, self.embed.transpose(0, 1))
        return self.embedding_proj(embedding)


class Encoder(nn.Module):
    def __init__(self, feat_num, hidden_dim, block_num, data_shape, dilations=[1, 1]):
        super().__init__()
        self.input_projection = nn.Linear(feat_num, hidden_dim)
        self.blocks = TCN(args=None, d_model=hidden_dim, block_num=block_num, data_shape=data_shape,
                          dilations=dilations)

    def forward(self, input):
        #print(f'input:{input.shape}')
        output = self.input_projection(input)
        output = self.blocks(output)
        
        return output


class Decoder(nn.Module):
    def __init__(self, feat_num, hidden_dim, block_num, data_shape, kernel_size=3, dilations=[1, 1]):
        super().__init__()
        self.output_projection = nn.Linear(hidden_dim, feat_num)
        self.blocks = TCN(args=None, kernel_size=kernel_size, d_model=hidden_dim, block_num=block_num, data_shape=data_shape, dilations=dilations)

    def forward(self, input):
        return self.output_projection(self.blocks(input))

# Window-level Vector Quantization with decompose, equipped with linear transformation (SimVQ)
class W_SimVQ_decompose(nn.Module):
    def __init__(self, configs, dim_table):
        super().__init__()
        
        # input
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        token_len = configs.token_len

        assert seq_len % token_len == 0
        assert pred_len % token_len == 0

        # model config
        hidden_dim = configs.d_model
        n_embed = configs.n_embed
        block_num = configs.block_num
        wave_length = configs.wave_length

        # decompose
        self.decomp = series_decomp(kernel_size=25)

        self.token_len = token_len
        self.pred_len = pred_len
        self.seq_len = seq_len

        self.hidden_dim = hidden_dim
        self.n_embed = n_embed
        self.wave_patch = (wave_length, hidden_dim)

        # Revin config
        self.revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        self.dataset_mapping = {i:dataset for i, dataset in enumerate(configs.cross_dataset)}

        self.quantize = Quantize(hidden_dim, n_embed)

        self.enc_list = nn.ModuleList([])
        self.quantize_input_list = nn.ModuleList([])
        self.dec_L_list = nn.ModuleList([])
        self.dec_P_list = nn.ModuleList([])
        self.quantize_output_list = nn.ModuleList([])
        self.revin_layer_list = nn.ModuleList([])

        for k, v in self.dataset_mapping.items():
            enc_in = dim_table[v]
            data_shape = (token_len, enc_in)
            self.enc_list.append(Encoder(enc_in, hidden_dim, block_num, data_shape))
            self.quantize_input_list.append(nn.Conv2d(1, hidden_dim, kernel_size=self.wave_patch, stride=self.wave_patch))
            self.quantize_output_list.append(nn.Conv1d(int(data_shape[0] / wave_length), data_shape[0], kernel_size=1))

            self.dec_L_list.append(Decoder(data_shape[1], hidden_dim, block_num, data_shape, kernel_size=3))
            self.dec_P_list.append(Decoder(data_shape[1], hidden_dim, block_num, data_shape, kernel_size=3))

            self.revin_layer_list.append(RevIN(enc_in, affine=affine, subtract_last=subtract_last))
        
        # Season predictor
        self.season_predictor = Season_predictor(seq_len=seq_len, pred_len=pred_len, n_domain=len(configs.cross_dataset))

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

            quant_P = quantize_output_layer(quant_P)

        dec_P_trend = dec_P_layer(quant_P)

        ### Channel independent: [B*C*num, token_len] -> [B, L, C] 
        dec_L_trend = torch.reshape(dec_L_trend, (B, L1, C))
        dec_L_trend_denorm = revin_layer(dec_L_trend, 'denorm')

        # decode_P
        dec_P_trend = torch.reshape(dec_P_trend, (B, L2, C))
        dec_P_trend_denorm = revin_layer(dec_P_trend, 'denorm')

        # Season predictor
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

    def get_codebook_weight(self, reflection=True, return_numpy=True):
        copy_codebook = self.quantize.embedding.weight.data.clone()
        if reflection:  
            with torch.no_grad():
                copy_codebook = self.quantize.embedding_proj(copy_codebook)
        if return_numpy:
            return copy_codebook.detach().cpu().numpy()
        else:
            return copy_codebook
    
    def set_random_replace(self, random_replace_num):
        self.random_replace_num = random_replace_num
    
    def elect_codebook(self, elected_ids, rate):
        self.quantize.embedding.weight = \
            nn.Parameter(self.quantize.embedding.weight[elected_ids])
        
        print(f"Quantize Codebook's shape after election@ top {rate}%: \
            {self.quantize.embedding.weight.shape}")
    
    def get_name(self):
        return self.__class__.__name__
    
    def get_embedding(self, id):
        return self.quantize.embed_code(id)
    
    def decode_ids(self, id):
        quant = self.get_embedding(id)
        quant = self.quantize_output(quant)  # 2*100*64 -> 2*5000*64
        dec = self.dec_P(quant)

        return dec

if __name__ == '__main__':
    # Create model and test data
    configs = type('Config', (), {
        'token_len': 96,
        'd_model': 64,
        'n_embed': 512,
        'block_num': 2,
        'wave_length': 8,
        'enc_in': 7,
        'revin': True,
        'affine': False,
        'subtract_last': False
    })()
    
    model = W_SimVQ_decompose(configs)
    init_look_back = torch.randn(64, 96, 7)
    init_horizon = torch.randn(64, 96, 7)
    
    # Forward pass
    (dec_L, diff_L, id_L), dec_P = model(init_look_back, init_horizon)
    
    # Check gradients before backward
    print("Before backward:")
    print("dec_P requires_grad:", dec_P.requires_grad)
    print("dec_P grad:", dec_P.grad)
    print("dec_P.grad_fn:", dec_P.grad_fn)
    print("dec_P is_leaf:", dec_P.is_leaf)
    
    # Try to compute gradients
    loss = dec_P.mean()
    loss.backward()
    
    print("\nAfter backward:")
    print("dec_P grad:", dec_P.grad)
    print("dec_P.grad_fn:", dec_P.grad_fn)
    
    # Check gradients of model parameters
    print("\nGradients of model parameters:")
    print("dec_P.parameters() grad:", [p.grad for p in model.dec_P.parameters()])
    print("enc.parameters() grad:", [p.grad for p in model.enc.parameters()])
    print("quantize.parameters() grad:", [p.grad for p in model.quantize.parameters()])
    print("quantize_input.parameters() grad:", [p.grad for p in model.quantize_input.parameters()])
