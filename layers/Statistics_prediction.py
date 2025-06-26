import torch
import torch.nn as nn
import copy
from torch.nn.init import xavier_normal_, constant_


class Statistics_prediction(nn.Module):
    def __init__(self, configs):
        super(Statistics_prediction, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.period_len = configs.token_len
        self.channels = configs.enc_in if configs.features == 'M' else 1
        self.station_type = configs.station_type

        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))

    def _build_model(self):
        args = copy.deepcopy(self.configs)
        args.seq_len = self.configs.seq_len // self.period_len
        args.label_len = self.configs.label_len // self.period_len
        args.enc_in = self.configs.enc_in
        args.dec_in = self.configs.enc_in
        args.moving_avg = 3
        args.c_out = self.configs.c_out
        args.pred_len = self.pred_len_new
        self.model = MLP(args, mode='mean').float()
        self.model_std = MLP(args, mode='std').float()

    def normalize(self, ori_input):
        if 'adaptive' in self.station_type:
            bs, len, dim = ori_input.shape
            input = ori_input.reshape(bs, -1, self.period_len, dim)
            mean = torch.mean(input, dim=-2, keepdim=True)
            
            if self.station_type == 'adaptive_mean':
                dim2reduce = tuple(range(1, ori_input.ndim-1))
                out_std = torch.sqrt(torch.var(ori_input, dim=dim2reduce, keepdim=True, unbiased=False) + self.epsilon).detach()
                
                std = torch.std(input, dim=-2, keepdim=True)
                norm_input = (input - mean) / (std + self.epsilon)
                
                # adaptive mean
                mean_all = torch.mean(ori_input, dim=1, keepdim=True)
                outputs_mean = self.model(mean.squeeze(2) - mean_all, ori_input - mean_all) * self.weight[0] + mean_all * self.weight[1]
                
                # prior std
                outputs_std = out_std.squeeze(2).repeat(1, self.pred_len_new, 1)
            else:
                std = torch.std(input, dim=-2, keepdim=True)
                norm_input = (input - mean) / (std + self.epsilon)
                
                # adaptive mean
                mean_all = torch.mean(ori_input, dim=1, keepdim=True)
                outputs_mean = self.model(mean.squeeze(2) - mean_all, ori_input - mean_all) * self.weight[0] + mean_all * self.weight[1]
                
                # adaptive std
                outputs_std = self.model_std(std.squeeze(2), ori_input)
    
            outputs = torch.cat([outputs_mean, outputs_std], dim=-1)

            return norm_input.reshape(bs, len, dim), outputs[:, -self.pred_len_new:, :]

        else:
            return input, None

    def de_normalize(self, input, station_pred):
        if 'adaptive' in self.station_type:
            bs, len, dim = input.shape
            input = input.reshape(bs, -1, self.period_len, dim)
            mean = station_pred[:, :, :self.channels].unsqueeze(2)
            std = station_pred[:, :, self.channels:].unsqueeze(2)
            output = input * (std + self.epsilon) + mean
            return output.reshape(bs, len, dim)

        else:
            return input


class MLP(nn.Module):
    def __init__(self, configs, mode):
        super(MLP, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.period_len = configs.token_len
        self.mode = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)
