import torch
import torch.nn as nn

def get_frequencies():
    # freq predictor
    frequencies = torch.concatenate((1440/torch.arange(5, 61, 10), 
                24/torch.arange(1, 48, 0.5),
                1/torch.arange(2, 28, 0.5),
                1/torch.arange(28, 52*7, 7), 
                1/torch.arange(52*7, 52*7*10+1, 26*7))
            )
	
    return frequencies

class Season_predictor(nn.Module):
	def __init__(self, seq_len, pred_len, n_domain, dropout=0.1):
		
		super().__init__()
		
		# data config
		self.seq_len = seq_len
		self.pred_len = pred_len

		self.n_domain = n_domain
  
		# print(frequencies)
		# exit(0)
  
		# Basis frequencies, shape [1, 1, 432]
		frequencies = get_frequencies()

		self.shape_num = frequencies.shape[-1] # 
		# self.register_buffer('frequencies', frequencies)
		self.frequencies = nn.Parameter(frequencies, requires_grad=True)

		self.temp = nn.Parameter(torch.tensor([200 / self.shape_num], requires_grad=True))

		# PRED = sigma( A * cos(wx + b_1) + B * sin(wx + b_2))

		self.sin_coeffies = nn.Sequential(
			nn.Linear(self.seq_len, 64),
			nn.GELU(),
			nn.Linear(64, self.shape_num * 2)
		)
  
		self.cos_coeffies = nn.Sequential(
			nn.Linear(self.seq_len, 64),
			nn.GELU(),
			nn.Linear(64, self.shape_num * 2)
		)

		self.sin_coeffies_list = nn.ModuleList([self.sin_coeffies] * self.n_domain)
		self.cos_coeffies_list = nn.ModuleList([self.cos_coeffies] * self.n_domain)
  
		# self.sin_coeffies = nn.Linear(self.seq_len, self.shape_num * 2)
		# self.cos_coeffies = nn.Linear(self.seq_len, self.shape_num * 2)
  
		self.dropout = nn.Dropout(dropout)
		
	def forward(self, z, dataset_id):  # z: [bs x seq_len x nvars]
	 
		# channel independent
		B, L, C = z.shape
		z = z.permute(0, 2, 1) # [B, C, L]
  
		sin_coeffs = self.dropout(self.sin_coeffies_list[dataset_id](z)).reshape(B, 1, C, -1, 2)
		cos_coeffs = self.dropout(self.cos_coeffies_list[dataset_id](z)).reshape(B, 1, C, -1, 2)
		
		# print(sin_coeffs.shape, cos_coeffs.shape)
  
		times = torch.arange(0, self.pred_len, 1).to(z.device).reshape(-1, 1, 1)
  
		# calc sin & cos
		sin_w, sin_b = sin_coeffs[:, :, :, :, 0], sin_coeffs[:, :, :, :, 1] 
		cos_w, cos_b = cos_coeffs[:, :, :, :, 0], cos_coeffs[:, :, :, :, 1] 
  
		# exit(0)

		sins = sin_w * torch.sin(2*torch.pi*self.frequencies*times).unsqueeze(0) #  + sin_b
		coss = cos_w * torch.cos(2*torch.pi*self.frequencies*times).unsqueeze(0) #  + cos_b

		pred = sins.sum(-1) + coss.sum(-1)
  
		pred = pred * self.temp

		return pred
	
# class Season_predictor_linear(nn.Module):
# 	def __init__(self, seq_len, pred_len, dropout=0.1):
		
# 		super().__init__()
		
# 		self.linear = nn.Linear(seq_len, pred_len)

# 		# data config
# 		self.seq_len = seq_len
# 		self.pred_len = pred_len
		
# 	def forward(self, z):  # z: [bs x seq_len x nvars]
	 
# 		# channel independent
# 		B, L, C = z.shape
# 		z = z.permute(0, 2, 1).reshape(B*C, 1, L)

# 		output = self.linear(z)

# 		pred = output.reshape(B, C, -1).permute(0,2,1)

# 		return pred