import torch
import torch.nn as nn

# %% ../../nbs/035_models.InceptionTime.ipynb 4
# This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@timeseriesAI.co based on:

# Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., ... & Petitjean, F. (2019). 
# InceptionTime: Finding AlexNet for Time Series Classification. arXiv preprint arXiv:1909.04939.
# Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime

class Concat(nn.Module):
    def __init__(self, dim=1): 
        super().__init__()
        self.dim = dim
    def forward(self, *x): return torch.cat(*x, dim=self.dim)
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'

class Add(nn.Module):
    def forward(self, x, y): 
        super().__init__()
        return x.add(y)
    def __repr__(self): 
        return f'{self.__class__.__name__}'

class ConvBlock(nn.Sequential):
    "Create a sequence of conv1d (`ni` to `nf`), activation (if `act_cls`) and `norm_type` layers."
    def __init__(self, ni, nf, ks=3, stride=1, padding='same', norm='Batch'):
        super().__init__()
        layers = []
        layers.append(nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=padding))
        layers.append(nn.BatchNorm1d(nf))
        super().__init__(*layers)

class Reshape(nn.Module):
    def __init__(self, *shape): 
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.contiguous().reshape(x.shape[0], -1) if not self.shape else x.contiguous().reshape(-1) if self.shape == (-1,) else x.contiguous().reshape(x.shape[0], *self.shape)
    def __repr__(self): return f"{self.__class__.__name__}({', '.join(['bs'] + [str(s) for s in self.shape])})"

class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"
    def __init__(self, output_size=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Reshape()
    def forward(self, x):
        return self.flatten(self.gap(x))

class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True):
        super().__init__()
        ks = [ks // (2**i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        bottleneck = bottleneck if ni > 1 else False
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False) if bottleneck else nn.Identity()
        self.convs = nn.ModuleList([nn.Conv1d(nf if bottleneck else ni, nf, k, bias=False, padding='same') for k in ks])
        self.maxconvpool = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(ni, nf, 1, bias=False)])
        self.concat = Concat()
        self.bn = nn.BatchNorm1d(nf * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)

        x = self.concat([l(x) for l in self.convs] + [self.maxconvpool(input_tensor)])

        return self.act(self.bn(x))

class InceptionBlock(nn.Module):
    def __init__(self, ni, nf=32, residual=True, depth=6, **kwargs):
        super().__init__()
        self.residual, self.depth = residual, depth
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for d in range(depth):
            self.inception.append(InceptionModule(ni if d == 0 else nf * 4, nf, **kwargs))
            if self.residual and d % 3 == 2: 
                n_in, n_out = ni if d == 2 else nf * 4, nf * 4
                self.shortcut.append(nn.BatchNorm1d(n_in) if n_in == n_out else ConvBlock(n_in, n_out, ks=1))
        self.add = Add()
        self.act = nn.ReLU()
        
    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2: res = x = self.act(self.add(x, self.shortcut[d//3](res)))
        return x

class InceptionTime(nn.Module):
    def __init__(self, c_in, c_out, seq_len=None, nf=64):
        super().__init__()
        nf = c_out // 4
        self.inceptionblock = InceptionBlock(c_in, nf)
        # self.gap = GAP1d(1)

    def forward(self, x):
        x = self.inceptionblock(x)
        # x = self.gap(x)
        return x
    
if __name__ == '__main__':
    model = InceptionTime(c_in=64, c_out=64)
    x = torch.randn(256, 64, 96) # (batch_size, channels, sequence_length)
    print(model(x).shape)
    exit(0)