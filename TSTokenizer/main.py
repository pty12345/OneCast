import json
import torch
import random
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')
from data_provider.data_factory import data_provider
from args import args
from process import Trainer
from models.W_SimVQ_decompose import W_SimVQ_decompose
import torch.utils.data as Data

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def get_data(flag):
        data_set, data_loader = data_provider(args, flag)
        return data_set, data_loader

# Feature dimension of each dataset
dim_table = {
    'FRED-MD': 107,
    'NYSE': 5,
    'Covid-19': 948,
    'Wike2000': 2000,
    'ETTh2': 7,
    'ETTm2': 7,
    'electricity': 321,
    'weather': 21,
    'traffic': 862,
    'CzeLan': 11,
}

def main():
    seed_everything(seed=2024)

    # train_dataset = Dataset(device=args.device, mode='train', args=args)
    # train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    # test_dataset = Dataset(device=args.device, mode='test', args=args)
    # test_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    train_data_dict, train_loader_dict = get_data(flag='train')
    vali_data_dict, vali_loader_dict = get_data(flag='val')
    test_data_dict, test_loader_dict = get_data(flag='test')

    print('dataset initial ends')
    # model = VQVAE(data_shape=(args.token_len, args.enc_in), hidden_dim=args.d_model, n_embed=args.n_embed,
    #                 wave_length=args.wave_length, block_num=args.block_num)
    
    # model = W_VQVAE(args)
    
    if args.vq_model == 'W_SimVQ_decompose':
        model = W_SimVQ_decompose(args, dim_table)
    else:
        raise ValueError('Invalid VQ model name')
    
    print('model initial ends')

    trainer = Trainer(args, model, train_loader_dict, vali_loader_dict, test_loader_dict, verbose=True)
    print('trainer initial ends')

    if args.is_training:
        trainer.train()

    trainer.test()


if __name__ == '__main__':
    main()

