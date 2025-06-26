import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import sys
os.chdir(sys.path[0])
parser = argparse.ArgumentParser(description=' for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model', type=str, required=False, default='TokenTime')

# backbone
parser.add_argument('--backbone', type=str, required=False, default='gpt2', help='options:[gpt2, decode_only]')
parser.add_argument('--backbone_dim', type=int, required=False, default=256, help='backbone dimension')
                    
# data loader
parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--valid_ratio', type=float, default=0.1, help='valid ratio')

# cross-dataset
parser.add_argument('--cross_dataset', type=str, default=None, help='cross-dataset')
parser.add_argument('--adaptive_dataset', type=str, default=None, help='adaptive-dataset')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# TSTokenizer
parser.add_argument("--token_len", type=int, default=16)

parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--block_num", type=int, default=2)

parser.add_argument("--n_embed", type=int, default=256)
parser.add_argument("--wave_length", type=int, default=8)
parser.add_argument("--wave_stride", type=int, default=4)

# revin
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')

### VQVAE
parser.add_argument('--vqvae_path', type=str, default=None, help='location of VQVAE checkpoints')
parser.add_argument('--VQ_type', type=str, default='W_SimVQ_decompose', help='type of VQVAE')

parser.add_argument('--elect_rate', type=float, default=1, help='electing rate of vq codebooks')

parser.add_argument('--VQ_epoch', type=int, default=1, help='epoch of VQVAE')


### TokenTime
parser.add_argument('--cfg_path', type=str, default='configs.yaml', help='config')
parser.add_argument('--infer_step', type=int, default=8)

# number of layers of classifier
parser.add_argument('--n_classifier_layer', type=int, default=2)

parser.add_argument('--mask_schedule', type=str, default='cosine', help='mask schedule')
parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio')

### Other Parameter
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') 
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')

parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

### optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--infer_batch_size', type=int, default=128)
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

### unify
parser.add_argument('--is_unify', action='store_true', help='unify train', default=False)
parser.add_argument('--unify_lr', type=float, default=0.0001, help='unify learning rate')

### adaptive
parser.add_argument('--load_unify', type=int, default=0, help='fine-tune based on unify model')
parser.add_argument('--adaptive_lr', type=float, default=0.0001, help='adaptive learning rate')

### GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    num_gpus = torch.cuda.device_count()
    args.device_ids = list(range(num_gpus))
    
    print("Available GPU IDs:", args.device_ids)
    args.gpu = args.device_ids[0]
    # args.gpu = '2'
    
args.dataset = args.data_path.split('.')[0]

if args.cross_dataset is not None:
    # Convert string format [ETTh1,ETTh2] to list
    args.dataset = args.model_id.rsplit('_', 2)[0]
    if args.dataset == 'Cross_small':
        args.cross_dataset = 'FRED-MD,Covid-19,NYSE,Wike2000'
        args.cross_dataset = [x.strip() for x in args.cross_dataset.strip('\"').split(',')]
    elif args.dataset == 'Cross_large':
        args.cross_dataset = 'ETTh2,ETTm2,weather,traffic,CzeLan'
        args.cross_dataset = [x.strip() for x in args.cross_dataset.strip('\"').split(',')]

if args.vqvae_path is None:

    args.vqvae_path = 'TSTokenizer/checkpoints//{}_{}_sl{}_pl{}_dm{}_dr{}_emb{}_wl{}_bl{}_{}_epoch{}'.format(
        args.dataset,
        args.token_len,
        args.seq_len,
        args.pred_len,
        args.d_model,
        args.dropout,
        args.n_embed,
        args.wave_length,
        args.block_num,
        args.VQ_type,
        args.VQ_epoch,
    )
        
print('Args in experiment:')
print(args)
Exp = Exp_Main
mses = []
maes = []

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_er{}_wl{}_tl{}_ne{}_nl{}_{}_Ve{}_Exp{}'.format(args.model_id, args.backbone, args.elect_rate, args.wave_length, args.token_len, args.n_embed, args.n_classifier_layer, args.VQ_type, args.VQ_epoch, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        
        if args.is_unify:
            exp.unify_train(setting, save_root=f"{args.checkpoints}//unify")
            mse,mae = exp.test(setting, save_root=f"{args.checkpoints}//unify")
        else:
            exp.adaptive_train(setting, save_root=f"{args.checkpoints}//adaptive")
            mse,mae = exp.test(setting, save_root=f"{args.checkpoints}//adaptive/{args.adaptive_dataset}")

        # if args.do_predict:
        #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        #     exp.predict(setting, True)

        torch.cuda.empty_cache()
        
    # print('average mse:{0:.4f}±{1:.4f}, mae:{2:.4f}±{3:.4f}'.format(np.mean(
    # mses), np.std(mses), np.mean(maes), np.std(maes))) 
    
else:
    ii = 0
        
    setting = '{}_{}_er{}_wl{}_tl{}_ne{}_nl{}_{}_Ve{}_Exp{}'.format(args.model_id, args.backbone, args.elect_rate, args.wave_length, args.token_len, args.n_embed, args.n_classifier_layer, args.VQ_type, args.VQ_epoch, ii)

    
    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    
    if args.is_unify:
        exp.test(setting, test=1, save_root=f"{args.checkpoints}//unify")
    else:
        exp.test(setting, test=1, save_root=f"{args.checkpoints}//adaptive/{args.adaptive_dataset}")
    torch.cuda.empty_cache()
