import argparse
import os
import json

parser = argparse.ArgumentParser()
# basic config
parser.add_argument("--is_training", type=int, default=1)
parser.add_argument("--save_path", type=str, default=None)
parser.add_argument("--load_path", type=str, default=None)

# dataset
parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

# seq_len
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='label sequence length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# dataloader
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--token_len', type=int, default=336, help='input sequence length')
parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

parser.add_argument('--root_path', type=str, default='./dataset/ETT-small', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=64)

parser.add_argument("--num_workers", type=int, default=10)

# cross-dataset
parser.add_argument('--cross_dataset', type=str, default=None, help='cross-dataset')

# model args
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--enc_in", type=int, default=21)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--block_num", type=int, default=2)

parser.add_argument("--n_embed", type=int, default=256)
parser.add_argument("--wave_length", type=int, default=8)
parser.add_argument("--wave_stride", type=int, default=8)

parser.add_argument("--vq_model", type=str, default='SimVQ', help='options:[SimVQ, VanillaVQ, SimVQ_CNN]')

parser.add_argument("--valid_ratio", type=float, default=0.1, help='valid ratio')

parser.add_argument("--model_id", type=str, default=None, help='model id')

# hyper parameters of training loss calculation
parser.add_argument("--latent_loss_weight", type=float, default=0.25)
parser.add_argument("--trend_loss_weight", type=float, default=0.1)

# Revin
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')

# train args
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--lr_decay_rate", type=float, default=0.99)
parser.add_argument("--lr_decay_steps", type=int, default=300)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--num_epoch", type=int, default=60)
parser.add_argument("--eval_per_steps", type=int, default=300)
parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument("--eval_per_epoch", action="store_true", help="eval per epoch if True")

parser.add_argument('--vq_setting', type=str, default=None, help='VQ setting. e.g., "unfreeze_codebook"')
parser.add_argument('--enable_high_freq_token', action='store_true', help='Enable high frequency token')

args = parser.parse_args()

# Train_data,Test_data = load_ETT(Path="/data/tinyy/vqvae1/dataset/ETT-small",folder=args.data)

args.dataset = args.data_path.split('.')[0]

if args.cross_dataset is not None:
    # Convert string format [ETTh1,ETTh2] to list
    args.dataset = args.model_id
    args.cross_dataset = [x.strip() for x in args.cross_dataset.strip('\"').split(',')]

if args.save_path is None:
    path_str = 'checkpoints//{}_{}_sl{}_pl{}_dm{}_dr{}_emb{}_wl{}_bl{}_{}_epoch{}'.format(
            args.dataset,
            args.token_len,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.dropout,
            args.n_embed,
            args.wave_length,
            args.block_num,
            args.vq_model,
            args.num_epoch
            )
        
    args.save_path = path_str
    
if not os.path.exists(args.save_path):
    print("Creating save dir: {}".format(args.save_path))
    os.makedirs(args.save_path)

with open(args.save_path + "/args.json", "w") as f:
    tmp = args.__dict__
    json.dump(tmp, f, indent=1)
    print(args)
