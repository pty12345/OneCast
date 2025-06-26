from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_ETT_hour_decomp
from data_provider.data_loader import Dataset_Custom, Dataset_Solar, Dataset_Pred
from data_provider.data_loader import Dataset_PEMS, Dataset_Wind
from data_provider.data_loader import Dataset_TFB
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh1_decomp': Dataset_ETT_hour_decomp,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'wind': Dataset_Wind,
    'TFB': Dataset_TFB,
    'electricity': Dataset_Custom,
    'weather': Dataset_Custom,
    'traffic': Dataset_Custom,
    'CzeLan': Dataset_TFB,
}

data_path_dict = {
    'FRED-MD': 'FRED-MD.csv',
    'NYSE': 'NYSE.csv',
    'Covid-19': 'Covid-19.csv',
    'Wike2000': 'Wike2000.csv',
    'ETTh2': 'ETTh2.csv',
    'ETTm2': 'ETTm2.csv',
    'electricity': 'electricity.csv',
    'weather': 'weather.csv',
    'traffic': 'traffic.csv',
    'CzeLan': 'CzeLan.csv',
}

def data_provider(args, flag):
    
    
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag in ['test', 'pred']:
        shuffle_flag = False
        drop_last = False
        batch_size = args.infer_batch_size if flag == 'test' else 4
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set_dict = {}
    data_loader_dict = {}

    for dataset_name in args.cross_dataset:
        try:
            Data = data_dict[args.data]
        except:
            Data = data_dict[dataset_name]

        if 'ETT' in dataset_name and args.pred_len == 192 and flag == 'test':
            batch_size = 80 # for Multi-GPU training

        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=data_path_dict[dataset_name],
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )
    
        print(flag, len(data_set))
        if len(data_set) < batch_size: 
            batch_size = len(data_set)
            
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        
        data_set_dict[dataset_name] = data_set  
        data_loader_dict[dataset_name] = data_loader
        
    return data_set_dict, data_loader_dict
