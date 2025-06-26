from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute
from data_provider.data_loader import Dataset_ETT_hour_decomp
from data_provider.data_loader import Dataset_Custom, Dataset_Solar, Dataset_Pred
from data_provider.data_loader import Dataset_PEMS, Dataset_Wind
from data_provider.data_loader import Dataset_TFB
from torch.utils.data import DataLoader
from collections import Counter


from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'TFB': Dataset_TFB,
    'CzeLan': Dataset_TFB,
    'ETTh1_decomp': Dataset_ETT_hour_decomp,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'weather': Dataset_Custom,
    'traffic': Dataset_Custom,
    'electricity': Dataset_Custom,
}

data_path_dict = {
    'FRED-MD': 'FRED-MD.csv',
    'NYSE': 'NYSE.csv',
    'Covid-19': 'Covid-19.csv',
    'Wike2000': 'Wike2000.csv',
    'ETTh2': 'ETTh2.csv',
    'ETTm2': 'ETTm2.csv',
    'CzeLan': 'CzeLan.csv',
    'weather': 'weather.csv',
    'traffic': 'traffic.csv',
    'electricity': 'electricity.csv',
}


def data_provider(args, flag):

    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.test_batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.train_batch_size
        freq = args.freq

    data_set_dict = {}
    data_loader_dict = {}

    for dataset_name in args.cross_dataset:
        try:
            Data = data_dict[args.data]
        except:
            Data = data_dict[dataset_name]

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
