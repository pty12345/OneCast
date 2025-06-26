import os
import torch
import numpy as np
import glob
from functools import partial
def disable_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def freeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    model = model.eval()
    model.train = disable_train

def normalize(x):
    mean = torch.mean(x, dim=-2, keepdim=True).detach()
    stdev = torch.sqrt(torch.var(x, dim=-2, keepdim=True, unbiased=False) + 1e-8).detach()
    return (x - mean) / stdev, mean, stdev

def euclidean_distance(x, y, norm=False, independent=False):
    if norm:
        x, mean_x, stdev_x = normalize(x)
        y, mean_y, stdev_y = normalize(y)

    if independent == False:
        _x = x.view(x.shape[0], 1, -1)  # [B1, 1, N]
        _y = y.view(1, y.shape[0], -1)  # [1, B2, N]

        dist = torch.norm(_x - _y, p=1, dim=-1)  # [B1, B2]
        # print('dist.shape: ', dist.shape)
        # exit(0)
        return dist
    

    else:
        raise NotImplementedError('Euclidean distance with independent is not implemented')


def cosine_distance(x, y, norm=False, independent=False):
    if norm:
        x, mean_x, stdev_x = normalize(x)
        y, mean_y, stdev_y = normalize(y)
    
    if independent == False:
        _x = x.view(x.shape[0], -1)  # [B1, N]
        _y = y.view(y.shape[0], -1)  # [B2, N]
        
        similarity = torch.mm(_x, _y.t())  # [B1, B2]
        
        return -similarity # the smaller, the more similar
    
    else:
        _x = x.permute(2, 0, 1)  # [C, B1, L]
        _y = y.permute(2, 1, 0)  # [C, L, B2]
        
        similarity = torch.matmul(_x, _y)  # [C, B1, B2]
        
        return -similarity # the smaller, the more similar

def get_distance_func(distance_func='cosine', norm=False, independent=False):
    if distance_func == 'cosine':
        return partial(cosine_distance, norm=norm, independent=independent)
    elif distance_func == 'euclidean':
        return partial(euclidean_distance, norm=norm, independent=independent)
    else:
        raise ValueError(f"Distance function {distance_func} not supported")

def retrieval_neibor_by_topk(batch, data_loader, topk, distance_func='relative', pred_len=96):
    """
    batch_x: [B, L, D]
    data_loader: 
    topk: 
    distance_func: 
    """
    batch_x, batch_y = batch

    distance_func = get_distance_func(distance_func)
    
    dists, data_x_list, data_y_list = [], [], []
    for data_x, data_y, _, _ in data_loader:
        data_x = data_x.float().to(batch_x.device)
        data_y = data_y.float().to(batch_x.device)
        distance = distance_func(batch_x, data_x)  # [B1, B2]
        dists.append(distance)
        data_x_list.append(data_x)
        data_y_list.append(data_y)

    dists = torch.cat(dists, dim=-1)
    data_x = torch.cat(data_x_list, dim=0)
    data_y = torch.cat(data_y_list, dim=0)

    neibor_x, neibor_y = [], []
    top_dists = []

    for i in range(dists.shape[0]):
        top_dist, indices = torch.topk(dists[i], k=topk, largest=False)

        top_dists.append(top_dist.unsqueeze(0))
        # print(f'dists[{i}]:', _)

        neibor_x.append(data_x[indices].unsqueeze(0))
        neibor_y.append(data_y[indices].unsqueeze(0))

    neibor_x = torch.cat(neibor_x, dim=0)
    neibor_y = torch.cat(neibor_y, dim=0)
    neibor_y = neibor_y[:, :, -pred_len:, :]

    top_dists = torch.cat(top_dists, dim=0)

    # # ## plot
    save_dir = 'retrieval_series'
    os.makedirs(save_dir, exist_ok=True)

    import matplotlib.pyplot as plt
    _neibor_x = neibor_x.detach().cpu().numpy()
    _neibor_y = neibor_y.detach().cpu().numpy()
    print('neibor_x.shape: ', _neibor_x.shape)
    print('neibor_y.shape: ', neibor_y.shape)
    print('top_dists.shape: ', top_dists.shape)

    plt.clf()
    series_x = batch_x[0, :, -1].detach().cpu().numpy()
    series_y = batch_y[0, -pred_len:, -1].detach().cpu().numpy()
    
    assert _neibor_x.shape[1] == topk, f'{_neibor_x.shape[1]} != {topk}'
    for i in range(_neibor_x.shape[1]):
        neibor = np.concatenate((_neibor_x[0, i, :, -1], _neibor_y[0, i, :, -1]))
        plt.plot(neibor, label=f'series_y_{i}')

    plt.plot(np.concatenate((series_x, series_y)), label='series_input', color='black', linewidth=2)

    plt.legend()

    # Find the highest index among existing files
    existing_files = glob.glob(f'{save_dir}/retrieval_series_on_example_*.png')
    max_idx = -1
    for f in existing_files:
        try:
            idx_str = f.split('example_')[-1].split('.png')[0]
            max_idx = max(max_idx, int(idx_str))
        except:
            continue
    
    idx = max_idx + 1
    plt.savefig(f'{save_dir}/retrieval_series_on_example_{idx}.png')

    plt.close()

    dist_recoder_file = os.path.join(save_dir, 'dist_recoder.txt')
    with open(dist_recoder_file, 'a') as f:
        f.write(f"png_{idx}'s top dist: {top_dists[0]}\n\n")

    return neibor_y

def retrieval_neibor_by_topk_to_fixed(batch, data_loader, topk, distance_func='relative', pred_len=96):
    """
    batch_x: [B, L, D]
    data_loader: 
    topk: 
    distance_func: 
    """
    batch_x, batch_y = batch

    distance_func = get_distance_func(distance_func, norm=True)
    
    dists, data_x_list, data_y_list = [], [], []
    for data_x, data_y, _, _ in data_loader:
        data_x = data_x.float().to(batch_x.device)
        data_y = data_y.float().to(batch_x.device)
        distance = distance_func(batch_x, data_x)  # [B1, B2]
        dists.append(distance)
        data_x_list.append(data_x)
        data_y_list.append(data_y)

    dists = torch.cat(dists, dim=-1)
    data_x = torch.cat(data_x_list, dim=0)
    data_y = torch.cat(data_y_list, dim=0)

    neibor_x, neibor_y = [], []
    top_dists = []

    for i in range(dists.shape[0]):
        top_dist, indices = torch.topk(dists[i], k=topk, largest=False)

        top_dists.append(top_dist.unsqueeze(0))
        # print(f'dists[{i}]:', top_dist)
        # exit(0)

        neibor_x.append(data_x[indices].unsqueeze(0))
        neibor_y.append(data_y[indices].unsqueeze(0))

    neibor_x = torch.cat(neibor_x, dim=0)
    neibor_y = torch.cat(neibor_y, dim=0)
    neibor_y = neibor_y[:, :, -pred_len:, :]

    top_dists = torch.cat(top_dists, dim=0)

    data_dict = {
        'neibor_x': neibor_x,
        'neibor_y': neibor_y,
        'top_dists': top_dists,
        'batch_x': batch_x,
        'batch_y': batch_y,
        'topk': topk,
        'pred_len': pred_len,
    }

    # plot_neibor_series(data_dict)

    return neibor_x, neibor_y

def retrieval_neibor_by_topk_to_fixed_independent(batch, data_loader, topk, distance_func='relative', pred_len=96):
    """
    batch_x: [B, L, D]
    data_loader: 
    topk: 
    distance_func: 
    """
    batch_x, batch_y = batch

    distance_func = get_distance_func(distance_func, norm=True, independent=True)
    
    dists, data_x_list, data_y_list = [], [], []
    for data_x, data_y, _, _ in data_loader:
        data_x = data_x.float().to(batch_x.device)
        data_y = data_y.float().to(batch_x.device)
        distance = distance_func(batch_x, data_x)  # [B1, B2]
        dists.append(distance)
        data_x_list.append(data_x)
        data_y_list.append(data_y)

    dists = torch.cat(dists, dim=-1) # [C, B1, B_all]
    data_x = torch.cat(data_x_list, dim=0) # [B_all, L, C]
    data_y = torch.cat(data_y_list, dim=0) # [B_all, L, C]

    dists = dists.permute(1, 2, 0) # [B1, B_all, C]
    data_x = data_x.permute(0, 2, 1) # [B_all, C, L]
    data_y = data_y.permute(0, 2, 1) # [B_all, C, L]

    # print('dists.shape: ', dists.shape)
    # print('data_x.shape: ', data_x.shape)
    # print('data_y.shape: ', data_y.shape)

    # exit(0)

    neibor_x, neibor_y = [], []
    top_dists = []

    C = batch_x.shape[-1]
    indices_channel = torch.arange(C).unsqueeze(0).repeat(topk, 1)
    indices_channel = indices_channel.flatten()

    for i in range(dists.shape[0]):
        top_dist, indices = torch.topk(dists[i], k=topk, largest=False, dim=0)

        top_dists.append(top_dist.unsqueeze(0))
        # print(f'dists[{i}]:', _)

        # print('indices', indices)
        indices_topk = indices.flatten()

        # print('indices_topk', indices_topk)


        # print('indices_channel', indices_channel)

        _neibor_x = data_x[indices_topk, indices_channel, :].reshape(topk, C, -1)
        _neibor_y = data_y[indices_topk, indices_channel, :].reshape(topk, C, -1)

        neibor_x.append(_neibor_x.unsqueeze(0))
        neibor_y.append(_neibor_y.unsqueeze(0))

    neibor_x = torch.cat(neibor_x, dim=0).permute(0, 1, 3, 2)
    neibor_y = torch.cat(neibor_y, dim=0).permute(0, 1, 3, 2)


    neibor_y = neibor_y[:, :, -pred_len:, :]

    top_dists = torch.cat(top_dists, dim=0)

    data_dict = {
        'neibor_x': neibor_x,
        'neibor_y': neibor_y,
        'top_dists': top_dists,
        'batch_x': batch_x,
        'batch_y': batch_y,
        'topk': topk,
        'pred_len': pred_len,
    }

    # plot_neibor_series(data_dict)

    return neibor_x, neibor_y

def eval_best(batch_x, batch_y, data_x, data_y, token_len=16):
    """
    batch_x: [B, L, D]
    data_loader: 
    topk: 
    distance_func: 
    """

    B1, L, C = batch_x.shape
    B2, _, _ = data_x.shape

    mean_x = batch_x.mean(dim=-2, keepdim=True)
    stdev_x = batch_x.std(dim=-2, keepdim=True, unbiased=False) + 1e-8

    # print('mean_x.shape: ', mean_x.shape)
    # print('stdev_x.shape: ', stdev_x.shape)

    mean_data_x = data_x.mean(dim=-2, keepdim=True)
    stdev_data_x = data_x.std(dim=-2, keepdim=True, unbiased=False) + 1e-8

    # print('mean_data_x.shape: ', mean_data_x.shape)
    # print('stdev_data_x.shape: ', stdev_data_x.shape)

    token_data_y = data_y.reshape(B2, -1, token_len, C)
    mean_token_data_y = token_data_y.mean(dim=-2)
    stdev_token_data_y = token_data_y.std(dim=-2, unbiased=False) + 1e-8

    # print('mean_token_data_y.shape: ', mean_token_data_y.shape)
    # print('stdev_token_data_y.shape: ', stdev_token_data_y.shape)

    relative_mean_data_y = (mean_token_data_y - mean_data_x)
    relative_stdev_data_y = (stdev_token_data_y - stdev_data_x)

    # print('relative_mean_data_y.shape: ', relative_mean_data_y.shape)
    # print('relative_stdev_data_y.shape: ', relative_stdev_data_y.shape)

    mean_x = mean_x.unsqueeze(1) # [B1, 1, 1, C]
    relative_mean_data_y = relative_mean_data_y.unsqueeze(0) # [1, B2, fixed_len, C]

    stdev_x = stdev_x.unsqueeze(1) # [B1, 1, 1, C]
    relative_stdev_data_y = relative_stdev_data_y.unsqueeze(0) # [1, B2, fixed_len, C]

    # print('mean_x.shape: ', mean_x.shape)
    # print('relative_mean_data_y.shape: ', relative_mean_data_y.shape)

    pred_mean_batch_y = mean_x + relative_mean_data_y
    pred_stdev_batch_y = stdev_x + relative_stdev_data_y

    # print('pred_mean_batch_y.shape: ', pred_mean_batch_y.shape)
    # print('pred_stdev_batch_y.shape: ', pred_stdev_batch_y.shape)

    token_batch_y = batch_y.reshape(B1, -1, token_len, C)

    gt_mean_batch_y = token_batch_y.mean(dim=-2)
    gt_stdev_batch_y = token_batch_y.std(dim=-2, unbiased=False) + 1e-8

    gt_mean_batch_y = gt_mean_batch_y.unsqueeze(1)
    gt_stdev_batch_y = gt_stdev_batch_y.unsqueeze(1)

    mse_mean = ((gt_mean_batch_y - pred_mean_batch_y) ** 2).mean(dim=(-1, -2))
    mse_stdev = ((gt_stdev_batch_y - pred_stdev_batch_y) ** 2).mean(dim=(-1, -2))

    best_mean_idx = torch.argmin(mse_mean, dim=-1)
    best_stdev_idx = torch.argmin(mse_stdev, dim=-1)

    return best_mean_idx, best_stdev_idx
    
    # print('pred_mean_batch_y.shape: ', pred_mean_batch_y.shape)
    # print('gt_mean_batch_y.shape: ', gt_mean_batch_y.shape)
    # print('pred_stdev_batch_y.shape: ', pred_stdev_batch_y.shape)
    # print('gt_stdev_batch_y.shape: ', gt_stdev_batch_y.shape)

def retrieval_neibor_by_topk_to_fixed_breach(batch, data_loader, topk, distance_func='relative', pred_len=96):
    """
    batch_x: [B, L, D]
    data_loader: 
    topk: 
    distance_func: 
    """
    batch_x, batch_y = batch

    distance_func = get_distance_func(distance_func, norm=True)
    
    dists, data_x_list, data_y_list = [], [], []
    for data_x, data_y, _, _ in data_loader:
        data_x = data_x.float().to(batch_x.device)
        data_y = data_y.float().to(batch_x.device)
        distance = distance_func(batch_x, data_x)  # [B1, B2]
        dists.append(distance)
        data_x_list.append(data_x)
        data_y_list.append(data_y)

    dists = torch.cat(dists, dim=-1)
    data_x = torch.cat(data_x_list, dim=0)
    data_y = torch.cat(data_y_list, dim=0)

    batch_y = batch_y[:, -pred_len:, :]
    data_y = data_y[:, -pred_len:, :]

    best_mean_idx, best_stdev_idx = eval_best(batch_x, batch_y, data_x, data_y)

    # just mean
    neibor_x = data_x[best_mean_idx]
    neibor_y = data_y[best_mean_idx]

    neibor_x = neibor_x.unsqueeze(1)
    neibor_y = neibor_y.unsqueeze(1)
    # dists = dists[best_mean_idx]

    # print('neibor_x.shape: ', neibor_x.shape)
    # print('neibor_y.shape: ', neibor_y.shape)
    # print('dists.shape: ', dists.shape)

    # exit(0)

    return neibor_x, neibor_y
