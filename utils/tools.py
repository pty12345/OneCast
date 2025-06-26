import numpy as np
import torch, os
import matplotlib.pyplot as plt
import time
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

plt.switch_backend('agg')
def adjust_learning_rate(optimizer, epoch, learning_rate, printout=True):
    lr_adjust = {epoch: learning_rate if epoch < 3 else learning_rate * (0.9 ** ((epoch - 3) // 1))}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        self.ignore_index = ignore_index
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, target_mask):

        inputs = inputs[~target_mask]
        targets = targets[~target_mask]

        # 计算交叉熵损失
        return F.cross_entropy(inputs, targets)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    

def draw_similarity(codebook, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """

    plt.clf()

    from sklearn.metrics.pairwise import cosine_distances
    cm = cosine_distances(codebook.T)
    print(cm.shape)
    # exit(0)

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    
    if label_name is not None:
        plt.yticks(range(label_name.__len__()), label_name)
        plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    # for i in range(label_name.__len__()):
    #     for j in range(label_name.__len__()):
    #         color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
    #         value = float(format('%.2f' % cm[j, i]))
    #         plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    plt.clf()
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    # for i in range(label_name.__len__()):
    #     for j in range(label_name.__len__()):
    #         color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
    #         value = float(format('%.2f' % cm[j, i]))
    #         plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)


def plot_token_distribution(gt_tokens: torch.Tensor, pred_tokens: torch.Tensor, save_dir: str):
    _gt_tokens = gt_tokens.flatten().detach().cpu().numpy()
    _pred_tokens = pred_tokens.flatten().detach().cpu().numpy()
    
    # 使用 np.unique 获取数组中每个元素的出现次数
    gt_uni_elements, gt_cnts_elements = np.unique(_gt_tokens, return_counts=True)
    pred_uni_elements, pred_cnts_elements = np.unique(_pred_tokens, return_counts=True)

    plt.clf()

    # 绘制 Groundtruth 的 Token 分布
    plt.bar(gt_uni_elements, gt_cnts_elements, label='GroundTruth')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'gt_token_distribution.png'))
    
    plt.clf()
    
    # 绘制 Prediction 的 Token 分布
    plt.bar(pred_uni_elements, pred_cnts_elements, label='Prediction')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'pred_token_distribution.png'))
    
    plt.clf()
    
    # 绘制 Groundtruth 和 Prediction 的 Token 分布
    plt.bar(gt_uni_elements, gt_cnts_elements, label='GroundTruth')
    plt.bar(pred_uni_elements, pred_cnts_elements, label='Prediction')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'gt_pred_token_distribution.png'))
    
    plt.clf()
    
def plot_token_distribution_with_stratify(gt_tokens: torch.Tensor, pred_tokens: torch.Tensor, \
                save_dir: str, max_token_num=255, dataset='test', freq=True):
    
    os.makedirs(save_dir, exist_ok=True)
    
    _gt_tokens = gt_tokens.flatten().detach().cpu().numpy()
    _pred_tokens = pred_tokens.flatten().detach().cpu().numpy()
    
    # 使用 np.unique 获取数组中每个元素的出现次数
    gt_uni_elements, gt_cnts_elements = np.unique(_gt_tokens, return_counts=True)
    pred_uni_elements, pred_cnts_elements = np.unique(_pred_tokens, return_counts=True)
    
    if freq:
        gt_cnts_elements = gt_cnts_elements / gt_cnts_elements.sum()
        pred_cnts_elements = pred_cnts_elements / pred_cnts_elements.sum()

    plt.clf()

    plt.figure(figsize=(16, 8))

    # 绘制 Groundtruth 的 Token 分布
    plt.bar(gt_uni_elements, gt_cnts_elements, label='GroundTruth')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'gt_token_distribution_on_{dataset}.png'))
    
    plt.clf()
    
    # 绘制 Prediction 的 Token 分布
    plt.bar(pred_uni_elements, pred_cnts_elements, label='Prediction')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'pred_token_distribution_on_{dataset}.png'))
    
    plt.clf()
    
    # 绘制 Groundtruth 和 Prediction 的 Token 分布
    gt_cnts = np.zeros((max_token_num, ))
    gt_cnts[gt_uni_elements] = gt_cnts_elements
    
    pred_cnts = np.zeros((max_token_num, ))
    pred_cnts[pred_uni_elements] = pred_cnts_elements
    
    data1, data2 = gt_cnts, pred_cnts
    
    print('data: ', data1.shape, data2.shape)
    
    data_low = [min(d1, d2) for d1, d2 in zip(data1, data2)]
    data_high = [max(d1, d2) for d1, d2 in zip(data1, data2)]

    colors_low = ['blue' if d1 < d2 else 'orange' for d1, d2 in zip(data1, data2)]
    colors_high = ['orange' if d1 < d2 else 'blue' for d1, d2 in zip(data1, data2)]

    # 设置横坐标
    x = np.arange(len(data1))

    # print(x, data_low, data_high)

    x = np.concatenate((np.array([-1]), x))
    data_low = np.concatenate((np.array([0.0001]), data_low))
    data_high = np.concatenate((np.array([0.0002]), data_high))
    colors_low = ['blue'] + colors_low
    colors_high = ['orange'] + colors_high

    # 绘制柱状图
    data_high = (np.array(data_high) - np.array(data_low)).tolist()
    plt.bar(x, data_low, color=colors_low, label='GroundTruth')
    plt.bar(x, data_high, bottom=data_low, color=colors_high, label='Prediction') 

    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'gt_pred_token_distribution_on_{dataset}.png'))
    
    plt.clf()
    
    
def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()    
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        # x = np.arange(len(preds))
        # plt.plot(x[:16*6], preds[:16*6], linewidth=2)
        # plt.plot(x[16*6:16*7], preds[16*6:16*7], linewidth=2)
        # plt.plot(x[16*7:16*8], preds[16*7:16*8], linewidth=2)
        # plt.plot(x[16*8:16*9], preds[16*8:16*9], linewidth=2)
        # plt.plot(x[16*9:16*10], preds[16*9:16*10], linewidth=2)
        # plt.plot(x[16*10:16*11], preds[16*10:16*11], linewidth=2)
        # plt.plot(x[16*11:16*12], preds[16*11:16*12], linewidth=2)

        plt.plot(preds, label='Prediction', linewidth=2)

    

    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
        
    import thop
    with torch.cuda.device(0):
        flops,params = thop.profile(model.cuda(),inputs=x_shape)
        flops, params = thop.clever_format([flops, params], '%.3f')
        print('flops:', flops)
        print('params:', params)

    # from ptflops import get_model_complexity_info    
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))