import shutil
import pickle
import random
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import TokenTime
from layers.W_SimVQ_decompose_cross import W_SimVQ_decompose_cross
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.tools import plot_token_distribution, plot_token_distribution_with_stratify
from utils.metrics import metric, token_metric

from utils.sampling import get_mask_chedule
from utils.mask_utils import mask_or_random_replace_tokens
from utils.tools import CustomCrossEntropyLoss

import numpy as np
import torch
import torch.nn as nn

from pprint import pprint
from torch import optim
from omegaconf import OmegaConf
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from utils.dlutils import freeze_model

warnings.filterwarnings('ignore')

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

dataset_id_map = {
    # Cross small
    'FRED-MD': 0,
    'Covid-19': 1,
    'NYSE': 2,
    
    'Wike2000': 0,

    # Cross large
    'ETTh2': 0,
    'ETTm2': 1,
    'weather': 2,

    'traffic': 0,
    
    'CzeLan': 0,
}

pretrain_map = {
    "Cross_FRED_Covid_NYSE": ["FRED-MD", "Covid-19", "NYSE"],
    "Cross_Wike2000": ["Wike2000"],

    "Cross_ETTh2_ETTm2_weather": ["ETTh2", "ETTm2", "weather"],
    "Cross_traffic": ["traffic"],
    "Cross_CzeLan": ["CzeLan"],
}

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        
        self.extend_config = OmegaConf.load(self.args.cfg_path)
        self.mask_id = self.model.module.ts_tokenizer_len

        self.epsilon = 1e-5
        # self.saved_metric = self.args.saved_metric
        self.saved_metric = 'mse'

        self.dataset_id_map = dataset_id_map

    def _build_model(self):
        model_dict = {
            'TokenTime': TokenTime
        }
        
        # Load and freeze Time Series Tokenizer
        codebook_sz_dict = {
            "Cross_FRED_Covid_NYSE": 32,
            "Cross_Wike2000": 32,
            "Cross_ETTh2_ETTm2_weather": 256,
            "Cross_traffic": 128,
            "Cross_CzeLan": 128,
        }
        
        if self.args.VQ_type == 'W_SimVQ_decompose':
            vq_model = W_SimVQ_decompose_cross(self.args, dim_table, dataset_id_map, pretrain_map, codebook_sz_dict)
        else:
            raise ValueError("VQ type@ {} not supported!".format(self.args.VQ_type))
        
        
        self.args.elected_n_embed = vq_model.load_pretrain_VQ()
        
        freeze_model(vq_model)
        
        # Load backbone
        model = model_dict[self.args.model].Model(self.args).float()
        model.init_ts_tokenizer(vq_model)

        if self.args.use_multi_gpu and self.args.use_gpu:
            print('Using multi-gpu training')
            print('Available GPU IDs:', self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            vq_model = nn.DataParallel(vq_model, device_ids=self.args.device_ids)
            
        return model, vq_model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, lr):
        model_optim = optim.Adam(self.model.parameters(), lr=lr)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        
        return criterion
    
    def _select_mask_scheduler(self):
        mask_schedule = get_mask_chedule(self.args.mask_schedule, ratio=self.args.mask_ratio)
        return mask_schedule
    
    @torch.no_grad()
    def prepare_inputs_and_labels(self, batch_x, batch_y, mask_schedule, \
                                min_masking_rate=0.0, is_train=True, dataname=None):
        
        # print('batch: ', batch_x.shape, batch_y.shape)

        # input_tokens = self.vq_model.get_code(batch_x) + self.model.text_tokenizer_len
        # output_tokens = self.vq_model.get_code(batch_y) + self.model.text_tokenizer_len

        output_tokens = self.vq_model.module.get_code(batch_y, dataname=dataname)
        input_tokens = self.vq_model.module.get_code(batch_x, dataname=dataname)

        # print('input_tokens: ', input_tokens.shape, output_tokens.shape)
        # exit(0)
        
        # print('input_tokens: ', input_tokens.shape, output_tokens.shape)
        # exit(0)
        
        if not is_train: # for testing
            masked_output_tokens = torch.ones_like(output_tokens) * self.mask_id
            input_ids = torch.cat((input_tokens, masked_output_tokens), dim=-1)
            out_token_shape = output_tokens.shape
            
            return input_ids, output_tokens, out_token_shape
        
        # print('token: ', input_tokens.shape, output_tokens.shape)

        # create MLM mask and labels
        masked_output_tokens, output_labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
            output_tokens,
            self.mask_id,
            self.extend_config,
            mask_schedule=mask_schedule,
            is_train=is_train,
        )
        
        # print('masked_output_tokens: ', masked_output_tokens.shape)
        # print('output_labels: ', output_labels.shape)
        # print('loss_weight: ', loss_weight)
        # print('mask_prob: ', mask_prob)
        
        # print('mask_out_0', masked_output_tokens[0])
        # print('out_0', output_labels[0])
        
        input_ids = torch.cat((input_tokens, masked_output_tokens), dim=-1)
            
        # set labels
        input_labels = (torch.ones_like(input_tokens) * -100)
        labels = torch.cat((input_labels, output_labels), dim=-1)

        return input_ids, labels, mask_prob, output_tokens

    @torch.no_grad()
    def decode_ts(self, output_ids, B, look_back=None, dataname=None):
        # output_ids: [B_C, num*num_t]
        B, n_nt = output_ids.shape
        output_ids = torch.reshape(output_ids, (-1, self.vq_model.module.num_t))
        
        decode_ts = self.vq_model.module.decode_ids(output_ids, dataname=dataname).squeeze() # [B_C_num_t, token_len]
        decode_ts = torch.reshape(decode_ts, (B, -1, decode_ts.shape[-1]))

        if 'decompose' in self.args.VQ_type:
            with torch.no_grad():
                pred_season = self.vq_model.module.pred_season(look_back, dataname=dataname)

                decode_ts = decode_ts + pred_season

        else:
            raise ValueError(f'VQ type {self.args.VQ_type} not supported!')
        
        # denormalize
        decode_ts = self.vq_model.module.de_norm(decode_ts, dataname=dataname)
        return decode_ts

    def vali(self, vali_loader_dict, criterion, mask_schedule=None, mode='e2e'):
        metrics = {"loss": {}, "mse": {}, "mae": {}, "acc": {}}
        self.model.eval()
        with torch.no_grad():
            whole_loss = []
            avg_mse = 0
            for dataname, vali_loader in vali_loader_dict.items():  
                total_loss = []
                for j, (batch_x, batch_y, _, _) in tqdm(enumerate(vali_loader)):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.device)
                            
                    ts_ids, labels, mask_prob, _ = \
                        self.prepare_inputs_and_labels(batch_x, batch_y, mask_schedule, is_train=True, dataname=dataname)

                    # print('mask_prob: ', mask_prob.shape)
                    # print('=====================')
                    
                    inputs = {
                        'text_ids': None,
                        'ts_ids': ts_ids.to(self.device)
                    }
                    # exit(0)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(inputs)
                    else:
                        outputs = self.model(inputs)
                        
                    outputs = outputs.view(-1, outputs.size(-1))
                    # labels = labels.view(-1)
                    labels = labels.view(-1)
                    
                    loss = criterion(outputs, labels)
                        
                    total_loss.append(loss.item())
                
                whole_loss.extend(total_loss)

                metrics["loss"].update({dataname: np.average(total_loss)})

                # record mse
                preds, trues, _, output_tokens, gt_tokens = self.test_func(vali_loader, None, object='vali', dataname=dataname)
                preds = np.concatenate(preds, axis=0)
                trues = np.concatenate(trues, axis=0)

                preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
                trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

                # print(preds.shape, trues.shape)
                # exit(0)
                
                mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

                token_metric_dict = token_metric(output_tokens, gt_tokens)
                metrics["mse"].update({dataname: mse})
                metrics["mae"].update({dataname: mae})
                metrics["acc"].update({dataname: token_metric_dict['accuracy']})

                avg_mse += mse

        metrics["loss"].update({"whole_loss": np.average(whole_loss)})
        metrics["mse"].update({"avg_mse": avg_mse / len(vali_loader_dict)})

        self.model.train()
        
        return metrics

    def get_data_iter(self, data_loader_dict):
        data_iters = {}
    
        dataset_list, r = [], []
        for idx, dataname in enumerate(data_loader_dict.keys()):
            train_loader = data_loader_dict[dataname]

            length = len(train_loader)
            r = r + [idx] * length

            data_iters.update({dataname:iter(train_loader)})
            dataset_list.append(dataname)

        return data_iters, dataset_list, r

    def unify_train(self, setting, save_root='checkpoints'):
        self.saved_metric = 'entropy'

        print('\n################# Unify Train #################')

        train_data_dict, train_loader_dict = self._get_data(flag='train')
        vali_data_dict, vali_loader_dict = self._get_data(flag='val')
        test_data_dict, test_loader_dict = self._get_data(flag='test')

        
        path = os.path.join(save_root, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer(lr=self.args.unify_lr)
        criterion = self._select_criterion()
        mask_schedule = self._select_mask_scheduler()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            train_gt_id_list, train_pred_id_list = [], []

            data_iters, dataset_list, r = self.get_data_iter(train_loader_dict)
            random.shuffle(r)

            for i in tqdm(r, desc=f"Epoch {epoch+1}", ncols=120):
                batch_x, batch_y, batch_x_mark, batch_y_mark = next(data_iters[dataset_list[i]])

                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.device)
                
                ts_ids, labels, mask_prob, gt_tokens = \
                    self.prepare_inputs_and_labels(batch_x, batch_y, mask_schedule, is_train=True, dataname=dataset_list[i])
                
                train_gt_id_list.append(gt_tokens)
                # print('break 2')
                # exit(0)
                
                inputs = {
                    'text_ids': None,
                    'ts_ids': ts_ids.to(self.device)
                }
                # exit(0)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                pred_ids = torch.argmax(outputs, dim=-1)
                train_pred_id_list.append(pred_ids[:, -self.args.pred_len // self.args.wave_length:])
                    
                outputs = outputs.view(-1, outputs.size(-1))

                # print('outputs: ', outputs.shape)
                # print('labels: ', labels.shape)
                # exit(0)

                labels = labels.view(-1)
                loss = criterion(outputs, labels)
                train_loss.append(loss.item())
                
                # print(loss.item())
                # exit(0)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * len(r) - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # report the accuracy of the train set
            train_gt_id_list = torch.cat(train_gt_id_list, dim=0)
            train_pred_id_list = torch.cat(train_pred_id_list, dim=0)

            train_metrics = token_metric(train_pred_id_list, train_gt_id_list)
            
            print("\n\n### Epoch: {} cost time: {} ###".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_metrics = self.vali(vali_loader_dict, criterion, mask_schedule)
            test_metrics = self.vali(test_loader_dict, criterion, mask_schedule)

            print("Epoch: {0}, Steps: {1}, Train Loss: {2:.7f}".format(epoch + 1, len(r), train_loss))
            for dataname in vali_loader_dict.keys():
                print("{0} | Vali Loss: {1:.7f} Test Loss: {2:.7f} | Vali MSE: {3:.7f} Vali MAE: {4:.7f} Vali ACC: {5:.7f} Test MSE: {6:.7f} Test MAE: {7:.7f} Test ACC: {8:.7f}".format(dataname, vali_metrics['loss'][dataname], test_metrics['loss'][dataname], vali_metrics['mse'][dataname], vali_metrics['mae'][dataname], vali_metrics['acc'][dataname], test_metrics['mse'][dataname], test_metrics['mae'][dataname], test_metrics['acc'][dataname]))
            
            if self.saved_metric == 'entropy':
                early_stopping(vali_metrics['loss']['whole_loss'], self.model, path)
            elif self.saved_metric == 'mse':
                early_stopping(vali_metrics['mse']['whole_loss'], self.model, path)
            else:
                raise ValueError(f'Saved metric {self.saved_metric} not supported!')

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args.unify_lr)
            
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def adaptive_train(self, setting, save_root='checkpoints'):
        self.saved_metric = 'mse'
        
        print('\n################# Adaptive Train #################')

        if self.args.load_unify == 1:
            unify_root = save_root.replace("adaptive", "unify")
            self.model.load_state_dict(torch.load(os.path.join(unify_root, setting, 'checkpoint.pth')))

        self.model.module.set_state('adaptive')
        
        train_data_dict, train_loader_dict = self._get_data(flag='train')
        vali_data_dict, vali_loader_dict = self._get_data(flag='val')
        test_data_dict, test_loader_dict = self._get_data(flag='test')

        train_loader_dict = {self.args.adaptive_dataset: train_loader_dict[self.args.adaptive_dataset]}
        vali_loader_dict = {self.args.adaptive_dataset: vali_loader_dict[self.args.adaptive_dataset]}
        test_loader_dict = {self.args.adaptive_dataset: test_loader_dict[self.args.adaptive_dataset]}



        path = os.path.join(save_root, self.args.adaptive_dataset, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer(lr=self.args.adaptive_lr)
        criterion = self._select_criterion()
        mask_schedule = self._select_mask_scheduler()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            train_gt_id_list, train_pred_id_list = [], []

            train_loader = train_loader_dict[self.args.adaptive_dataset]
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.device)

                # batch_x, _ = self._normalize_input((batch_x, None), mode='train')
                # batch_y, _ = self._normalize_input((batch_y, None), mode='train')
                
                # print('break 1')
                
                ts_ids, labels, mask_prob, gt_tokens = \
                    self.prepare_inputs_and_labels(batch_x, batch_y, mask_schedule, is_train=True, dataname=self.args.adaptive_dataset)
                
                # print('ts_ids: ', ts_ids.shape)
                # print('labels: ', labels.shape)
                # print('soft_labels: ', soft_labels.shape)
                # print('mask_prob: ', mask_prob.shape)
                # print('gt_tokens: ', gt_tokens.shape)
                # exit(0)
                
                train_gt_id_list.append(gt_tokens)
                # print('break 2')
                # exit(0)
                
                inputs = {
                    'text_ids': None,
                    'ts_ids': ts_ids.to(self.device)
                }
                # exit(0)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                pred_ids = torch.argmax(outputs, dim=-1)
                train_pred_id_list.append(pred_ids[:, -self.args.pred_len // self.args.wave_length:])
                    
                outputs = outputs.view(-1, outputs.size(-1))
                # labels = labels.view(-1)
                target_mask = (labels.flatten() == -100)

                labels = labels.view(-1)

                loss = criterion(outputs, labels)
                train_loss.append(loss.item())
                
                # print(loss.item())
                # exit(0)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * len(train_loader) - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                    # report the accuracy of the train set
            train_gt_id_list = torch.cat(train_gt_id_list, dim=0)
            train_pred_id_list = torch.cat(train_pred_id_list, dim=0)

            train_metrics = token_metric(train_pred_id_list, train_gt_id_list)
            
            print("\n\n### Epoch: {} cost time: {} ###".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            vali_metrics = self.vali(vali_loader_dict, criterion, mask_schedule)
            test_metrics = self.vali(test_loader_dict, criterion, mask_schedule)

            print("Epoch: {0}, Steps: {1}, Train Loss: {2:.7f}".format(epoch + 1, i, train_loss))
            for dataname in vali_loader_dict.keys():
                print("{0} | Vali Loss: {1:.7f} Test Loss: {2:.7f} | Vali MSE: {3:.7f} Vali MAE: {4:.7f} Vali ACC: {5:.7f} Test MSE: {6:.7f} Test MAE: {7:.7f} Test ACC: {8:.7f}".format(dataname, vali_metrics['loss'][dataname], test_metrics['loss'][dataname], vali_metrics['mse'][dataname], vali_metrics['mae'][dataname], vali_metrics['acc'][dataname], test_metrics['mse'][dataname], test_metrics['mae'][dataname], test_metrics['acc'][dataname]))
            
            if self.saved_metric == 'entropy':
                early_stopping(vali_metrics['loss']['whole_loss'], self.model, path)
            elif self.saved_metric == 'mse':
                early_stopping(vali_metrics['mse']['avg_mse'], self.model, path)
            else:
                raise ValueError(f'Saved metric {self.saved_metric} not supported!')

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args.adaptive_lr)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test_func(self, test_loader, folder_path, object='test', dataname=None):
        self.model.eval()
        
        preds = []
        trues = []
        inputx = []

        print("successfully enter test_func...")
        
        output_tokens_list, gt_tokens_list = [], []
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.device)

                ori_batch_x = batch_x.clone()
                ori_batch_y = batch_y.clone()
                
                B = batch_x.shape[0]

                # batch_x, stat_prediction_y = self._normalize_input((ori_batch_x, ori_batch_y), mode='infer', train_loader=train_loader)
                # batch_y, _ = self._normalize_input((ori_batch_y, None), mode='train')
                
                ts_ids, gt_tokens, out_token_shape = \
                    self.prepare_inputs_and_labels(batch_x, batch_y, mask_schedule=None, is_train=False, dataname=dataname)
                    
                # print('ts_ids: ', ts_ids.shape)
                    
                inputs = {
                    'text_ids': None,
                    'ts_ids': ts_ids.to(self.device)
                }
                
                kwargs = {
                    'time_step': self.args.infer_step,
                    'out_token_num': out_token_shape[1]
                }

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        output_tokens_ls = self.model(inputs, mode='gen_ts', **kwargs)
                else:
                    output_tokens_ls = self.model(inputs, mode='gen_ts', **kwargs)

                    
                gt_tokens_list.append(gt_tokens)
                output_tokens_list.append(output_tokens_ls[0])

                # output_tokens = torch.argmax(logits, dim=-1)
                
                # output_tokens = output_tokens[:, -out_token_shape[1]:]

                outputs_list = []
                for output_tokens in output_tokens_ls:
                    outputs = self.decode_ts(output_tokens, B=B, look_back=ori_batch_x, dataname=dataname)
                    outputs_list.append(outputs)

                outputs = torch.stack(outputs_list, dim=0)
                outputs = outputs.mean(dim=0)
                
                outputs = outputs.detach().cpu().numpy()
                ori_batch_x = ori_batch_x.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = ori_batch_y.detach().cpu().numpy()  # batch_y.detach().cpu().numpy()  # .squeeze()
                
                # print(pred.shape)

                preds.append(pred)
                trues.append(true)
                inputx.append(ori_batch_x)
                # if i % 20 == 0 and object=='test':
                #     input = ori_batch_x
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))
                    
        output_tokens_list = torch.cat(output_tokens_list, dim=0)
        gt_tokens_list = torch.cat(gt_tokens_list, dim=0)
                    
        return preds, trues, inputx, output_tokens_list, gt_tokens_list

    def test(self, setting, test=0, save_root='checkpoints'):
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        train_data_dict, train_loader_dict = self._get_data(flag='train')
        vali_data_dict, vali_loader_dict = self._get_data(flag='val')
        test_data_dict, test_loader_dict = self._get_data(flag='test')

        if self.args.is_unify == 0: # adaptive
            test_loader_dict = {self.args.adaptive_dataset: test_loader_dict[self.args.adaptive_dataset]}
        
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(save_root, setting, 'checkpoint.pth')))
            
        # print(self.model)
        # exit(0)

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        # self.statistics_pred.eval()
        
        # plot train distrubution
        # preds, trues, inputx, output_tokens, gt_tokens = \
        #     self.test_func(train_loader, folder_path, object='train')
            
        # plot_token_distribution_with_stratify(gt_tokens, output_tokens, \
        #     save_dir=os.path.join(save_root, setting), max_token_num=self.args.elected_n_embed, dataset='train')
        
        # plot test distrubution

        
        # output_tokens, gt_tokens = torch.zeros(10), torch.ones(10)
        from utils.tools import draw_similarity, draw_confusion_matrix
        codebook = self.vq_model.module.get_codebook().detach().cpu().numpy()
        draw_similarity(codebook, label_name=[str(i) for i in range(codebook.shape[1])], pdf_save_path=os.path.join(save_root, setting, 'codebook_similarity.pdf'))
        # exit(0)

        # plot train distrubution
        # preds, trues, inputx, output_tokens, gt_tokens = \
        #     self.test_func(train_loader, folder_path, object='test')
        
        # plot_token_distribution_with_stratify(gt_tokens, output_tokens, \
        #     save_dir=os.path.join(save_root, setting), max_token_num=self.args.elected_n_embed, dataset='train')
        
        # plot valid distrubution
        # preds, trues, inputx, output_tokens, gt_tokens = \
        #     self.test_func(valid_loader, folder_path, object='test')
        
        # plot_token_distribution_with_stratify(gt_tokens, output_tokens, \
        #     save_dir=os.path.join(save_root, setting), max_token_num=self.args.elected_n_embed, dataset='vali')

        # plot test distrubution
        for i, (dataname, test_loader) in enumerate(test_loader_dict.items()):
            print(f"\nTesting {dataname}...")
            preds, trues, inputx, output_tokens, gt_tokens = \
                self.test_func(test_loader, folder_path, object='test', dataname=dataname)
            
            token_metric_dict = token_metric(output_tokens, gt_tokens)
            
            print(f"Token Metric: ")
            pprint(token_metric_dict)

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            inputx = np.concatenate(inputx, axis=0)
            
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
            print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            # print('rmse:{}, mape:{}, mspe:{}, corr{}'.format(rmse, mape, mspe, corr))

        return mse, mae

        # label_name = [str(i) for i in range(len(list(set(gt_tokens.flatten().detach().cpu().numpy()))))]
        
        # print(output_tokens.shape, gt_tokens.shape)

        
        # gt_tokens_np = gt_tokens.flatten().detach().cpu().numpy()
        # output_tokens_np = output_tokens.flatten().detach().cpu().numpy()
        # draw_confusion_matrix(gt_tokens_np, output_tokens_np, label_name=label_name, \
        #     pdf_save_path=os.path.join(save_root, setting, 'confuse_matrix.pdf'))
        # exit(0)

        # plot_token_distribution_with_stratify(gt_tokens, output_tokens, \
        #     save_dir=os.path.join(save_root, setting), max_token_num=self.args.elected_n_embed, dataset='test')
        

            
        # plot_token_distribution_with_stratify(gt_tokens, output_tokens, \
        #     save_dir=os.path.join(save_root, setting), max_token_num=self.args.elected_n_embed, dataset='train')  
        
        # if self.args.test_flop:
        #     input_x = torch.rand_like(batch_x)
        #     test_params_flop(self.model, (input_x,))
        #     exit()
        print('########')
            


        # result save
        if self.args.do_predict:
            folder_path = './Prediction/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            np.save(folder_path + 'real_prediction.npy', preds[0])
            np.save(folder_path + 'origin_series.npy',inputx[0])
            np.save(folder_path + 'ground_truth.npy',trues[0])

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        
        # print(preds.shape, trues.shape, inputx.shape)
        
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        print('rmse:{}, mape:{}, mspe:{}, corr{}'.format(rmse, mape, mspe, corr))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        origin_inputs = []
        ground_truth = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                        
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)
                origin_input = batch_x.squeeze().detach().cpu().numpy()
                origin_inputs.append(origin_input)
                truth = batch_y.squeeze().detach().cpu().numpy()
                ground_truth.append(truth)
                
        origin_inputs = np.array(origin_inputs)
        print(origin_inputs.shape)
        origin_inputs = origin_inputs.reshape(-1,origin_inputs.shape[-2],origin_inputs.shape[-1])
        print(origin_inputs.shape)
        ground_truth = np.array(ground_truth)
        print(ground_truth.shape)
        preds = np.array(preds)
        ground_truth = ground_truth.reshape(-1,ground_truth.shape[-2],ground_truth.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        print(preds.shape)
        # result save
        folder_path = './prediction/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        np.save(folder_path + 'real_prediction.npy', preds)
        np.save(folder_path + 'origin_series.npy',origin_inputs)
        np.save(folder_path + 'ground_truth.npy',ground_truth)
        return

        np.save(folder_path + 'ground_truth.npy',ground_truth)
        return

