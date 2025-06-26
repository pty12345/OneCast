import os
import time
import torch
import pickle
import torch.nn as nn

import numpy as np
import random
from tqdm import tqdm
from loss import Criterion
from torch.optim.lr_scheduler import LambdaLR

from utils.tools import plot_token_distribution, plot_token_distribution_with_stratify
from utils.tools import plot_and_save_reconstruction, plot_PCA, statistic_freqs
from utils.tools import plot_results, plot_and_save_reconstruction_double

class Trainer():
    def __init__(self, args, model, train_data_loaders, vali_data_loaders, test_data_loaders, verbose=False):
        self.args = args
        self.verbose = verbose
        self.device = args.device
        self.print_process(self.device)
        self.model = model.to(torch.device(self.device))

        # datasets 
        self.train_loader_dict = train_data_loaders
        self.vali_loader_dict = vali_data_loaders
        self.test_loader_dict = test_data_loaders

        # cross dataset, divided based on dataset attribute and experiment
        self.cross_dataset = args.cross_dataset
        
        # learning parameters
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps
        self.weight_decay = args.weight_decay
        self.model_name = self.model.get_name()
        self.print_process(self.model_name)

        # loss function
        self.cr = Criterion(self.model, latent_loss_weight=args.latent_loss_weight, trend_loss_weight=args.trend_loss_weight)

        # training parameters
        self.step = 0
        self.num_epoch = args.num_epoch
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        
        if args.load_path is not None:
            self.load_path = args.load_path  
        else:
            self.load_path = args.save_path

        # recording strategy
        self.best_metric = 1e9
        self.metric = 'reconst_mse'

    def train(self):
        self.print_process('\n\n######### Start Training #########')
        train_one_epoch_func = self._train_one_epoch
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)

        # training
        for epoch in range(self.num_epoch):
            loss_epoch, time_cost = train_one_epoch_func(epoch)
            self.print_process(
                'Basic Model train epoch:{0}, loss:{1:.6f}, training_time:{2:.6f}'.format(epoch + 1, loss_epoch, time_cost))
            
        self.print_process(self.best_metric)
        return self.best_metric
      
    def _eval(self, epoch):
        metric_dict = {}
        for key in ['train', 'valid', 'test']:
            if key == 'train': data_loaders = self.train_loader_dict
            elif key == 'valid': data_loaders = self.vali_loader_dict
            elif key == 'test': data_loaders = self.test_loader_dict
            
            _metric = self.eval_model_vqvae(data_loaders)
            metric_dict[key] = _metric
            
            for dataname, metric in _metric.items():
                print(f'{key} on {dataname}: ', end='')
                self.print_process(metric)

            print('\n')
        
        metric = metric_dict['valid']
        
        # record the best model by validation metric
        total_metric = sum(metric[self.metric] for metric in metric_dict['valid'].values())
        if total_metric < self.best_metric:
            self.model.eval()
            torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
            
            if not self.args.eval_per_epoch:
                print('best model saved at step{0}'.format(self.step))
            else:
                print('best model saved at epoch{0}'.format(epoch))
            
            self.best_metric = total_metric

            print(f'best metric update to: {self.best_metric}')
            
        print('\n\n')

        self.model.train()
        
    def _get_all_ids(self, data_loader_dict):
        # get test token distribution and calculate mse
        ids = []
        with torch.no_grad():
            for dataset_id, (dataname, data_loader) in enumerate(data_loader_dict.items()):
                for idx, (batch_x, batch_y, _, _) in enumerate(data_loader):
                    batch_x = batch_x.float().to(self.args.device)
                    batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.args.device)
                    return_dict = self.model(batch_x, batch_y, dataset_id=dataset_id)
                    
                ids.append(return_dict['id_L'].flatten())
        
        ids = torch.cat(ids).cpu().numpy()
        return ids

    def _train_one_epoch(self, epoch):
        t0 = time.perf_counter()
        self.model.train()

        data_iters = {}
        
        dataset_list, r = [], []
        for idx, dataname in enumerate(self.train_loader_dict.keys()):
            train_loader = self.train_loader_dict[dataname]

            length = len(train_loader)
            r = r + [idx] * length

            data_iters.update({dataname:iter(train_loader)})
            dataset_list.append(dataname)

        random.shuffle(r)

        loss_sum = 0

        tqdm_iter = tqdm(r, desc=f"Epoch {epoch+1}", ncols=120, position=0, leave=True)

        count = [0] * len(dataset_list)

        for i in tqdm_iter:
            self.optimizer.zero_grad()

            count[i] += 1

            try:
                batch_x, batch_y, batch_x_mark, batch_y_mark = next(data_iters[dataset_list[i]])
            except StopIteration:
                print(f"Dataset {dataset_list[i]} has no more data")
                print(f"Count: {count}")
                exit(0)

            batch_x = batch_x.float().to(self.args.device)
            batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.args.device)

            loss = self.cr.compute(batch_x, batch_y, dataset_id=i)

            loss_sum += loss.item()

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            self.step += 1
            if self.step % self.lr_decay_steps == 0:
                self.scheduler.step()
                print("Update learning rate to {}".format(self.optimizer.param_groups[0]['lr']))
            
            if (self.step % self.eval_per_steps == 0) and (self.args.eval_per_epoch == False):
                self._eval(epoch)
                
        if self.args.eval_per_epoch:
            self._eval(epoch)
            
            # plot the distribution of train and test tokens
            # train_ids = self._get_all_ids(self.train_loader)
            # test_ids = self._get_all_ids(self.test_loader)
            
            # plot_path = os.path.join(self.load_path, 'token_distribution_epoch{}'.format(epoch))
            
            # plot_token_distribution_with_stratify(train_ids, test_ids, \
            #     save_dir=plot_path, max_token_num=self.args.n_embed, freq=True)

        return loss_sum, time.perf_counter() - t0

    def eval_model_vqvae(self, data_loaders):
        self.model.eval()
        metric = {'reconst_mse_L_trend': 0, 'reconst_mse_P_trend': 0, \
                   'reconst_mse_P_res': 0, 'reconst_mse_P': 0, 'latent_mse': 0}

        metrics = {dataname:metric.copy() for dataname in data_loaders.keys()}

        with torch.no_grad():
            for dataset_id, (dataname, data_loader) in enumerate(data_loaders.items()):
                recon_L_trend_list, recon_P_trend_list, recon_P_list = [], [], []
                gt_L_trend_list, gt_P_trend_list, gt_P_list = [], [], []
                for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                    batch_x = batch_x.float().to(self.args.device)
                    batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.args.device)

                    # compute the loss
                    out_dict = self.cr.compute(batch_x, batch_y, details=True, dataset_id=dataset_id)

                    recon_L_trend_list.append(out_dict['recon_L_trend'].detach().cpu())
                    recon_P_trend_list.append(out_dict['recon_P_trend'].detach().cpu())
                    recon_P_list.append(out_dict['recon_P'].detach().cpu())

                    gt_L_trend_list.append(out_dict['gt_L_trend'].detach().cpu())
                    gt_P_trend_list.append(out_dict['gt_P_trend'].detach().cpu())
                    gt_P_list.append(batch_y.detach().cpu())

                    metrics[dataname]['latent_mse'] += out_dict['latent_loss'].detach().cpu()

                # calculate the mse of the reconstruction
                pred_L_trend = torch.cat(recon_L_trend_list, dim=0)
                pred_P_trend = torch.cat(recon_P_trend_list, dim=0)
                pred_P = torch.cat(recon_P_list, dim=0)

                gt_L_trend = torch.cat(gt_L_trend_list, dim=0)
                gt_P_trend = torch.cat(gt_P_trend_list, dim=0)
                gt_P = torch.cat(gt_P_list, dim=0)

                mse = nn.MSELoss()

                metrics[dataname]['reconst_mse_L_trend'] = mse(pred_L_trend, gt_L_trend)
                metrics[dataname]['reconst_mse_P_trend'] = mse(pred_P_trend, gt_P_trend)
                metrics[dataname]['reconst_mse_P'] = mse(pred_P, gt_P)

                metrics[dataname]['latent_mse'] /= idx
                metrics[dataname]['reconst_mse'] = metrics[dataname]['reconst_mse_L_trend'] + metrics[dataname]['reconst_mse_P']

        return metrics
    
    def print_process(self, *x):
        if self.verbose:
            print(*x)

    def test(self):
        self.print_process('\n######### Start Testing #########')

        # print(self.load_path)
        # exit(0)
        
        state_dict = torch.load(os.path.join(self.load_path, 'model.pkl'), map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # plot_path = os.path.join(self.load_path, 'reconstruction_random_replace{}'.format(self.args.test_random_replace))
        # plot_and_save_reconstruction_double(self.model, self.test_loader, plot_path, self.args.pred_len)
        # print("Images have been saved.")
        
        # plot the low-dimensional representation of the code book
        
        # get reconst mse


        # if self.args.test_full_window:
        #     with torch.no_grad():
        #         for idx, (batch_x, batch_y, _, _) in enumerate(self.test_loader):
        #             batch_x = batch_x.float().to(self.args.device)
        #             batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.args.device)

        #             reconst, _, _  = self.model(batch_x, batch_y)
        #             model_load_path = self.args.save_path
        #             plot_results(seqs_x[0], reconst[0], model_load_path)
        #             break
     
        # get test token distribution and calculate mse
        mse = nn.MSELoss()
        total_recon_loss_L = 0.0
        total_recon_loss_P = 0.0
        total_batches = 0

        test_ids = []

        with torch.no_grad():
            for dataset_id, (dataname, test_loader) in enumerate(self.test_loader_dict.items()):
                reconst_L_trend_list, reconst_P_trend_list, reconst_P_list = [], [], []
                gt_L_trend_list, gt_P_trend_list, gt_P_list = [], [], []
                for idx, (batch_x, batch_y, _, _) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.args.device)
                    batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.args.device)
                    return_dict = self.model(batch_x, batch_y, dataset_id=dataset_id)

                    id_L = return_dict['id_L']

                    test_ids.append(id_L.flatten())
                    
                    reconst_L_trend_list.append(return_dict['dec_L_trend'].detach().cpu())
                    reconst_P_trend_list.append(return_dict['dec_P_trend'].detach().cpu())
                    reconst_P_list.append(return_dict['dec_P'].detach().cpu())

                    gt_L_trend_list.append(return_dict['gt_L_trend'].detach().cpu())
                    gt_P_trend_list.append(return_dict['gt_P_trend'].detach().cpu())
                    gt_P_list.append(batch_y.detach().cpu())

                # print('out_x: ', out_x.shape)
                # exit(0)

                pred_L_trend = torch.cat(reconst_L_trend_list, dim=0)
                pred_P_trend = torch.cat(reconst_P_trend_list, dim=0)
                pred_P = torch.cat(reconst_P_list, dim=0)

                gt_L_trend = torch.cat(gt_L_trend_list, dim=0)
                gt_P_trend = torch.cat(gt_P_trend_list, dim=0)
                gt_P = torch.cat(gt_P_list, dim=0)

            # print('pred_L: ', pred_L.shape)
            # print('pred_P: ', pred_P.shape)
            # print('gt_x: ', gt_x.shape)
            # print('gt_y: ', gt_y.shape)
            # exit(0)

                mse = nn.MSELoss()
                
                avg_recon_loss_L_trend = mse(pred_L_trend, gt_L_trend)

                avg_recon_loss_P_trend = mse(pred_P_trend, gt_P_trend)
                avg_recon_loss_P = mse(pred_P, gt_P)
                
                print(dataname, ":")
                print('reconstruct loss L_trend(mse) on test dataset: {:.6f}\n'.format(avg_recon_loss_L_trend))
                print('reconstruct loss P_trend(mse) on test dataset: {:.6f}\n'.format(avg_recon_loss_P_trend))
                print('reconstruct loss P(mse) on test dataset: {:.6f}\n'.format(avg_recon_loss_P))

        # exit(0)
                        
        # plot the distribution of train and test tokens

        
        plot_path = os.path.join(self.load_path, 'token_distribution')
        
        test_ids = torch.cat(test_ids).cpu().numpy()
        
        # print the statistics of the token distribution
        # 
        
        # codebook_plot_path = os.path.join(self.load_path, 'codebook_with_used_freqs.png')
        # codebook = self.model.get_codebook_weight()
        

        # exit(0)
        train_ids = self._get_all_ids(self.train_loader_dict)
        # plot_PCA(train_ids, codebook, codebook_plot_path, max_token_num=self.args.n_embed)
        
        statistic_freqs(train_ids.flatten())
        # test_ids = self._get_all_ids(self.test_loader)
        plot_token_distribution_with_stratify(train_ids, test_ids, \
            save_dir=plot_path, max_token_num=self.args.n_embed)
        
        # count the frequence of train tokens
        freq = np.bincount(train_ids)
        fixed_freq = np.where(freq > 0, freq, 1e7)
        
        print(len(freq))
        
        n_classes = len(set(train_ids))
        weight = len(train_ids) / (n_classes * fixed_freq)
        
        mask = freq > 0
        train_tokens = train_ids.flatten()
        train_uni_elements,  train_cnts_elements = \
            np.unique(train_tokens, return_counts=True)
            
        weight_dict = {
            'weight': weight,
            'mask': mask,
            'train_uni_elements': train_uni_elements,
            'train_cnts_elements': train_cnts_elements,
            'total_nums': len(train_ids)
        }
        
        print("Successfully save weight.pkl")
        
        save_w_path = os.path.join(self.load_path, 'weight.pkl')
        pickle.dump(weight_dict, open(save_w_path, 'wb'))
        
        exit(0)
        
        # Just calculate the minimun weight from existing tokens
        
        # print((freq > 0).shape)
        
        # real_min_weight = np.min(weight, where=(freq > 0), initial=np.inf)
        # max_weight = real_min_weight * 20
        # weight = np.clip(weight, a_min=None, a_max=max_weight)
        
        # print("#### Weight Statistics: ####")
        # print(weight.shape, max(weight), min(weight)) # min:0.11 max: 647
        

        
        print("#### Token Distribution Analysis ####")
        print("Training Set: Used token is {}, Total token is {}".format(len(set(train_ids)), self.args.n_embed))
        print("Test Set: Used token is {}, Total token is {}".format(len(set(test_ids)), self.args.n_embed))


        avg_recon_loss = total_recon_loss / total_batches
        
        print('reconstruct loss(mse) on test dataset: {:.6f}\n'.format(avg_recon_loss))

        plot_path = os.path.join(self.load_path, 'reconstruction')
        os.makedirs(plot_path, exist_ok=True)
        
        plot_and_save_reconstruction(self.model, self.test_loader, plot_path)
        print("Images have been saved.")
            