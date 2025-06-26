import torch.nn as nn
class Criterion:
    def __init__(self, model, latent_loss_weight=0.1, trend_loss_weight=0.1):
        self.model = model
        self.latent_loss_weight = latent_loss_weight
        self.lamb = trend_loss_weight
        self.mse = nn.MSELoss()

    def compute(self, batch_x, batch_y, details=False, dataset_id=None):
        return_dict = self.model(batch_x, batch_y, dataset_id)

        dec_L_trend = return_dict['dec_L_trend']
        dec_P_trend = return_dict['dec_P_trend']
        dec_P = return_dict['dec_P']

        gt_L_trend = return_dict['gt_L_trend']
        gt_P_trend = return_dict['gt_P_trend']
        gt_P = return_dict['gt_P']
        
        recon_loss_L_trend = self.mse(dec_L_trend, gt_L_trend)
        recon_loss_P_trend = self.mse(dec_P_trend, gt_P_trend)
        recon_loss_P = self.mse(dec_P, gt_P)

        diff_L = return_dict['diff_L']
        latent_loss = diff_L.mean()

        loss = recon_loss_L_trend + self.lamb * recon_loss_P_trend + recon_loss_P \
            + self.latent_loss_weight * latent_loss
        
        if details:
            return {
                'gt_L_trend': gt_L_trend,  
                'gt_P_trend': gt_P_trend,
                'gt_P': gt_P,

                'recon_L_trend': dec_L_trend,
                'recon_P_trend': dec_P_trend,
                'recon_P': dec_P,

                'loss': loss,
                'recon_loss_L_trend': recon_loss_L_trend,
                'recon_loss_P_trend': recon_loss_P_trend,
                'recon_loss_P': recon_loss_P,
                'latent_loss': latent_loss
            }
        else:
            return loss
        
            