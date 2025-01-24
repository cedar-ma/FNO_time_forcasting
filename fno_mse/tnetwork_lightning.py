import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tFNO3D import tFNO3DModel
from physics import divergence

class FNO3D(pl.LightningModule):
    def __init__(self,
                 net_name='test3',
                 model=tFNO3DModel,
                 in_channels=10,
                 out_channels=1,
                 modes1=8,
                 modes2=8,
                 modes3=8,
                 width=20,
                 lr=1e-3,
                 beta_1=1, 
                 beta_2=0,
               ):
        
        super(FNO3D, self).__init__()

        # if hparams:
        #     self.save_hyperparameters(hparams)
        
        self.net_name = net_name
        self.lr = lr
        self.PE_lr = lr / 10
        self.beta_1, self.beta_2 = beta_1, beta_2

        # self.model_pretraining = GetFNO3DModel(in_channels=in_channels,
        #                                 out_channels=out_channels,
        #                                 modes1=modes1,
        #                                 modes2=modes2,
        #                                 modes3=modes3,
        #                                 width=width)
        #                                 # GetFNO3DModel(in_channels=in_channels,
        #                                 #        out_channels=out_channels,
        #                                 #        modes1=modes1,
        #                                 #        modes2=modes2,
        #                                 #        modes3=modes3,
        #                                 #        width=width)
        
        self.model = tFNO3DModel(in_channels=in_channels,
                                        out_channels=out_channels,
                                        modes1=modes1,
                                        modes2=modes2,
                                        modes3=modes3,
                                        width=width)

        # self.is_pretraining = False
        # self.num_epochs_pretraining = num_epochs_pretraining
    
    def forward(self, x):
        # if self.is_pretraining:
        #     return self.model_pretraining(x)
        # else:
        #     return self.model_PE(x)
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        sigma, j = batch
        # j = torch.cat([jx, jy, jz], dim=1)
        jhat = self(sigma)
        jhat = torch.squeeze(jhat,dim=-1)
        # jhat = j.reshape((1, 1, 10, 1280, 550))

        # if self.is_pretraining:
        #     kernels = self.model_pretraining.kernels
        # else:
        #     kernels = self.model_PE.kernels
        # Compute divergence and log it
        # div = divergence(j, kernels)
        # divhat = divergence(jhat, kernels)
        # total_div = torch.sum(divhat)
        # self.log("train_div", total_div, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)

        loss = 0
        for j, jhat in zip([j], [jhat]):
            j_loss = F.mse_loss(j.view((1, -1)), jhat.view((1, -1)))
           # j_loss = F.binary_cross_entropy(jhat.view((1, -1)), j.view((1, -1)))
            # div_loss = F.mse_loss(div.view((1, -1)), divhat.view((1, -1)))
            #self.log(f"loss_{j}", j_loss, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)
            loss += self.beta_1 * j_loss #+ self.beta_2 * div_loss
            
        self.log("loss", loss, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        sigma, j = batch
        # j = torch.cat([jx, jy, jz], dim=1)
        jhat = self(sigma)
        jhat = torch.squeeze(jhat,dim=-1)
        # jhat = j.reshape((1, 1, 10, 1280, 550))

        # if self.is_pretraining:
        #     kernels = self.model_pretraining.kernels
        # else:
        #     kernels = self.model_PE.kernels
        # Compute divergence and log it
        # div = divergence(j, kernels)
        # divhat = divergence(jhat, kernels)
        # total_div = torch.sum(divhat)
        # self.log("val_div", total_div, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)

        val_loss = 0
        for j, jhat in zip([j], [jhat]):
           # j_loss = F.binary_cross_entropy(jhat.view((1, -1)), j.view((1, -1)))
            j_loss = F.mse_loss(j.view((1, -1)), jhat.view((1, -1)))
            # div_loss = F.mse_loss(div.view((1, -1)), divhat.view((1, -1)))
            #self.log(f"val_loss_{j}", j_loss, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)
            val_loss += self.beta_1 * j_loss #+ self.beta_2 * div_loss
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        sigma, j = batch
        # j = torch.cat([jx, jy, jz], dim=1)
        #j = torch.squeeze(j,dim=-1)
        jhat = self(sigma)
        jhat = torch.squeeze(jhat,dim=-1)
        # jhat = j.reshape((1, 1, 10, 1280, 550))

        # if self.is_pretraining:
        #     kernels = self.model_pretraining.kernels
        # else:
        #     kernels = self.model_PE.kernels
        # Compute divergence and log it
        # div = divergence(j, kernels)
        # divhat = divergence(jhat, kernels)
        # total_div = torch.sum(divhat)
        # self.log("test_div", total_div, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)

        test_loss = 0
        component_loss = []
        for j, jhat in zip([j], [jhat]):
           # j_loss = F.binary_cross_entropy(jhat.view((1, -1)), j.view((1, -1)))
            j_loss = F.mse_loss(j.view((1, -1)), jhat.view((1, -1)))
            component_loss.append(j_loss)
            # div_loss = F.mse_loss(div.view((1, -1)), divhat.view((1, -1)))
            #self.log(f"val_loss_{j}", j_loss, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)
            test_loss += self.beta_1 * j_loss #+ self.beta_2 * div_loss
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)

        test_metrics = {
            'loss': test_loss,
            'component_loss': component_loss
        }

        return test_metrics
    
    def predict_step(self, batch, batch_idx):
        sigma, j = batch
        # j = torch.cat([jx, jy, jz], dim=1)
        jhat = self(sigma)
        jhat = torch.squeeze(jhat,dim=-1)
        # jhat = j.reshape((1, 1, 10, 1280, 550))

        # Add divergence
        # if self.is_pretraining:
        #     kernels = self.model_pretraining.kernels
        # else:
        #     kernels = self.model_PE.kernels
        # Compute divergence and log it
        # div = divergence(j, kernels)
        # divhat = divergence(jhat, kernels)
        # total_divhat = torch.sum(divhat)

        # True divergence
        # j = torch.cat([jx, jy, jz], dim=1)
        # div = divergence(j, kernels)
        # total_div = torch.sum(div)

        predictions = {
            'j': [j],
            'jhat': [jhat],
            # 'div': total_div,
            # 'div_hat': total_divhat
        }

        return predictions
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        # scheduler = CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer]#, [scheduler]
    
    def on_epoch_end(self):
        if self.current_epoch + 1 == self.num_epochs_pretraining:  # Switch after the last pretraining epoch
            print("Switching to fine-tuning phase.")
            self.is_pretraining = False  # Toggle to fine-tuning phase
            self.adjust_learning_rate()   # Adjust learning rate
            self.log("Switching to fine-tuning phase.", prog_bar=True)
    
    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.PE_lr
