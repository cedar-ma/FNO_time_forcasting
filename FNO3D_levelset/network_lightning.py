import lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from FNO3D import GetFNO3DModel

class FNO3D(pl.LightningModule):
    def __init__(self,
                 net_name='Blah',
                 model=GetFNO3DModel,
                 in_channels=10,
                 out_channels=3,
                 modes1=8,
                 modes2=8,
                 modes3=8,
                 width=20,
                 lr=1e-3,
                 beta_1=1,
                 beta_2=0,
                 ):
        
        super(FNO3D, self).__init__()

        
        self.net_name = net_name
        self.lr = lr
        self.PE_lr = lr / 10
        self.beta_1, self.beta_2 = beta_1, beta_2

        
        self.model = GetFNO3DModel(in_channels=in_channels,
                                        out_channels=out_channels,
                                        modes1=modes1,
                                        modes2=modes2,
                                        modes3=modes3,
                                        width=width)
        
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        sigma, j, _, _ = batch
        jhat = self(sigma)
        jhat = torch.squeeze(jhat,dim=-1)

        loss = 0
        for j, jhat in zip([j], [jhat]):
            j_loss = F.mse_loss(j.view((1, -1)), jhat.view((1, -1)))
            loss += self.beta_1 * j_loss #+ self.beta_2 * div_loss
            
        self.log("loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)#rank_zero_only=True)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        sigma, j, _, _ = batch
        jhat = self(sigma)
        jhat = torch.squeeze(jhat,dim=-1)

        val_loss = 0
        for j, jhat in zip([j], [jhat]):
            j_loss = F.mse_loss(j.view((1, -1)), jhat.view((1, -1)))
            val_loss += self.beta_1 * j_loss #+ self.beta_2 * div_loss
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)#rank_zero_only=True)
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        sigma, j, _, _ = batch

        jhat = self(sigma)
        jhat = torch.squeeze(jhat,dim=-1)

        test_loss = 0
        component_loss = []
        for j, jhat in zip([j], [jhat]):
            j_loss = F.mse_loss(j.view((1, -1)), jhat.view((1, -1)))
            component_loss.append(j_loss)
            test_loss += self.beta_1 * j_loss# + self.beta_2 * div_loss
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)#rank_zero_only=True)

        test_metrics = {
            'loss': test_loss,
            'component_loss': component_loss
        }

        return test_metrics
    
    def predict_step(self, batch, batch_idx):
        sigma, j, means, stds = batch
        jhat = self(sigma)
        print('Normal prediction range: ',jhat.min(),jhat.max())
        print('Normal truth range: ',j.min(),j.max())
        
        jhat = torch.squeeze(jhat,dim=-1)
        jhat = self.z_score_back_transform(jhat, means, stds) 
        j = self.z_score_back_transform(j, means, stds)
        print('Back transform prediction range: ',jhat.min(),jhat.max())
        print('Back transform truth range: ',j.min(),j.max())
        binary_predictions = (jhat <= 0).float()
        j = (j <= 0).float()
        predictions = {
            'j': j,
            'jhat': binary_predictions,
        }
        return predictions

    def z_score_back_transform(self, normalized_data, means, stds):
        """
        Back-transform Z-score normalized data to its original scale.
        Args:
            normalized_data (torch.Tensor): Normalized data of shape [batch_size, height, width, seq_len].
            means (np.ndarray): Means of shape [height, width, 1].
            stds (np.ndarray): Standard deviations of shape [height, width, 1].
        Returns:
            original_data (torch.Tensor): Back-transformed data of shape [batch_size, height, width, seq_len].
        """
        # Convert means and stds to PyTorch tensors
        #means = torch.tensor(means, dtype=torch.float32)  # Shape: [height, width, 1]
        #stds = torch.tensor(stds, dtype=torch.float32)    # Shape: [height, width, 1]

        # Reshape means and stds to match the batch and channel dimensions
        means = means.unsqueeze(0)  # Shape: [1, height, width, 1]
        stds = stds.unsqueeze(0)    # Shape: [1, height, width, 1]

        # Back-transform
        original_data = (normalized_data * stds) + means
        return original_data

    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
        return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'val_loss',  # Metric to monitor
            "interval": "epoch",  # Check every epoch
            "frequency": 5
            }
         }


