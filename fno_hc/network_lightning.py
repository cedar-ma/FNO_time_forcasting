import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from FNO3D import tFNO3DModel

class FNO3D(pl.LightningModule):
    def __init__(self,
                 net_name='Blah',
                 model=tFNO3DModel,
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

        # if hparams:
        #     self.save_hyperparameters(hparams)
        self.model = tFNO3DModel(in_channels=in_channels,
                                 out_channels=out_channels,
                                 modes1=modes1,
                                 modes2=modes2,
                                 modes3=modes3,
                                 width=width)
        self.net_name = net_name
        self.lr = lr
        self.PE_lr = lr / 10
        self.beta_1, self.beta_2 = beta_1, beta_2

    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        sigma, j = batch
        jhat = self(sigma)
        # jhat = torch.squeeze(jhat, dim=-1)
        jhat = jhat.permute(0, 4, 1, 2, 3)
        #j = F.one_hot(j, num_classes=3).float()
        loss = 0
        for j, jhat in zip([j], [jhat]):
            j_loss = F.cross_entropy(jhat, j)
            loss += self.beta_1 * j_loss
            
        self.log("loss", loss, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        sigma, j = batch
        jhat = self(sigma)
        jhat = jhat.permute(0, 4, 1, 2, 3)
        #j = F.one_hot(j, num_classes=3).float()
        val_loss = 0
        for j, jhat in zip([j], [jhat]):
            j_loss = F.cross_entropy(jhat, j)
            val_loss += self.beta_1 * j_loss
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        sigma, j = batch
        jhat = self(sigma)
        jhat = jhat.permute(0, 4, 1, 2, 3)
        #j = F.one_hot(j, num_classes=3).float()
        test_loss = 0
        component_loss = []
        for j, jhat in zip([j], [jhat]):
            j_loss = F.cross_entropy(jhat, j)
            component_loss.append(j_loss)
            test_loss += self.beta_1 * j_loss
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)

        test_metrics = {
            'loss': test_loss,
            'component_loss': component_loss
        }

        return test_metrics
    
    def predict_step(self, batch, batch_idx):
        sigma, j = batch
        jhat = self(sigma)
        jhat = jhat.softmax(dim=-1)
        jhat = torch.argmax(jhat, dim=-1)#.view(j.shape)

        predictions = {
            'j': [j],
            'jhat': [jhat],
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
