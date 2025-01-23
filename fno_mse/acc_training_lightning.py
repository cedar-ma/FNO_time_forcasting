import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from tnetwork_lightning import FNO3D
from tdataloader_utils import NumpyDataset, get_dataloader
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import wandb
from lightning.pytorch.loggers import WandbLogger
import glob
from argparse import ArgumentParser
from lightning.pytorch.accelerators import find_usable_cuda_devices
torch.cuda.empty_cache()

def main(hparams):
    wandb.init(project="PE-FNO3D", name='PEFNO-5')
    logger = WandbLogger()
    run_id = logger.experiment.id
    # Preprocessing args
    net_name = "tgl2cis1"
    torch.set_float32_matmul_precision('medium')
    epochs = 39
    val_interval = 5
    lr = 1e-4


    # Load the data
    seed = 189031465

    n_samples = 201
    split = [0.6, 0.2, 0.2]
    image_ids = np.random.randint(low=0, high=201, size=(n_samples,))
    # train_dataloader, val_dataloader, test_dataloader = load_data()
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(image_ids, 
                                                                       data_path="/scratch/08780/cedar996/lbfoam/fno/lbfoam_sims",
                                                                       seed=seed,
                                                                       split=split,
                                                                       num_workers=2,
                                                                       pin_memory=True)
    # TODO: Try loading a model first
    try:
        model_dir = f'lightning_logs/{net_name}/checkpoints'
        model_loc = glob.glob(f'{model_dir}/*val*.ckpt')[0]
        print(f'Loading {model_loc}')
        model = FNO3D.load_from_checkpoint(model_loc)

    except IndexError:
        # Instantiate the model
        print('Instantiating a new model...')
        model = FNO3D(net_name=net_name,
                      in_channels=1,
                      out_channels=1,
                      modes1=12,
                      modes2=12,
                      modes3=12,
                      width=30,
                      beta_1=1,
                      beta_2=0,
                      num_epochs_pretraining=epochs,
                      lr=lr)

    # Add some checkpointing callbacks
    cbs = [ModelCheckpoint(monitor="loss", filename="{epoch:02d}-{loss:.2f}",
                           dirpath=logger.log_dir,
                           save_top_k=1,
                           mode="min"),
           ModelCheckpoint(monitor="val_loss", filename="{epoch:02d}-{val_loss:.2f}",
                           dirpath=logger.log_dir,
                           save_top_k=1,
                           mode="min"),
           EarlyStopping(monitor="val_loss", check_finite=False, patience=9999)]

    trainer = pl.Trainer(
       # strategy="ddp", 
       # num_nodes=1,
        callbacks=cbs,  # Add the checkpoint callback
        max_epochs=epochs,
        check_val_every_n_epoch=val_interval,
        log_every_n_steps=n_samples * split[0],
        logger=logger,
        accelerator=hparams.accelerator,
        devices=hparams.devices,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    wandb.finish()

    predictions = trainer.predict(model, dataloaders=test_dataloader)

    i = 0

    for batch in predictions:
        # print(batch['div'], batch['div_hat'])
        for j in range(20):
            plt.close('all')
            plt.subplot(1, 2, 1)
            h = plt.imshow(batch['j'][-1].cpu().numpy()[0, :, :, j])
            plt.colorbar(orientation='horizontal')
            plt.title('Ground Truth')
            cmin, cmax = h.get_clim()
            plt.subplot(1, 2, 2)
            plt.imshow(batch['jhat'][-1].cpu().numpy()[0, :, :, j], vmin=cmin, vmax=cmax)
            plt.colorbar(orientation='horizontal')
            plt.title('Prediction')
#        plt.show()
            plt.savefig(f'./fno_figures/epoch_gpu/prediction_s{i}_t{10+j}.png',dpi=300)
        i = i + 1

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)
