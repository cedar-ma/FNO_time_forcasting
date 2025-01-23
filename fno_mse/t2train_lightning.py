import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from tnetwork_lightning import FNO3D
from tFNO3D import tFNO3DModel
from tdataloader_utils import NumpyDataset, get_dataloader
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import wandb
from lightning.pytorch.loggers import WandbLogger
import glob
from pathlib import Path
import json


if __name__ == "__main__":

    # Preprocessing args
    # net_name = "Blah_Blah"
    torch.set_float32_matmul_precision('medium')
    # epochs = 5
    # val_interval = 5
    # lr = 1e-4


    # Load the data
    seed = 189031465
    np.random.seed(seed)
    n_samples = 202
    split = [0.6, 0.2, 0.2]
    image_ids = np.random.randint(low=0, high=201, size=(n_samples,))
    # train_dataloader, val_dataloader, test_dataloader = load_data()
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(image_ids, 
                                                                       data_path="/scratch/08780/cedar996/lbfoam/fno/lbfoam_sims",
                                                                       seed=seed,
                                                                       split=split,
                                                                       num_workers=2,
                                                                       pin_memory=True)
    hparams = {
        'net_name': "mse_hang10",
        'learning_rate': 1e-3,
        'batch_size': 1,
        'epochs': 153,
        'val_interval': 5,
        'modes1': 12,
        'modes2': 12,
        'modes3': 6,
        'width': 20,
        'beta_1': 1,
        'beta_2': 1,
        # 'n_samples': n_samples,
        # 'n_train': len(train_ids),
        # 'n_val': len(val_ids),
        # 'n_test': len(test_ids),
        # 'train_ids': train_ids.tolist(),
        # 'val_ids': val_ids.tolist(),
        # 'test_ids': test_ids.tolist(),
        'model': "tFNO3DModel",
        'seed': 189031465,
        'patience': 13
    }
    wandb.init(project="FNO3D", name=hparams['net_name'],
               config=hparams, save_code=True, id=hparams['net_name'])
    logger = WandbLogger()
    run_id = logger.experiment.id
    # TODO: Try loading a model first
    try:
        model_dir = f"lightning_logs/{hparams['net_name']}/checkpoints"
        model_loc = glob.glob(f'{model_dir}/*val*.ckpt')[0]
        print(f'Loading {model_loc}')
        with open(f"lightning_logs/{hparams['net_name']}/hparam_config.json", 'r') as f:
            json_string = f.read()

        hparams = json.loads(json_string)

        hparams['seed'] = 189031465
        np.random.seed(hparams['seed'])
   #     hparams['epochs'] = 30

        model = FNO3D.load_from_checkpoint(model_loc,
                                           model=hparams['model'],
                                           in_channels=10,
                                           out_channels=1,
                                           modes1=hparams['modes1'],
                                           modes2=hparams['modes2'],
                                           modes3=hparams['modes3'],
                                           width=hparams['width'],
                                           beta_1=hparams['beta_1'],
                                           beta_2=hparams['beta_2'],
                                           lr=hparams['learning_rate'],
                                           )

    except IndexError:
        # Instantiate the model
        print('Instantiating a new model...')
        model = FNO3D(net_name=hparams['net_name'],
                      in_channels=10,
                      out_channels=1,
                      modes1=hparams['modes1'],
                      modes2=hparams['modes2'],
                      modes3=hparams['modes3'],
                      width=hparams['width'],
                      beta_1=hparams['beta_1'],
                      beta_2=hparams['beta_2'],
                      model=hparams['model'],
                      lr=hparams['learning_rate'],)
        log_path = Path(f"./lightning_logs/{hparams['net_name']}")
        log_path.mkdir(parents=True, exist_ok=True)
        with open(log_path / "hparam_config.json", 'w') as f:
            json.dump(hparams, f)

    # Add some checkpointing callbacks
    cbs = [ModelCheckpoint(monitor="loss", filename="{epoch:02d}-{loss:.2f}",
                           dirpath=logger.log_dir,
                           save_top_k=1,
                           mode="min"),
           ModelCheckpoint(monitor="val_loss", filename="{epoch:02d}-{val_loss:.2f}",
                           dirpath=logger.log_dir,
                           save_top_k=1,
                           mode="min"),
           EarlyStopping(monitor="val_loss", check_finite=False, patience=13)]

    trainer = pl.Trainer(
        callbacks=cbs,  # Add the checkpoint callback
        max_epochs=hparams['epochs'],
        check_val_every_n_epoch=hparams['val_interval'],
        log_every_n_steps=n_samples * split[0],
        logger=logger
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    wandb.finish()

#    predictions = trainer.predict(model, dataloaders=test_dataloader)
#    i = 0
#    for batch in predictions:
        # print(batch['div'], batch['div_hat'])
#        for j in range(0,20,2):
#            plt.close('all')
#            plt.figure(figsize=(8,3))
        #plt.close('all')
#            plt.subplot(1, 2, 1)
#            h = plt.imshow(batch['j'][-1].cpu().numpy()[0, :, :, j])
#            plt.colorbar(orientation='horizontal')
#            plt.title('Ground Truth')
#            cmin, cmax = h.get_clim()
#            plt.subplot(1, 2, 2)
#            plt.imshow(batch['jhat'][-1].cpu().numpy()[0, :, :, j], vmin=cmin, vmax=cmax)
#            plt.colorbar(orientation='horizontal')
#            plt.title('Prediction')
#            plt.tight_layout()
#            plt.savefig(f'./fno_figures/epoch_10/prediction_s{i}_t{10+j}.png',dpi=300)
        
#        i = i + 1
