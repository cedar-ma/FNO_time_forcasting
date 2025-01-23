import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping#, LearningRateMonitor, GradientAccumulationScheduler
from tnetwork_lightning import FNO3D
from tFNO3D import tFNO3DModel
from tdataloader_utils import get_dataloader, split_indices
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
import json


if __name__ == "__main__":
    # hparams = {
    #     'net_name': "hang",
    #     'learning_rate': 1e-3,
    #     'batch_size': 1,
    #     'epochs': 5,
    #     'val_interval': 5,
    #     'modes1': 12,
    #     'modes2': 12,
    #     'modes3': 12,
    #     'width': 30,
    #     'beta_1': 1,
    #     'beta_2': 1,
    #     'model': 'tFNO3DModel',
    #     'seed': 189031465,
    #     'patience': 9999
    # }
    # Preprocessing args
    net_name = "small"
    with open(f"lightning_logs/{net_name}/hparam_config.json", 'r') as f:
        json_string = f.read()

    hparams = json.loads(json_string)

    torch.set_float32_matmul_precision("medium")

    np.random.seed(189031465)

    n_samples = 5
    split = [0.6, 0.2, 0.2]
    image_ids = np.random.randint(low=0, high=4, size=(n_samples,))
    # train_dataloader, val_dataloader, test_dataloader = load_data()
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(image_ids,
                                                                       data_path="/scratch/08780/cedar996/lbfoam/fno/lbfoam_sims",
                                                                       seed=hparams['seed'],
                                                                       split=split,
                                                                       num_workers=2,
                                                                       pin_memory=True)

    _, _, test_ids = split_indices(image_ids, split, seed=hparams['seed'])
    try:
        model_dir = f"lightning_logs/{hparams['net_name']}/checkpoints"
        model_loc = glob.glob(f'{model_dir}/*val*.ckpt')[0]
        print(f'Loading {model_loc}')
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
                                           lr=hparams['learning_rate'], )
        model.eval()
    except IndexError:
        raise FileNotFoundError(
            f"Could not find checkpoint in {model_dir} or directory does not exist.")

    trainer = pl.Trainer()

    predictions = trainer.predict(model, dataloaders=test_dataloader)
    save_path = Path(f"/scratch/08780/cedar996/lbfoam/fno/results/{hparams['net_name']}/predictions")
    png_path = save_path / "figures"
    npy_path = save_path / "data"
    png_path.mkdir(parents=True, exist_ok=True)
    npy_path.mkdir(parents=True, exist_ok=True)


    for i, batch in enumerate(predictions):
        np.save(npy_path / f"Sample{test_ids[i]:04}_prediction", batch["jhat"][-1][0, :, :, :].squeeze().cpu().numpy())

        for j in range(0, 20, 1):

            plt.close('all')
            plt.figure(figsize=(12, 3))
            # plt.close('all')
            plt.subplot(1, 3, 1)
            h = plt.imshow(batch['j'][-1].cpu().numpy()[0, :, :, j])
            plt.colorbar(orientation='horizontal')
            plt.title('Ground Truth')
            plt.axis('off')
            cmin, cmax = h.get_clim()
            plt.subplot(1, 3, 2)
            plt.imshow(batch['jhat'][-1].cpu().numpy()[0, :, :, j], vmin=cmin, vmax=cmax)
            plt.colorbar(orientation='horizontal')
            plt.title('Prediction')
            plt.axis('off')

            y = batch['j'][-1].cpu().squeeze()
            y_pred = batch['jhat'][-1].cpu().squeeze()
            y_rel_err = abs(y[:,:,j] - y_pred[:,:,j]) / abs(y[:,:,j])
            y_rel_err[torch.isnan(y_rel_err)] = 0

            plt.subplot(1, 3, 3)
            plt.imshow(y_rel_err[:, :], vmin=0, vmax=np.nanpercentile(y_rel_err[:, :], 98))
            plt.colorbar(orientation='horizontal')
            plt.title('Rel. Error')
            plt.axis('off')
            plt.suptitle(f"Sample {test_ids[i]:04}")
            plt.tight_layout()
            plt.savefig(png_path/f'prediction_Sample{test_ids[i]:04}_t{10 + j}.png', dpi=300)


