import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping#, LearningRateMonitor, GradientAccumulationScheduler
from network_lightning import FNO3D
from FNO3D import GetFNO3DModel
from dataloader_utils import get_dataloader, split_indices
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
import json
from quality import calculate_quality_over_sim
import matplotlib as mpl

mpl.rcParams.update({'font.size': 14})


if __name__ == "__main__":

    net_name = "sign_zscore_t663"
    with open(f"lightning_logs/{net_name}/hparam_config.json", 'r') as f:
        json_string = f.read()

    hparams = json.loads(json_string)

    torch.set_float32_matmul_precision("medium")

    n_samples = 10
    split = [0.6, 0.2, 0.2]
    image_ids = np.random.randint(low=0, high=159, size=(n_samples,))
    in_T = int(hparams['T_in'])
    out_T=int(hparams['T_out'])
    time_series = np.arange(in_T, in_T+out_T,1)
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(image_ids,
                                                                       data_path="/scratch/08780/cedar996/lbfoam/level_set",
                                                                       t_in=hparams['T_in'],
                                                                       t_out=hparams['T_out'],
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
                                           in_channels=hparams['T_in'],
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
    x, y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
    predictions = trainer.predict(model, dataloaders=test_dataloader)
    save_path = Path(f"/scratch/08780/cedar996/lbfoam/level_set/results/{hparams['net_name']}")
    png_path = save_path / "figures"
    png_path.mkdir(parents=True, exist_ok=True)


    for i, batch in enumerate(predictions):
        data_truth = batch["j"][0, :, :, :].squeeze().cpu().numpy()
        data_pred = batch["jhat"][0, :, :, :].squeeze().cpu().numpy()


        for j in range(0, out_T, 1):

            plt.close('all')
            plt.figure(figsize=(12, 4))
            # plt.close('all')
            plt.subplot(1, 3, 1)
            #plt.contour(x, y, data_truth[...,j], levels=[0], colors='b')
            h = plt.imshow(data_truth[:, :, j])
            plt.colorbar(orientation='horizontal', fraction=0.05,pad=0.1)
            plt.title('Ground Truth')
            plt.axis('off')
            cmin, cmax = h.get_clim()
            plt.subplot(1, 3, 2)
            #plt.contour(x, y, data_pred[...,j], levels=[0], colors='b')
            plt.imshow(data_pred[:, :, j], vmin=cmin, vmax=cmax)
            plt.colorbar(orientation='horizontal', fraction=0.05,pad=0.1)
            plt.title('Prediction')
            plt.axis('off')

            y = data_truth[:, :, j]
            y_pred = data_pred[:, :, j]
            y_rel_err = abs(y-y_pred) / (abs(y)+1e-8)
            y_rel_err[np.isnan(y_rel_err)] = 0
            y_rel_err[np.isinf(y_rel_err)] = 0

            plt.subplot(1, 3, 3)
            plt.imshow(y_rel_err[:, :], vmin=-0.1, vmax=0.1)
            plt.colorbar(orientation='horizontal', fraction=0.05,pad=0.1)
            plt.title('Rel. Error')
            plt.axis('off')
            plt.suptitle(f"Sample {test_ids[i]:04} at time step {hparams['T_in'] + j}")
            plt.tight_layout()
            plt.savefig(png_path/f'prediction_Sample{test_ids[i]:04}_t{j}.png', dpi=300)


