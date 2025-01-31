import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping#, LearningRateMonitor, GradientAccumulationScheduler
from network_lightning import FNO3D
from FNO3D import tFNO3DModel
from dataloader_utils import get_dataloader, split_indices
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
import json
from quality import calculate_quality_over_sim


if __name__ == "__main__":

    net_name = "small"
    with open(f"lightning_logs/{net_name}/hparam_config.json", 'r') as f:
        json_string = f.read()

    hparams = json.loads(json_string)

    torch.set_float32_matmul_precision("medium")

    n_samples = 100
    split = [0.6, 0.2, 0.2]
    image_ids = np.random.randint(low=0, high=799, size=(n_samples,))
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(image_ids,
                                                                       data_path="/scratch/08780/cedar996/lbfoam/fno/lbfoam_rough",
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
    quality = np.empty((2*len(test_ids), hparams['T_out']), dtype=np.float64)


    for i, batch in enumerate(predictions):
        np.save(npy_path / f"Sample{test_ids[i]:04}_prediction", batch["jhat"][-1][0, :, :, :].squeeze().cpu().numpy())
        geom_file = Path(f"/scratch/08780/cedar996/lbfoam/fno/lbfoam_sims/135_{test_ids[i]:04}_1280_550/fracture.raw")
        data_truth = batch["j"][-1][0, :, :, :].squeeze().cpu().numpy()
        data_pred = batch["jhat"][-1][0, :, :, :].squeeze().cpu().numpy()

        quality[2*i] = calculate_quality_over_sim(data_truth, geom_file,nx=1280, ny=550,  skip=1)
        quality[2*i+1] = calculate_quality_over_sim( data_pred, geom_file, nx=1280, ny=550,skip=1)

        plt.figure()
        plt.plot(quality[2 * i], 'k--', label='Ground Truth')
        plt.plot(quality[2 * i + 1], 'r--', label='Prediction')
        plt.xlabel('Time step')
        plt.ylabel('Quality')
        plt.xlim(hparams['T_in'],hparams['T_in']+hparams['T_out'])
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(png_path / f'Quality_Sample{test_ids[i]:04}.png', dpi=300)

        for j in range(0, hparams['T_out'], 1):

            plt.close('all')
            plt.figure(figsize=(12, 3))
            # plt.close('all')
            plt.subplot(1, 3, 1)
            h = plt.imshow(data_truth[:, :, j])
            plt.colorbar(orientation='horizontal')
            plt.title('Ground Truth')
            plt.axis('off')
            cmin, cmax = h.get_clim()
            plt.subplot(1, 3, 2)
            plt.imshow(data_pred[:, :, j], vmin=cmin, vmax=cmax)
            plt.colorbar(orientation='horizontal')
            plt.title('Prediction')
            plt.axis('off')

            y = data_truth[:, :, j]
            y_pred = data_pred[:, :, j]
            y_rel_err = abs(y-y_pred) / abs(y)
            y_rel_err[np.isnan(y_rel_err)] = 0
            y_rel_err[np.isinf(y_rel_err)] = 0

            plt.subplot(1, 3, 3)
            plt.imshow(y_rel_err[:, :], vmin=0, vmax=np.nanpercentile(y_rel_err[:, :], 98))
            plt.colorbar(orientation='horizontal')
            plt.title('Rel. Error')
            plt.axis('off')
            plt.suptitle(f"Sample {test_ids[i]:04} at time step {10 + j}")
            plt.tight_layout()
            plt.savefig(png_path/f'prediction_Sample{test_ids[i]:04}_t{10 + j}.png', dpi=300)


    quality_df = pd.DataFrame(quality)
    quality_df.to_csv(npy_path / f"Truth_target_quality.csv", index=False)