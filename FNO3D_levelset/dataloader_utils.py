import numpy as np
from scipy.ndimage import distance_transform_edt as edist
import torch
from hdf5storage import loadmat
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path
import torch.nn.functional as F

_MIN = 0
_MAX = 1000

class NumpyDataset(Dataset):
    def __init__(self, image_ids, data_dir, t_in, t_out):
        self.image_ids = image_ids
        self.data_dir = Path(data_dir)
        self.t_in = t_in
        self.t_out = t_out

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        try:
            # Load data from .npy files
            input_field = np.fromfile(self.data_dir / "training_data"/f"input_{self.image_ids[idx]:04}_0_35.raw", dtype=np.float32).reshape((self.t_in, 128,128))
            input_field = np.transpose(input_field, (1,2,0))
            input_field, _, _ = self.z_score_normalize(input_field)
            
            input_field = torch.from_numpy(input_field).float()
            input_field = input_field.reshape(128, 128, 1, self.t_in).repeat([1, 1, self.t_out, 1])
            
            
            # Target fields
            output_field = np.fromfile(self.data_dir / "training_data"/f"target_{self.image_ids[idx]:04}_40_115.raw", dtype=np.float32).reshape((self.t_out, 128, 128))
            output_field = np.transpose(output_field, (1,2,0))
            output_field, original_means, original_stds = self.z_score_normalize(output_field)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File {e} not found.")

        output_field = torch.from_numpy(output_field).float()
        original_means = torch.tensor(original_means)
        original_stds = torch.tensor(original_stds)
        #print('Normal Target', output_field.min(), output_field.max())
        return input_field, output_field, original_means, original_stds

    def z_score_normalize(self, data):
        """
        Perform Z-score normalization along the last dimension (seq_len).
        Args:
            data (np.ndarray): Input data of shape [height, width, seq_len].
        Returns:
            normalized_data (np.ndarray): Normalized data of shape [height, width, seq_len].
            means (np.ndarray): Means along the seq_len dimension, shape [height, width].
            stds (np.ndarray): Standard deviations along the seq_len dimension, shape [height, width].
        """
        means = np.mean(data, axis=-1, keepdims=True)  # Shape: [height, width, 1]
        stds = np.std(data, axis=-1, keepdims=True)    # Shape: [height, width, 1]
        stds[stds==0] = 1e-8
        normalized_data = (data - means) / stds
        return normalized_data, means, stds
    

def get_dataloader(image_ids, data_path, t_in, t_out, split, batch=1, num_workers=4, seed=1261613, **kwargs):

    dataset = NumpyDataset(image_ids=image_ids, data_dir=data_path, t_in=t_in, t_out=t_out)
    generator = torch.Generator().manual_seed(seed)
    assert len(split) == 3, "Split must be a list of length 3."
    assert sum(split) == 1., "Sum of split must equal one."
    train_set, val_set, test_set = random_split(dataset, split, generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, persistent_workers=True, num_workers=num_workers, **kwargs)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, persistent_workers=True, num_workers=num_workers, **kwargs)
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False, num_workers=num_workers, **kwargs)

    return train_loader, val_loader, test_loader

def split_indices(indices, split, seed=None):
    if seed is not None:
        np.random.seed(seed)

    assert len(split) == 3, "Split must be a list of length 3."
    assert sum(split) == 1.0, "Sum of split must equal one."

    np.random.shuffle(indices)
    train_size = int(split[0] * len(indices))
    val_size = int(split[1] * len(indices))

    train_ids = indices[:train_size]
    val_ids = indices[train_size: (val_size + train_size)]
    test_ids = indices[(val_size + train_size):]

    return train_ids, val_ids, test_ids

