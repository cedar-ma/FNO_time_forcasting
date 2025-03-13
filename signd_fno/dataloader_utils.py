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
            input_field = np.fromfile(self.data_dir / f"135_{self.image_ids[idx]:04}_1280_550"/"input0_9_100.raw", dtype=np.uint8).reshape((self.t_in, 550,1280))
            input_field = np.transpose(input_field, (1,2,0))
            input_field = input_field[:,:550,:]
            input_field = self.sign_distance(input_field)
            #print('Signed input', input_field.min(), input_field.max())
            input_field, _, _ = self.z_score_normalize(input_field)
            #print('Normal input', input_field.min(), input_field.max())
            
            input_field = torch.from_numpy(input_field).float()
            #input_field = torch.tensor(input_field, dtype=torch.long)
            input_field = input_field.reshape(550, 550, 1, self.t_in).repeat([1, 1, self.t_out, 1])
            
            # one-hot encoding
            #input_field = F.one_hot(input_field, num_classes=3).float()
            #input_field = input_field.view(550, 550, self.t_out, -1)
            
            # Target fields
            output_field = np.fromfile(self.data_dir / f"135_{self.image_ids[idx]:04}_1280_550"/"target10_39_100.raw", dtype=np.uint8).reshape((self.t_out, 550, 1280))
            output_field = np.transpose(output_field, (1,2,0))
            output_field = output_field[:,:550,:]
            output_field = self.sign_distance(output_field)
            #print('Signed output', output_field.min(), output_field.max())
            output_field, original_mins, original_maxs = self.z_score_normalize(output_field)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File {e} not found.")

        output_field = torch.from_numpy(output_field).float()
        original_mins = torch.tensor(original_mins)
        original_maxs = torch.tensor(original_maxs)
        #print('Normal Target', output_field.min(), output_field.max())
        #print('Target slicewise mins', original_mins)
        #print('Target slicewise maxs', original_maxs)
        return input_field, output_field, original_mins, original_maxs

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
    
    def normalize_slices(self, data_3d):
        """
        Normalize to [-1,1]

        """
        original_mins = []
        original_maxs = []
        normalized_data = np.zeros_like(data_3d, dtype=np.float32)
        for i in range(data_3d.shape[-1]):
            slice_data = data_3d[...,i]
            slice_min = np.min(slice_data)
            slice_max = np.max(slice_data)
            if slice_max != slice_min:
                normalized_data[...,i] = 2*(slice_data - slice_min)/(slice_max - slice_min) - 1
                original_mins.append(slice_min)
                original_maxs.append(slice_max)
            else:
                normalized_data[...,i] = slice_data

        return normalized_data, original_mins, original_maxs

    def normalize01_slices(self, data_3d):
        """
        Normalize to [0,1]
        """
        original_mins = []
        original_maxs = []
        normalized_data = np.zeros_like(data_3d, dtype=np.float32)
        for i in range(data_3d.shape[-1]):
            slice_data = data_3d[...,i]
            slice_min = np.min(slice_data)
            slice_max = np.max(slice_data)
            if slice_max != slice_min:
                normalized_data[...,i] = (slice_data - slice_min)/(slice_max - slice_min)
                original_mins.append(slice_min)
                original_maxs.append(slice_max)
            else:
                normalized_data[...,i] = slice_data

        return normalized_data, original_mins, original_maxs



    def sign_distance(self, image):
        """
        Returns Signed distance transform map of the input image
        """

        grain = edist(image)
        sample = -1 * image + 1
        sample[sample==-1] = 0
        bubble = edist(sample)
        total = grain - bubble

        return total

    def slicewise_edt(self, image):
        slice_edt = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            slice_edt[i,:,:] = self.sign_distance(image[i,:,:])
        return slice_edt


    def normalize_grayscale(self, x: np.ndarray):
        """Normalize the unique gray levels to [0, 1]

        Parameters:
        ---
            x: np.ndarray of unique gray levels (for uint8: 0-255)
        """
        return (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    
    def linear_transform(self, x: np.ndarray, min_val: float=0., max_val: float=1.) -> np.ndarray:
        """Linearly varying conductivity
        
        Parameters:
        ---
            x: np.ndarray of unique gray levels (for uint8: 0-255)
            min_val: minimum phase conductivity
            max_val: maximum phase conductivity
        """
        val = self.normalize_grayscale(x)
        return val * (max_val - min_val) + min_val
    
    def zero_bounds(image):
        """ Make all boundary faces zero. This is useful because Dirichlet BCs are enforced in these voxels, and therefore, do not need to be trained.
        Parameters:
        ---
        image: 3D ndarray
        returns 3D ndarray copy image with boundary faces set equal to zero.
        """

        zero_bound = np.zeros_like(image)
        zero_bound[1:-1, 1:-1, 1:-1] = image[1:-1, 1:-1, 1:-1]
        return zero_bound

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

