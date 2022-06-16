import numpy as np
from loguru import logger
import torch
from torch.utils.data import Dataset

from common.utils import read_npy_file, get_pairs_noisy_clean_paths
from config.config import settings


class MelDenoiseDataset(Dataset):
    '''
    Mel-spectrogram - dataset for denoising.
    '''

    def __init__(self, root_dir: str, max_length: int=1500, noisy_folder_name: str=settings.DATA.NOISY_FOLDER_NAME):
        self.root_dir = root_dir
        self.paths_to_files = get_pairs_noisy_clean_paths(noisy_folder=root_dir + '\\' + noisy_folder_name)
        self.max_length = max_length
        self.noisy_folder_name = noisy_folder_name


    def __len__(self) -> int:
        return len(self.paths_to_files)


    def preprocess_array(self, arr: np.ndarray) -> np.ndarray:
        '''
        This method preprocesses given array with following transforms: transpose, convert to dtype=float32, pad to max_length.

        Parameters:
        ----------

        arr: np.ndarray
            Folder that contains noisy signals.

        Returns:
        -------

        preprocessed_array: np.ndarray
            Array after preprocessing.
        '''
        arr = arr.T
        arr = arr.astype('float32')

        return np.pad(arr, pad_width=((0, 0), (0, self.max_length - arr.shape[1])), mode='constant', constant_values=0)


    def __getitem__(self, idx) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        noisy = read_npy_file(self.paths_to_files[idx][0])
        clean = read_npy_file(self.paths_to_files[idx][1])
        initial_length = noisy.shape[0]
        noisy = self.preprocess_array(noisy)
        clean = self.preprocess_array(clean)
        clean = clean.flatten()

        sample = {settings.TRAIN.INPUT_LABEL : noisy, settings.TRAIN.OUTPUT_LABEL : clean, settings.TRAIN.SIGNAL_LENGTH: initial_length}

        return sample
