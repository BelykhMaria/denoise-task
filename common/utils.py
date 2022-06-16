import os
import numpy as np
from loguru import logger

from config.config import settings


def read_npy_file(path_to_file: str) -> np.ndarray:
    '''
    This function reads npy file.

    Parameters:
    -----------
        path_to_file: str
            Path to file with name of the file

    Returns:
    -------
        audio_file: np.ndarray
            Array representation of audio file.
    '''

    return np.load(path_to_file, allow_pickle=True)


def get_pairs_noisy_clean_paths(noisy_folder: str, ext: str = '.npy') -> list:
    '''
    This function returns paths to all noisy files and corresponding to them clean files.

    Parameters:
    ----------

    noisy_folder: str
        Folder that contains noisy signals
    ext: str (default = '.npy')
        The extension of files to search

    Returns:
    -------

        paths_pairs: list
            List of lists, where each sublist contains paths to noisy file and corresponding clean file.
    '''

    logger.info(f'getting all paths to noisy {ext} format files in {noisy_folder} and corresponding clean')
    noisy_paths = [os.path.join(root, name) for root, _, files in os.walk(noisy_folder) for name in files if name.endswith((ext))]
    paths_pairs = []
    for noisy_file in noisy_paths:
        clean_file = noisy_file.replace(settings.DATA.NOISY_FOLDER_NAME, settings.DATA.CLEAN_FOLDER_NAME)
        paths_pairs.append([noisy_file, clean_file])
    return paths_pairs


def weights_init_uniform(m):
    '''
    This function initialize model with uniformly distributed weights.
    '''

    logger.info('initializing model weights')
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)
