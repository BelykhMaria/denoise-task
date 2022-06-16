import torch
import random

from train.train import TrainDenoiseModel
from common.model import GruNet
from common.data import MelDenoiseDataset
from config.config import settings


if __name__ == "__main__":
    random.seed(10)
    model = GruNet(hidden_dim=30, input_dim=1500, output_dim=1500*80, bidirectional=True)
    train_dataset = MelDenoiseDataset(root_dir=settings.DATA.DATA_PATH + '\\' + settings.DATA.TRAIN_FOLDER_NAME)
    valid_dataset = MelDenoiseDataset(root_dir=settings.DATA.DATA_PATH + '\\' + settings.DATA.VALID_FOLDER_NAME)
    optimizer = torch.optim.Adam
    loss = torch.nn.MSELoss()
    train_denoise_model = TrainDenoiseModel(loss=loss, model=model, optimizer=optimizer, train_dataset=train_dataset, valid_dataset=valid_dataset)
    train_denoise_model()
