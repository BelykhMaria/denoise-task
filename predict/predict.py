import torch
from torchmetrics import MeanSquaredError
from tqdm import tqdm
import copy
import numpy as np

from torch.utils.data import DataLoader
from config.config import settings
from common.model import GruNet

class PredictDenoiseModel():
    def __init__(self, test_dataset,
                model_path: str = settings.DATA.DENOISE_MODEL_PATH, 
                batch_size: int = 1):

        model = GruNet(hidden_dim=30, input_dim=1500, output_dim=1500*80, bidirectional=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(copy.deepcopy(torch.load(model_path,device)))
        model.to(device)
        self.model = model
        self.device = device
        self.model_path = model_path 
        self.test_dataset = test_dataset
        self.batch_size = batch_size


    def predict(self) -> list:
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

        model = self.model
        model.eval()

        test_mse = 0
        y_predict = []
        for batch in tqdm(test_loader):
            X_batch = batch[settings.TRAIN.INPUT_LABEL]
            y_batch = batch[settings.TRAIN.OUTPUT_LABEL]
            initial_length = batch[settings.TRAIN.SIGNAL_LENGTH]

            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            with torch.no_grad():
                y_pred = model(X_batch)

                y_pred = y_pred.double()
                y_batch = y_batch.double()

                mse = MeanSquaredError().to(self.device)
                test_mse += mse(y_pred, y_batch)

                y_pred = y_pred.reshape(80, 1500)
                y_predict.append(y_pred[:, :initial_length])

        print(f'Test MSE: {test_mse/len(test_loader)}')

        return y_predict


    def __call__(self):
        return self.predict()
