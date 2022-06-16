import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from config.config import settings
from common.utils import weights_init_uniform

class TrainDenoiseModel():
    def __init__(self, model, loss, optimizer, train_dataset, valid_dataset,
                lr: float = settings.TRAIN.LEARNING_RATE,
                epochs: int = settings.TRAIN.EPOCHS,
                batch_size: int = settings.TRAIN.BATCH_SIZE):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self.device = device
        self.model = model
        self.criterion = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer(model.parameters(), lr=lr)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset


    def train(self):
        logger.info('started training model')
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=True)

        model = self.model
        model.apply(weights_init_uniform)
        model.train()

        for e in range(1, self.epochs+1):
            epoch_loss = 0

            for batch in tqdm(train_loader):
                X_batch = batch[settings.TRAIN.INPUT_LABEL]
                y_batch = batch[settings.TRAIN.OUTPUT_LABEL]

                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()

                y_pred = model(X_batch)

                y_pred = y_pred.double()
                y_batch = y_batch.double()

                loss = self.criterion(y_pred, y_batch)               
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()

            valid_epoch_loss = 0

            for val_batch in tqdm(valid_loader):
                X_valid_batch = val_batch[settings.TRAIN.INPUT_LABEL]
                y_valid_batch = val_batch[settings.TRAIN.OUTPUT_LABEL]

                X_valid_batch, y_valid_batch = X_valid_batch.to(self.device), y_valid_batch.to(self.device)
                y_valid_pred = model(X_valid_batch)

                y_valid_pred = y_valid_pred.double()
                y_valid_batch = y_valid_batch.double()

                valid_epoch_loss += self.criterion(y_valid_pred, y_valid_batch).item()

            print(f'Epoch {e+0:03}: | Loss: {epoch_loss / len(train_loader):.5f}')
            print(f'Epoch {e+0:03}: | Val Loss: {valid_epoch_loss / len(valid_loader):.5f}')
            print('----------------------------------------------')

        logger.info('model trained')
        return model

    
    @staticmethod
    def save_model(model, path: str = settings.DATA.DENOISE_MODEL_PATH):
        logger.info(f'saving model to {path}')
        torch.save(model.state_dict(), path)

    
    def __call__(self):
        trained_model = self.train()
        self.save_model(trained_model)
