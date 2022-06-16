from predict.predict import PredictDenoiseModel
from common.data import MelDenoiseDataset
from config.config import settings

if __name__ == "__main__":
    test_dataset = MelDenoiseDataset(root_dir=settings.DATA.DATA_PATH + '\\' + settings.DATA.TEST_FOLDER_NAME)
    y_predict = PredictDenoiseModel(test_dataset = test_dataset)()
