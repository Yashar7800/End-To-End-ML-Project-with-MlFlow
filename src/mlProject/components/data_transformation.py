import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self,config: DataTransformationConfig):
        self.config = config

    # here we can add different transformation techniques like PCA, StandardScaler
    # since this data is already clean there is no need for above fucntions

    def train_test_split(self):
        data = pd.read_csv(self.config.data_path)

        train,test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir,'train.csv'),index = False)
        test.to_csv(os.path.join(self.config.root_dir,'test.csv'),index = False)

        logger.info("splitting Data into Train and Test")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)