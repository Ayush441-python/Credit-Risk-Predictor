import os 
import sys 
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

import pandas as pd


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def inititate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df_train = pd.read_csv("notebook/data/cs-training.csv")
            df_test = pd.read_csv("notebook/data/cs-test.csv")
            logging.info('Read the dataset as dataframe')

            os.makedirs("artifacts", exist_ok=True)

            logging.info("Ingestion of the data is complete")
            df_train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            df_test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )



        except Exception as e:
            raise CustomException(e,sys)

        

            