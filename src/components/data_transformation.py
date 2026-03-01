import os 
import sys

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def data_transformation(self,train_path,test_path):

        try:
            df_train=pd.read_csv(train_path)
            df_test=pd.read_csv(test_path)
            logging.info("data reading is completed")

            df_train.fillna(df_train.median(), inplace=True)
            df_test.fillna(df_train.median(), inplace=True)
            
            df_train.drop("Unnamed: 0",axis=1,inplace=True)
            df_test.drop("Unnamed: 0",axis=1,inplace=True)

            logging.info("dropping null values or useless column")

            input_train_df=df_train.drop(columns=["SeriousDlqin2yrs"])
            target_train_df=df_train["SeriousDlqin2yrs"]

            input_test_df=df_test.drop(columns=["SeriousDlqin2yrs"])
            target_test_df=df_test["SeriousDlqin2yrs"]

            scaler = StandardScaler()

            input_feature_train_arr=scaler.fit_transform(input_train_df)
            input_feature_test_arr=scaler.transform(input_test_df)

            train_arr = np.c_[input_feature_train_arr, target_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_arr, target_test_df.to_numpy()]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=scaler
            )

            logging.info("Preprocessor object saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )

        except Exception as e:    
            raise CustomException(e, sys)
                
    
