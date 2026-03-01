from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.pipeline.train_pipeline import ModelTrainer
from src.logger import logging
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.inititate_data_ingestion()

    data_trans=DataTransformation()
    train_arr,test_arr,_=data_trans.data_transformation(train_data,test_data)

    train = ModelTrainer()
    score = train.initiate_model_trainer(train_arr,test_arr)
    print("Best model acc",score)
    
    logging.info("Everything is fine")
