import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from src.utils import save_object, evaluate_models
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):

        try:
            logging.info("Splitting training and test input data")

            X_train, y_train = (
                train_arr[:, :-1],
                train_arr[:, -1]
            )
            X_test, y_test = (
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            logging.info("Creating models")

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            }

            logging.info("Evaluating models")

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            logging.info(f"Model report: {model_report}")

            # Get best model score
            best_model_score = max(
                model_report.values(),
                key=lambda x: x["test_accuracy"]
            )["test_accuracy"]

            # Get best model name
            best_model_name = max(
                model_report,
                key=lambda x: model_report[x]["test_accuracy"]
            )

            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name}")
            logging.info(f"Best model accuracy: {best_model_score}")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best model saved successfully")
            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
