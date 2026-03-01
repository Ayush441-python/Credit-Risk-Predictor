import os 
import sys
import dill
import pickle

import numpy as np 
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):

    try:
        report = {}
        for model_name, model in models.items():

            logging.info(f"Training {model_name}")

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            # Store results
            report[model_name] = {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "precision": test_precision,
                "recall": test_recall,
                "f1_score": test_f1
            }

            logging.info(f"{model_name} test accuracy: {test_accuracy}")

        return report

    except Exception as e:
        raise CustomException(e, sys)



def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

