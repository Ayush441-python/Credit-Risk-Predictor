import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

        
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        RevolvingUtilizationOfUnsecuredLines,
        age,
        NumberOfTime30_59DaysPastDueNotWorse,
        DebtRatio,
        MonthlyIncome,
        NumberOfOpenCreditLinesAndLoans,
        NumberOfTimes90DaysLate,
        NumberRealEstateLoansOrLines,
        NumberOfTime60_89DaysPastDueNotWorse,
        NumberOfDependents
    ):

        self.RevolvingUtilizationOfUnsecuredLines = RevolvingUtilizationOfUnsecuredLines
        self.age = age
        self.NumberOfTime30_59DaysPastDueNotWorse = NumberOfTime30_59DaysPastDueNotWorse
        self.DebtRatio = DebtRatio
        self.MonthlyIncome = MonthlyIncome
        self.NumberOfOpenCreditLinesAndLoans = NumberOfOpenCreditLinesAndLoans
        self.NumberOfTimes90DaysLate = NumberOfTimes90DaysLate
        self.NumberRealEstateLoansOrLines = NumberRealEstateLoansOrLines
        self.NumberOfTime60_89DaysPastDueNotWorse = NumberOfTime60_89DaysPastDueNotWorse
        self.NumberOfDependents = NumberOfDependents


    def get_data_as_dataframe(self):

        try:
            custom_data_input_dict = {

                "RevolvingUtilizationOfUnsecuredLines": [self.RevolvingUtilizationOfUnsecuredLines],
                "age": [self.age],
                "NumberOfTime30-59DaysPastDueNotWorse": [self.NumberOfTime30_59DaysPastDueNotWorse],
                "DebtRatio": [self.DebtRatio],
                "MonthlyIncome": [self.MonthlyIncome],
                "NumberOfOpenCreditLinesAndLoans": [self.NumberOfOpenCreditLinesAndLoans],
                "NumberOfTimes90DaysLate": [self.NumberOfTimes90DaysLate],
                "NumberRealEstateLoansOrLines": [self.NumberRealEstateLoansOrLines],
                "NumberOfTime60-89DaysPastDueNotWorse": [self.NumberOfTime60_89DaysPastDueNotWorse],
                "NumberOfDependents": [self.NumberOfDependents]

            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

    
