import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        country: str, year: int, status: str, adult_mortality: float, 
        infant_deaths: int, alcohol: float,percentage_expenditure:float,hepatitis_b:float,measles: int,
        under_five_deaths: int,polio: float,total_expenditure : float,
        diphtheria :float , hiv_aids :float,population: float,thinness_1_19_years: float,
        thinness_5_9_years :float,income_composition :float, bmi: float, gdp: float, 
        schooling: float):

        self.country = country
        self.year = year
        self.status = status
        self.adult_mortality = adult_mortality
        self.infant_deaths = infant_deaths
        self.alcohol = alcohol
        self.percentage_expenditure = percentage_expenditure
        self.hepatitis_b = hepatitis_b
        self.measles = measles
        self.under_five_deaths = under_five_deaths
        self.polio = polio
        self.total_expenditure = total_expenditure
        self.diphtheria = diphtheria
        self.hiv_aids = hiv_aids
        self.population = population
        self.thinness_1_19_years = thinness_1_19_years
        self.thinness_5_9_years = thinness_5_9_years
        self.income_composition = income_composition
        self.bmi = bmi
        self.gdp = gdp
        self.schooling = schooling


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Country": [self.country],
                "Year": [self.year],
                "Status": [self.status],
                "Adult Mortality": [self.adult_mortality],
                "infant deaths": [self.infant_deaths],
                "Alcohol": [self.alcohol],
                "percentage expenditure": [self.percentage_expenditure],
                "Hepatitis B": [self.hepatitis_b],
                "Measles ": [self.measles],
                "under-five deaths ": [self.under_five_deaths],
                "Polio": [self.polio],
                "Total expenditure": [self.total_expenditure],
                "Diphtheria ": [self.diphtheria],
                " HIV/AIDS": [self.hiv_aids],
                "Population": [self.population],
                " thinness  1-19 years": [self.thinness_1_19_years],
                " thinness 5-9 years": [self.thinness_5_9_years],
                "Income composition of resources": [self.income_composition],
                " BMI ": [self.bmi],
                "GDP": [self.gdp],
                "Schooling": [self.schooling]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)