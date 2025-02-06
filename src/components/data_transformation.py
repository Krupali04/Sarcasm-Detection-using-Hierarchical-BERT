import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ['Year', 
                                'Adult Mortality',
                                'infant deaths', 
                                'Alcohol', 
                                'percentage expenditure', 
                                'Hepatitis B', 'Measles ', 
                                ' BMI ', 
                                'under-five deaths ', 
                                'Polio', 
                                'Total expenditure', 
                                'Diphtheria ', 
                                ' HIV/AIDS', 
                                'GDP', 
                                'Population', 
                                ' thinness  1-19 years', 
                                ' thinness 5-9 years', 
                                'Income composition of resources', 
                                'Schooling']
            categorical_columns = ['Country', 'Status']

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="mean")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ("ordinal_encoder",OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)

                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()


            target_column_name='Life expectancy '
            logging.info(train_df[target_column_name])

            logging.info(f"Checking missing values in target column '{target_column_name}'")
            missing_train = train_df[target_column_name].isnull().sum()
            missing_test = test_df[target_column_name].isnull().sum()

            if missing_train > 0 or missing_test > 0:
              logging.info(f"Missing values found in target column: {missing_train} in train, {missing_test} in test")
            
            train_df[target_column_name] = train_df[target_column_name].fillna(train_df[target_column_name].mean())
            test_df[target_column_name] = test_df[target_column_name].fillna(test_df[target_column_name].mean())

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path,
                )
        
        except Exception as e:
            raise CustomException(e,sys)


