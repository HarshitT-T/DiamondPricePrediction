from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

##pipelines
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

import sys,os 
from dataclasses import dataclasses

import numpy as np
import pandas as pd


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

##DataTransformation COnfig

@dataclasses
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


## Data INgestion class 

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):

        try:
            logging.info('Data Transformation initiated')
            
            cat_columns = ['cut','color','clarity']
            num_columns = ['carat','depth','table','x','y','z']

            cut_ranking = ['Fair','Good','Very Good','Premium','Ideal' ]
            col_category = ['D','E','F','J','H','I','J']
            cla_category = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline Initaited')

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy = 'most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_ranking,col_category,cla_category])),
                    ('scaler',StandardScaler())
                ]

            
            )
            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,num_columns),
                ('cat_pipeline',cat_pipeline,cat_columns)
            ])
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:

            logging.info("Error in Data Transforamtion")
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            #Reading Training and Test Data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Read train and test data completed')
            logging.info(f'Train DataFrame Head : \n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head : /n{test_df.head().to_string()}')

            logging.info('Obtaining Preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            ## feature into independent and dependent features

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df(columns = drop_columns,axis = 1)
            target_feature_test_df = test_df[target_column_name]

            ## applyt the tarnsformation

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df) 
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and testing datasets')


            train_arr = np.c_[input_feature_train_arr,np.arr(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_obj(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj
            )

            logging.info('Processor pickle in created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except e as Exception:
            logging.info('Exception occured in the initiate_datatransformation')

            raise CustomException(e,sys)





        



