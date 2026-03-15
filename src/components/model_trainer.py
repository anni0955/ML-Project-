import os 
import sys 
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor

from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting trainning and test data')
            x_train, y_train, x_test, y_test = (train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1])

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'KNN': KNeighborsRegressor(),
                'XGBoost': XGBRegressor(),
                'Adaboost': AdaBoostRegressor(),
                'SVM': SVR(),
                'Catboost': CatBoostRegressor()
            }

            params = {
                'Random Forest':{
                    'criterion':['squared_error', 'absolute_error'],
                    'n_estimators': [50, 100, 200],
                    'max_features': ['sqrt', 'log2', None]
                },
                'Decision Tree':{
                    'criterion':['squared_error', 'absolute_error'],
                    'max_features': ['sqrt', 'log2', None]
                },
                'Gradient Boosting': {
                    'learning_rate': [.1, .01, .05],
                    'subsample': [.6, .7, .8],
                    'n_estimators': [50, 100, 200]
                },
                'KNN': {
                    'n_neighbors': [3, 5, 7]
                },
                'XGBoost': {
                    'learning_rate': [.1, .01, .05],
                    'n_estimators': [50, 100, 200]
                },
                'Adaboost': {
                    'learning_rate': [.1, .01, .05],
                    'n_estimators': [50, 100, 200]
                },
                'Catboost': {
                    'learning_rate': [.1, .01, .05],
                    'n_estimators': [50, 100, 200],
                    'max_features': ['sqrt', 'log2', None]
                }
            }

            model_report = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, param=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < .6:
                raise CustomException('No best model found')
            
            logging.info('Best model on both training and testing dataset')

            save_object(file_path=self.model_trainer_config.train_model_file_path, obj=best_model)

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return (r2_square, best_model_name)
        
        except Exception as e:
            raise CustomException(e, sys)
            