import os
import sys
from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, 
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

BEST_THRESHOLD = -1.0

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def inititiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoostRegressor": AdaBoostRegressor(),
            }
            params={
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                              models=models, params=params)

            sorted_report = sorted(
                model_report.items(),
                key=lambda kv: kv[1]["mae"],
            )
            for name, metrics in sorted_report:
                logging.info(
                    "Model=%s MAE=%.4f RMSE=%.4f R2=%.4f MAPE=%.2f%%",
                    name,
                    metrics["mae"],
                    metrics["rmse"],
                    metrics["r2"],
                    metrics["mape"],
                )

            best_model_name = min(model_report, key=lambda name: model_report[name]["mae"])
            best_model_score = model_report[best_model_name]["mae"]
            best_model = models[best_model_name]

            if best_model_score < 0:
                logging.warning("Negative MAE detected (unexpected). Proceeding anyway.")
            logging.info(f"Best found model on both training and testing dataset: {best_model_name} with MAE: {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Ensure artifacts folder only contains expected .pkl files
            artifacts_dir = "artifacts"
            os.makedirs(artifacts_dir, exist_ok=True)
            keep_files = {
                "model.pkl",
                "preprocessor.pkl",
                "price_stats.pkl",
            }
            for filename in os.listdir(artifacts_dir):
                if filename.endswith(".pkl") and filename in keep_files:
                    continue
                if filename.endswith(".pkl") and filename not in keep_files:
                    continue
                if filename.endswith(".csv"):
                    try:
                        os.remove(os.path.join(artifacts_dir, filename))
                    except OSError:
                        pass

            predicted = best_model.predict(X_test)

            mae = mean_absolute_error(y_test, predicted)
            rmse = mean_squared_error(y_test, predicted) ** 0.5
            r2_square = r2_score(y_test, predicted)
            denom = np.where(y_test == 0, np.nan, y_test)
            mape = np.nanmean(np.abs((y_test - predicted) / denom)) * 100

            baseline_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)
            baseline_mae = mean_absolute_error(y_test, baseline_pred)
            baseline_rmse = mean_squared_error(y_test, baseline_pred) ** 0.5
            baseline_r2 = r2_score(y_test, baseline_pred)
            baseline_mape = np.nanmean(np.abs((y_test - baseline_pred) / denom)) * 100

            save_object(
                file_path=os.path.join("artifacts", "price_stats.pkl"),
                obj={
                    "train_prices": np.array(y_train, dtype=float),
                    "mae": float(mae),
                },
            )

            logging.info(
                "Model metrics: MAE=%.4f RMSE=%.4f R2=%.4f MAPE=%.2f%%",
                mae,
                rmse,
                r2_square,
                mape,
            )
            logging.info(
                "Baseline metrics: MAE=%.4f RMSE=%.4f R2=%.4f MAPE=%.2f%%",
                baseline_mae,
                baseline_rmse,
                baseline_r2,
                baseline_mape,
            )
            return {
                "mae": mae,
                "rmse": rmse,
                "r2": r2_square,
                "mape": float(mape),
                "baseline_mae": baseline_mae,
                "baseline_rmse": baseline_rmse,
                "baseline_r2": baseline_r2,
                "baseline_mape": float(baseline_mape),
            }
        

        except Exception as e:
            raise CustomException(e, sys)
