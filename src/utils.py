import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import json

from src.exception import CustomException

def get_latest_json(directory, pattern="*.json"):
    try:
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        files = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise FileNotFoundError(f"No JSON files matching {pattern} in {directory}")
        return str(files[0])
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def save_raw_json(data, file_path):
    """
    Save raw JSON to disk with timestamp.
    """
    try: 
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{file_path}_{timestamp}.json"

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        raise CustomException(e, sys)
    
def read_raw_json(file_path):
    try:
        with open(file_path, "r") as f:
            products_json = json.load(f)
        return products_json
    
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys) 

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for idx in range(len(list(models))):
            model = list(models.values())[idx]
            
            param = params[list(models.keys())[idx]]

            gs = GridSearchCV(model, param, cv=3) # why cv=3?
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train) # train model

            y_test_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_test_pred)
            rmse = mean_squared_error(y_test, y_test_pred) ** 0.5
            r2 = r2_score(y_test, y_test_pred)
            denom = np.where(y_test == 0, np.nan, y_test)
            mape = np.nanmean(np.abs((y_test - y_test_pred) / denom)) * 100

            report[list(models.keys())[idx]] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mape": mape,
            }

        return report
    except Exception as e:
        raise CustomException(e, sys)
