import sys
import os 
from dataclasses import dataclass


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass # provides __init__, __repr__, __eq__, 
class DataTransformationConfig:
    preprocess_obj_file_path=os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "shipping_cost",
                "screen_size_in",
                "resolution_pixels",
                "refresh_rate_hz",
                "response_time_ms",
            ]
            categorical_columns = [
                "condition",
                "brand",
                "color",
                "panel_type",
                "aspect_ratio",
                "hdr",
                "has_adaptive_sync",
            ]
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            ) 
            logging.info(f"Numerical columns encoding completed: {numerical_columns}")
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            logging.info(f"Categorical columns encoding completed: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )   

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "price_value"

            for df in (train_df, test_df):
                drop_cols = [c for c in ["item_id", "legacy_item_id", "item_url", "title"] if c in df.columns]
                if drop_cols:
                    df.drop(columns=drop_cols, inplace=True)

                if "resolution_width" in df.columns and "resolution_height" in df.columns:
                    df["resolution_pixels"] = df["resolution_width"] * df["resolution_height"]

                if "price_currency" in df.columns:
                    currency_values = df["price_currency"].dropna().unique()
                    if len(currency_values) > 1:
                        logging.warning("Multiple currencies detected in price_currency: %s", currency_values)
                    df.drop(columns=["price_currency"], inplace=True)

                if "shipping_cost_type" in df.columns and "shipping_cost" in df.columns:
                    df["shipping_cost"] = df["shipping_cost"].clip(lower=0, upper=100)

                if "model" in df.columns:
                    df.drop(columns=["model"], inplace=True)

                drop_optional = [
                    "condition_id",
                    "seller_feedback_pct",
                    "seller_feedback_score",
                    "buying_options",
                    "shipping_cost_type",
                ]
                drop_optional = [c for c in drop_optional if c in df.columns]
                if drop_optional:
                    df.drop(columns=drop_optional, inplace=True)

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"    
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocess_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
