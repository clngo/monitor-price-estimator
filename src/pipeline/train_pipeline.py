import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


def run_training_pipeline():
    try:
        logging.info("Starting training pipeline")

        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()

        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        trainer = ModelTrainer()
        metrics = trainer.inititiate_model_trainer(train_arr, test_arr)
        logging.info("Training pipeline complete: %s", metrics)
        return metrics
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
