# app.py
# This is the main entry point for your ML project.
# It sets up the environment, imports necessary modules, and runs the data ingestion process.
import sys
from pathlib import Path

# Add 'src/' to sys.path BEFORE importing anything from mlproject
project_root_dir = Path(__file__).parent / "src"
if str(project_root_dir) not in sys.path:
    sys.path.insert(0, str(project_root_dir))

from mlproject.logger import logging
from mlproject.exception import CustomException
from mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig
from mlproject.components.data_transformation import DataTransformationConfig,DataTransformation
from mlproject.components.model_trainer import ModelTrainerConfig,ModelTrainer
# --- END RIGOROUS PATH HANDLING ---



if __name__=="__main__":
    logging.info("The execution has started")

    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        print("✅ Data Ingestion complete.")

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transormation(train_data_path, test_data_path)
        print("✅ Data Transformation complete.")

        model_trainer = ModelTrainer()
        r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(f"🎯 Final R² Score: {r2:.4f}")

        logging.info("Data Ingestion is completed")
        logging.info(f"Train data path: {train_data_path}")
        logging.info(f"Test data path: {test_data_path}")   
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
       
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
    