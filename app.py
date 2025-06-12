# app.py
# This is the main entry point for your ML project.
# It sets up the environment, imports necessary modules, and runs the data ingestion process.

import sys
from pathlib import Path
import mlproject.exception
print(dir(mlproject.exception))  # Debug: show available names
# Add src to the sys.path
project_root_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root_dir))

from mlproject.logger import logging
from mlproject.exception import CustomException
from mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig
# --- END RIGOROUS PATH HANDLING ---


if __name__=="__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()



        logging.info("Data Ingestion is completed")
        logging.info(f"Train data path: {train_data_path}")
        logging.info(f"Test data path: {test_data_path}")   
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
    