import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

## Ensure the log dir exist
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

## logging configuration
logger=logging.getLogger("model_building")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"model_building.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formtter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formtter)
file_handler.setFormatter(formtter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)->dict:
    """Load parameters from yaml file."""
    try:
        with open(params_path,"r") as file:
            params=yaml.safe_load(file)
        logger.debug(f"Parameter retrived from {params_path}")
        return params
    except yaml.YAMLError as e:
        logger.error(f"YAML Error {e}")
        raise
    except Exception as e:
        logger.error(f"unexpected Error {e}")
        raise

def load_data(file_path:str)->pd.DataFrame:
    """Loading data from csv file"""
    try:
        df=pd.read_csv(file_path)
        logger.debug(f"Data loaded from : {file_path} with shape : {df.shape}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to pass the csv file : {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading the data : {e}")
        raise

def train_model(x_train:np.array,y_train:np.array,params:dict)->RandomForestClassifier:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """
    try:
        if x_train.shape[0]!=y_train.shape[0]:
            raise ValueError("The number of sample in x_train and y_train must be the same")
        
        logger.debug(f"Initializing random forest model with parameters: {params}")
        clf=RandomForestClassifier(n_estimators=params["n_estimators"],random_state=params["random_state"])

        logger.debug(f"Model training started with samples : {x_train.shape[0]}")
        clf.fit(x_train,y_train)
        logger.debug("Model training completed")
        
        return clf
    except ValueError as e:
        logger.error(f"value error during model training {e}")
        raise
    except Exception as e:
        logger.error(f"Error during model training : {e}")
        raise

def save_model(model,file_path:str)->None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        #ensure dir exist
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,"wb") as file:
            pickle.dump(model,file)
        logger.debug(f"Model saved to : {file_path}")
    except FileNotFoundError as e:
        logger.error(f"File path not found : {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occured while saving model : {e}")
        raise

def main():
    try:
        params=load_params("params.yaml")["model_building"]
        train_data=load_data("./data/processed/train.csv")
        x_train=train_data.iloc[:,:-1].values
        y_train=train_data.iloc[:,-1].values

        clf=train_model(x_train,y_train,params)

        model_save_path="models/model.pkl"
        save_model(clf,model_save_path)

    except Exception as e:
        logger.error(f"Failed to complete the model building process: {e}")
        print(f"Error : {e}")

if __name__=="__main__":
    main()
