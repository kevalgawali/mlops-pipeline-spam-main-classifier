import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from dvclive import Live

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

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

def load_model(file_path:str):
    """Load the trained model from a file."""
    try:
        with open(file_path,"rb") as f:
            model=pickle.load(f)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
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

def evaluate_model(clf,x_test:np.array,y_test:np.array)->dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred=clf.predict(x_test)
        y_pred_proba=clf.predict_proba(x_test)[:,1]

        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_pred_proba)
        metrics_dict={
            "accuracy":accuracy,
            "precison":precision,
            "recall":recall,
            "auc":auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics:dict,file_path:str)->None:
    """Save the evaluation metrics to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    try:
        params=load_params("params.yaml")
        clf=load_model("./models/model.pkl")
        test_data=load_data("./data/processed/test.csv")

        x_test=test_data.iloc[:,:-1].values
        y_test=test_data.iloc[:,-1].values

        metrics=evaluate_model(clf,x_test,y_test)

        with Live(save_dvc_exp=True) as live:
            live.log_metric("accuracy",metrics["accuracy"])
            live.log_metric("precison",metrics["precison"])
            live.log_metric("recall",metrics["recall"])
            live.log_metric("auc",metrics["auc"])

            live.log_params(params)

        save_metrics(metrics,"./reports/metrics.json")

    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")
    
if __name__ == '__main__':
    main()