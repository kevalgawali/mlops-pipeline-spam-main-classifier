import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

## Ensure the log dir exist
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

## logging configuration
logger=logging.getLogger("feature_engineering")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"feature_engineering.log")
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
        df.fillna("",inplace=True)
        logger.debug(f"Data loaded and Na filled from {file_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to pass the csv file : {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading the data : {e}")
        raise

def apply_tfidf(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features: int)->tuple:
    """Apply TfIdf to data."""
    try:
        vectorizer=TfidfVectorizer(max_features=max_features)
        
        x_train=train_data["text"].values
        y_train=train_data["target"].values
        x_test=test_data["text"].values
        y_test=test_data["target"].values
        x_train_bow=vectorizer.fit_transform(x_train)
        x_test_bow=vectorizer.transform(x_test)
        train_df=pd.DataFrame(x_train_bow.toarray())
        train_df["label"]=y_train
        test_df=pd.DataFrame(x_test_bow.toarray())
        test_df["label"]=y_test

        logger.debug("tfidf applyed and data transformed")
        return train_df,test_df
    except Exception as e:
        logger.error(f"Unexpected error while Applying TfIdf: {e}")
        raise

def save_data(df:pd.DataFrame,file_path:str)->None:
    """Save data to csv file."""
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug(f"Data saved to : {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error while daving data frame : {e}")

def main():
    try:
        params = load_params(params_path="params.yaml")
        max_features=params["feature_engineering"]["max_features"]
        # max_features = 50

        train_data=load_data("./data/interim/train.csv")
        test_data=load_data("./data/interim/test.csv")
        train_df,test_df=apply_tfidf(train_data,test_data,max_features)
        save_data(train_df,os.path.join("./data","processed","train.csv"))
        save_data(test_df,os.path.join("./data","processed","test.csv"))
    except Exception as e:
        logger.error(f"Unexpected error while feature engineering process")
        print(f"Error : {e}")

if __name__=="__main__":
    main()