import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk
nltk.download("stopwords")
nltk.download("punkt")

## Ensure the log dir exist
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

## logging configuration
logger=logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"data_preprocessing.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formtter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formtter)
file_handler.setFormatter(formtter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Tramsform the input text by covering it to lowercase,tokenization,removing stopworlds and punctuations and stemming.
    """
    ps=PorterStemmer()
    #conver text into lowercase
    text=text.lower()
    #tokenize the worlds
    text=nltk.word_tokenize(text)
    #removing non-alphabetic tokens
    text=[world for world in text if world.isalnum()]
    #remove stop worlds
    txet=[world for world in text if world not in stopwords.words("english") and world not in string.punctuation]
    #stem the world
    txet=[ps.stem(world) for world in text]
    #joining the tokens back into a single string
    return " ".join(text)

def preprocess_df(df:pd.DataFrame,text_column="text",target_column="target")->pd.DataFrame:
    """
    Preprocessing the df by encoding the target columns, removing dublicates and tranforming the text columns.
    """
    try:
        logger.debug("Preprocessing for data frame Started")
        #encode the target column
        encoder=LabelEncoder()
        logger.debug("Target columns Encoding Started")
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug("Target columns Encoded Done")

        #removing dublicates
        logger.debug("Removing Duplicates Started")
        df=df.drop_duplicates(keep="first")
        logger.debug("Removing Duplicates Done")

        ## applying text tranformation to the specified text column
        logger.debug("Text column transforming Started")
        df.loc[:,text_column]=df[text_column].apply(transform_text)
        logger.debug("Text column transforming Done")
        logger.debug("Preprocessing for data frame Done")
        return df
    except KeyError as e:
        logger.error(f"Column not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during text Text Normalization: {e}")
        raise

def main(text_column="text",target_column="target"):
    """
    main Function to load raw data , preprocess it and save the processed data
    """
    try:
        #fetch the data from data/raw
        train_data=pd.read_csv("./data/raw/train.csv")
        test_data=pd.read_csv("./data/raw/test.csv")
        logger.debug("Data loaded properly")

        #transform the data
        train_data_precessed=preprocess_df(train_data,text_column=text_column,target_column=target_column)
        test_data_precessed=preprocess_df(test_data,text_column=text_column,target_column=target_column)

        #Storing data inside data/preprocessed
        data_path=os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)
        train_data_precessed.to_csv(os.path.join(data_path,"train.csv"),index=False)
        test_data_precessed.to_csv(os.path.join(data_path,"test.csv"),index=False)

        logger.debug(f"processed data saved to : {data_path}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        logger.error(f"No data: {e}")
    except Exception as e:
        logger.error(f"failed to complete data transformation process: {e}")

if __name__=="__main__":
    main()
