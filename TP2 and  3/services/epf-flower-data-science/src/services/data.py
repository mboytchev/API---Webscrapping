import os
import json
from fastapi import APIRouter, HTTPException,Query
from pydantic import BaseModel
from pathlib import Path
import opendatasets as od
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from typing import List, Dict 
from google.cloud import firestore
from firebase_admin import firestore


CONFIG_FILE_PATH = "src/config/datasets.json"
KAGGLE_CONFIG_PATH = "src/config/kaggle.json"
DATA_DIR = "src/data"
IRIS_DIR = "src/data/iris"
PROCESSED_DIR = "src/data/processed_data"
MODEL_DIR = Path("src/models")
SPLIT_DATA_DIR = "src/data/split_data"
MODEL_PATH="src/models/iris_model.joblib"




def check_config_file():
    if not Path(CONFIG_FILE_PATH).exists():
        raise HTTPException(status_code=404, detail="The file datasets.json doesnt exist in existe pas dans src/config")

def load_config():
    check_config_file()
    with open(CONFIG_FILE_PATH, "r") as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_FILE_PATH, "w") as f:
        json.dump(config, f, indent=4)

def download_kaggle_dataset(dataset_url: str, destination: str):
    try:

        od.download(dataset_url, destination, force=True)

        dataset_name = dataset_url.split('/')[-1]
        dataset_path = Path(destination) / dataset_name
        csv_file = next((file for file in dataset_path.glob("*.csv")), None)
        
        if csv_file is None:
            raise HTTPException(status_code=404, detail="No CSV file found in the downloaded dataset")

        df = pd.read_csv(csv_file)
        if df.empty:
            raise HTTPException(status_code=404, detail="CSV file is empty")
        json_data = df.to_dict(orient="records")        
        return json_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during the downloading of CSV filde : {e}")

def process_dataset(file_path: str, output_path: str):
    """
    Processing a dataset (imputation of missing values, normalisation and target separation)
    """
    df = pd.read_csv(file_path)

    X = df.drop(columns=["Species"]) 
    y = df["Species"] 

    imputer = SimpleImputer(strategy='mean') 
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()  
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    processed_df = X_scaled.copy()  
    processed_df["Species"] = y 

  
    processed_df.to_csv(output_path, index=False)
    
    return output_path

def split_dataset(file_path: str, test_size: float = 0.2, random_state: int = 42, train_path: str = None, test_path: str = None):
    df = pd.read_csv(file_path)

    X = df.drop(columns=["Species"])
    y = df["Species"] 

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    if train_path:
        train_data.to_csv(train_path, index=False)

    if test_path:
        test_data.to_csv(test_path, index=False)

    return {"train": train_data.to_dict(orient="records"), "test": test_data.to_dict(orient="records")}

def load_model_parameters():
    try:
        with open('src/config/model_parameters.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during the loading of parameters model : {e}")