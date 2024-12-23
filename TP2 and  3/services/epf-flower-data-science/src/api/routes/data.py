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
import google.auth

from src.services.data import check_config_file, load_config, save_config, download_kaggle_dataset, process_dataset, split_dataset, load_model_parameters

router = APIRouter()

CONFIG_FILE_PATH = "src/config/datasets.json"
KAGGLE_CONFIG_PATH = "src/config/kaggle.json"
DATA_DIR = "src/data"
IRIS_DIR = "src/data/iris"
PROCESSED_DIR = "src/data/processed_data"
MODEL_DIR = Path("src/models")
SPLIT_DATA_DIR = "src/data/split_data"
MODEL_PATH="src/models/iris_model.joblib"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "src/config/apidatasources-firebase.json"

class Dataset(BaseModel):
    name: str
    url: str


class InputDataList(BaseModel):
    features : list

#get the stored datasets
@router.get("/datasets")
async def get_datasets():
    config = load_config()
    keys = list(config.keys())
    return {"datasets": keys}

#add a dataset
@router.post("/{name}")
async def add_dataset(name: str, url: str):
    try:
        config = load_config()
        if name in config:
            raise HTTPException(status_code=400, detail=f"Dataset '{name}' already exists")
        config[name] = {"name": name, "url": url}
        save_config(config)

        return {"message": f"Dataset'{name}' has been updated succesfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#update datasets
@router.put("/{name}")
async def update_dataset(name: str, url: str):
    try:
        config = load_config()  
        if name not in config:
            raise HTTPException(status_code=404, detail=f"dataset '{name}' doesnt exist")
        
        current_url = config[name]["url"]
        if current_url != url:
            config[name]["url"] = url
            save_config(config)
            return {"message": f"Dataset URL '{name}' updated successfully"}
        else:
            return {"message": f"dataset URL '{name}' already updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#download a dataset from kaggle
@router.get("/download-dataset/{dataset_key}")
def get_dataset(dataset_key: str):
    """
    download a dataset from kaggle
    """
    try:
        datasets_config = load_config()
        dataset_info = datasets_config.get(dataset_key)
        if not dataset_info:
            raise HTTPException(status_code=404, detail="dataset didn't find")

        dataset_url = dataset_info.get("url")
        if not dataset_url:
            raise HTTPException(status_code=400, detail="Missig URL")

        return download_kaggle_dataset(dataset_url, DATA_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#process a dataset
@router.get("/process-dataset/{dataset_key}")
def process_dataset_endpoint(dataset_key: str):
    """
    process dataset and return the processed file
    """
    try:
        datasets_config = load_config()
        dataset_info = datasets_config.get(dataset_key)

        if not dataset_info:
            raise HTTPException(status_code=404, detail="dataset didn't find")

        dataset_url = dataset_info.get("url")
        if not dataset_url:
            raise HTTPException(status_code=400, detail="URL missing")

        dataset_name = dataset_url.split('/')[-1]
        raw_file_path = os.path.join(IRIS_DIR, dataset_name + ".csv") 

        if not os.path.exists(raw_file_path):
            raise HTTPException(status_code=404, detail="Dataset file doesn't exist.")

        processed_file_path = os.path.join(
            PROCESSED_DIR, dataset_name + "_processed.csv"
        )

        processed_file = process_dataset(raw_file_path, processed_file_path)

        return {"message": "Le dataset a été traité avec succès.", "processed_file_path": processed_file}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#split a dataset
@router.get("/split-dataset/{dataset_key}")
def split_dataset_endpoint(dataset_key: str, test_size: float = 0.2, random_state: int = 42):
    """
    Split a dataset in train and test
    """
    try:
        datasets_config = load_config()
        dataset_info = datasets_config.get(dataset_key)

        if not dataset_info:
            raise HTTPException(status_code=404, detail="dataset didn't find")

        dataset_name = dataset_key + "_processed.csv"
        processed_file_path = os.path.join(
            PROCESSED_DIR, dataset_name
        )

        if not os.path.exists(processed_file_path):
            raise HTTPException(status_code=404, detail="The dataset has not been processed yet")

        train_file_path = os.path.join(
            SPLIT_DATA_DIR, dataset_key + "_train.csv"
        )
        test_file_path = os.path.join(
            SPLIT_DATA_DIR, dataset_key + "_test.csv"
        )

        split_data = split_dataset(processed_file_path, test_size=test_size, random_state=random_state, train_path=train_file_path, test_path=test_file_path)

        return {
            "message": "Dataset splitted succesfully",
            "train_file_path": train_file_path,
            "test_file_path": test_file_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#train the model
@router.post("/train-model/{dataset_key}")
async def train_model(dataset_key: str):
    """
    Train a model
    """
    try:
        model_params = load_model_parameters()

        train_file_path = os.path.join(SPLIT_DATA_DIR, f"{dataset_key}_train.csv")
        test_file_path = os.path.join(SPLIT_DATA_DIR, f"{dataset_key}_test.csv")

        if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
            raise HTTPException(status_code=404, detail="files with data splitted (train/test) not found")

        train_df = pd.read_csv(train_file_path)
        test_df = pd.read_csv(test_file_path)

        X_train = train_df.drop(columns=["Species","Id"])
        y_train = train_df["Species"]

        X_test = test_df.drop(columns=["Species","Id"])
        y_test = test_df["Species"]

        model = LogisticRegression(
            C=model_params['LogisticRegression']['C'],
            solver=model_params['LogisticRegression']['solver'],
            max_iter=model_params['LogisticRegression']['max_iter'],
            penalty=model_params['LogisticRegression']['penalty'],
            multi_class=model_params['LogisticRegression']['multi_class']
        )

        model.fit(X_train, y_train)

        model_file_path = os.path.join(MODEL_DIR, f"{dataset_key}_model.joblib")
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, model_file_path)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {
            "message": "Model has been trained succesfully",
            "model_file_path": model_file_path,
            "accuracy": accuracy
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#get prediction
@router.post("/predict/{model_name}")
async def predict_with_model(input_data: InputDataList):
    """
    Predict with the model train
    """
    try:
        model = joblib.load(MODEL_PATH)

        input_data = pd.DataFrame([
            input_data.features]) 

        prediction = model.predict(input_data)
        print(prediction)

        return {"message": "Prediction successful", "prediction": prediction.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#get parameters of firestore
@router.get("/parameters")
async def get_parameters():
    """
    Retrieve all parameters
    """
    db = firestore.Client()
    doc_ref = db.collection("parameters").document("parameters")
    doc = doc_ref.get()
    
    if doc.exists:
        return doc.to_dict()
    else:
        return {"error": "Parameters document not found."}
    
credentials, _ = google.auth.default()
db = firestore.Client(credentials=credentials)

#create or update parameters
@router.put("/parameters/")
async def add_or_update_parameter(
    key: str = Query(..., description="Parameters to create or update"),
    value: int = Query(..., description="Parameters value")
):
    """
    Create or update a parameters
    """
    try:
        doc_ref = db.collection("parameters").document("parameters")

        if not doc_ref.get().exists:
            doc_ref.set({key: value})
            return {"message": f"parameters created as {key}: {value}."}
        else:
            doc_ref.update({key: value})
            return {"message": f"parameters updated as {key}: {value}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))