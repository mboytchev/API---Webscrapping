import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, RootModel
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
import firebase_admin
from google.cloud import firestore
from firebase_admin import firestore

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


def check_config_file():
    if not Path(CONFIG_FILE_PATH).exists():
        raise HTTPException(status_code=404, detail="Le fichier datasets.json n'existe pas dans src/config")

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
            raise HTTPException(status_code=404, detail="Aucun fichier CSV trouvé dans le dataset téléchargé.")

        df = pd.read_csv(csv_file)
        if df.empty:
            raise HTTPException(status_code=404, detail="Le fichier CSV est vide.")
        json_data = df.to_dict(orient="records")        
        return json_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors du téléchargement et de la conversion en JSON : {e}")

def process_dataset(file_path: str, output_path: str):
    """
    Fonction pour traiter un dataset (imputation des valeurs manquantes, normalisation et séparation de la cible).
    Sauvegarde les résultats dans un fichier CSV.
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
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement des paramètres du modèle : {e}")

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
            raise HTTPException(status_code=400, detail=f"Le dataset '{name}' existe déjà.")
        config[name] = {"name": name, "url": url}
        save_config(config)

        return {"message": f"Le dataset '{name}' a été ajouté avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#update datasets
@router.put("/{name}")
async def update_dataset(name: str, url: str):
    try:
        config = load_config()  
        if name not in config:
            raise HTTPException(status_code=404, detail=f"Le dataset '{name}' n'existe pas.")
        
        current_url = config[name]["url"]
        if current_url != url:
            config[name]["url"] = url
            save_config(config)
            return {"message": f"L'URL du dataset '{name}' a été mise à jour avec succès."}
        else:
            return {"message": f"L'URL du dataset '{name}' est déjà à jour."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#download a dataset from kaggle
@router.get("/download-dataset/{dataset_key}")
def get_dataset(dataset_key: str):
    try:
        datasets_config = load_config()
        dataset_info = datasets_config.get(dataset_key)
        if not dataset_info:
            raise HTTPException(status_code=404, detail="Dataset non trouvé dans la configuration.")

        dataset_url = dataset_info.get("url")
        if not dataset_url:
            raise HTTPException(status_code=400, detail="URL du dataset manquante dans la configuration.")

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
            raise HTTPException(status_code=404, detail="Dataset non trouvé dans la configuration.")

        dataset_url = dataset_info.get("url")
        if not dataset_url:
            raise HTTPException(status_code=400, detail="URL du dataset manquante dans la configuration.")

        dataset_name = dataset_url.split('/')[-1]
        raw_file_path = os.path.join(IRIS_DIR, dataset_name + ".csv") 

        if not os.path.exists(raw_file_path):
            raise HTTPException(status_code=404, detail="Le fichier du dataset n'existe pas.")

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
    Split the dataset in train and test.
    """
    try:
        datasets_config = load_config()
        dataset_info = datasets_config.get(dataset_key)

        if not dataset_info:
            raise HTTPException(status_code=404, detail="Dataset non trouvé dans la configuration.")

        dataset_name = dataset_key + "_processed.csv"
        processed_file_path = os.path.join(
            PROCESSED_DIR, dataset_name
        )

        if not os.path.exists(processed_file_path):
            raise HTTPException(status_code=404, detail="The dataset has not been processed yet.")

        train_file_path = os.path.join(
            SPLIT_DATA_DIR, dataset_key + "_train.csv"
        )
        test_file_path = os.path.join(
            SPLIT_DATA_DIR, dataset_key + "_test.csv"
        )

        split_data = split_dataset(processed_file_path, test_size=test_size, random_state=random_state, train_path=train_file_path, test_path=test_file_path)

        return {
            "message": "Le dataset a été divisé avec succès.",
            "train_file_path": train_file_path,
            "test_file_path": test_file_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#train the model
@router.post("/train-model/{dataset_key}")
async def train_model(dataset_key: str):
    """
    Endpoint to train a classification model and save it in the src/models directory.
    """
    try:
        model_params = load_model_parameters()

        train_file_path = os.path.join(SPLIT_DATA_DIR, f"{dataset_key}_train.csv")
        test_file_path = os.path.join(SPLIT_DATA_DIR, f"{dataset_key}_test.csv")

        if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
            raise HTTPException(status_code=404, detail="Les fichiers de données divisées (train/test) sont introuvables.")

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
            "message": "Le modèle a été entraîné et sauvegardé avec succès.",
            "model_file_path": model_file_path,
            "accuracy": accuracy
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/{model_name}")
async def predict_with_model(input_data: InputDataList):
    """
    Endpoint pour effectuer des prédictions avec un modèle entraîné.
    """
    try:
        model = joblib.load(MODEL_PATH)

        input_data = pd.DataFrame([input_data.features]) 

        prediction = model.predict(input_data)
        print(prediction)

        return {"message": "Prediction successful", "prediction": prediction.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/parameters")
async def get_parameters():
    # Retrieve the parameters document from 
    db = firestore.Client()
    doc_ref = db.collection("parameters").document("parameters")
    doc = doc_ref.get()
    
    if doc.exists:
        # Return the data as a response
        return doc.to_dict()
    else:
        # Return an error message if the document doesn't exist
        return {"error": "Parameters document not found."}