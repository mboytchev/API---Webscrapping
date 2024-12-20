import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import opendatasets as od
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

router = APIRouter()

CONFIG_FILE_PATH = "src/config/datasets.json"
KAGGLE_CONFIG_PATH = "src/config/kaggle.json"
DATA_DIR = "src/data"
INPUT_DIR = "src/data/iris"
OUTPUT_DIR = "src/data/processed"

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
    # Lire le fichier CSV
    df = pd.read_csv(file_path)

    # Séparer les features (X) et la cible (y)
    X = df.drop(columns=["Species"])  # Les features sont toutes les colonnes sauf "Species"
    y = df["Species"]  # La cible est la colonne "Species"

    # Traitement des valeurs manquantes (imputation) sur les features (X)
    imputer = SimpleImputer(strategy='mean')  # Remplacer les valeurs manquantes par la moyenne
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Normalisation des données (X_imputed)
    scaler = StandardScaler()  # Standardiser les features
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    # Combiner X_scaled et y (target) dans un seul DataFrame
    processed_df = X_scaled.copy()  # Copier les features traitées
    processed_df["Species"] = y  # Ajouter la cible "Species" à la fin du DataFrame

    # Sauvegarder le DataFrame traité dans un fichier CSV
    processed_df.to_csv(output_path, index=False)
    
    return output_path

class Dataset(BaseModel):
    name: str
    url: str

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

@router.get("/process-dataset/{dataset_key}")
def process_dataset_endpoint(dataset_key: str):
    """
    Traitement d'un dataset et retour du fichier traité.
    """
    try:
        datasets_config = load_config()
        dataset_info = datasets_config.get(dataset_key)

        if not dataset_info:
            raise HTTPException(status_code=404, detail="Dataset non trouvé dans la configuration.")

        dataset_url = dataset_info.get("url")
        if not dataset_url:
            raise HTTPException(status_code=400, detail="URL du dataset manquante dans la configuration.")

        # Chemin du fichier de dataset brut
        dataset_name = dataset_url.split('/')[-1]
        raw_file_path = os.path.join(INPUT_DIR, dataset_name + ".csv")  # Assurez-vous que le fichier est téléchargé au bon endroit

        # Vérification si le fichier existe
        if not os.path.exists(raw_file_path):
            raise HTTPException(status_code=404, detail="Le fichier du dataset n'existe pas.")

        # Chemin pour sauvegarder le fichier traité
        processed_file_path = os.path.join(OUTPUT_DIR, dataset_name + "_processed.csv")

        # Appeler la fonction de traitement
        processed_file = process_dataset(raw_file_path, processed_file_path)

        return {"message": "Le dataset a été traité avec succès.", "processed_file_path": processed_file}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
