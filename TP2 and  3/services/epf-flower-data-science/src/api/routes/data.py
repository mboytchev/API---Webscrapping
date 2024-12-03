import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import opendatasets as od
from kaggle.api.kaggle_api_extended import KaggleApi

router = APIRouter()

# Chemins des fichiers de configuration
CONFIG_FILE_PATH = "src/config/datasets.json"
KAGGLE_CONFIG_PATH = "src/config/kaggle.json"
DATA_DIR = "src/data"

# Vérification de l'existence du fichier de configuration
def check_config_file():
    if not Path(CONFIG_FILE_PATH).exists():
        raise HTTPException(status_code=404, detail="Le fichier datasets.json n'existe pas dans src/config")

# Charger la configuration des datasets depuis le fichier JSON
def load_config():
    check_config_file()
    with open(CONFIG_FILE_PATH, "r") as f:
        return json.load(f)

# Sauvegarder la configuration des datasets dans le fichier JSON
def save_config(config):
    with open(CONFIG_FILE_PATH, "w") as f:
        json.dump(config, f, indent=4)

# Télécharger un dataset à partir de Kaggle en utilisant 'opendatasets' ou l'API Kaggle
def download_kaggle_dataset(dataset_url: str, destination: str):
    try:
        api = KaggleApi()
        api.authenticate()  # Authentification à l'API Kaggle

        # Extraire le nom du dataset à partir de l'URL
        dataset_name = dataset_url.split('/')[-1]

        # Télécharger le dataset avec 'opendatasets'
        od.download(dataset_url, destination, force=True)

        return {"message": f"Dataset {dataset_name} téléchargé avec succès dans {destination}."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors du téléchargement : {e}")

# Modèle Pydantic pour la gestion des datasets
class Dataset(BaseModel):
    name: str
    url: str

# Route pour récupérer les datasets configurés
@router.get("/datasets")
async def get_datasets():
    config = load_config()
    keys = list(config.keys())
    return {"datasets": keys}

# Route pour ajouter un dataset à la configuration
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

# Route pour mettre à jour un dataset
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

# Route pour télécharger un dataset spécifié
@router.get("/download-dataset/{dataset_key}")
def get_dataset(dataset_key: str):
    """
    Télécharger un dataset spécifié dans le fichier de configuration.
    """
    try:
        datasets_config = load_config()

        # Vérifier si la clé du dataset existe dans le fichier de config
        dataset_info = datasets_config.get(dataset_key)
        if not dataset_info:
            raise HTTPException(status_code=404, detail="Dataset non trouvé dans la configuration.")

        dataset_url = dataset_info.get("url")
        if not dataset_url:
            raise HTTPException(status_code=400, detail="URL du dataset manquante dans la configuration.")

        # Télécharger le dataset
        return download_kaggle_dataset(dataset_url, DATA_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

