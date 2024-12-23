import pytest
from fastapi import HTTPException
from pathlib import Path
from unittest.mock import mock_open, patch
import json
import os
import pandas as pd
from data import check_config_file,save_config, load_config,process_dataset, CONFIG_FILE_PATH

# Test case for the check_config_file function
def test_check_config_file():
    if Path(CONFIG_FILE_PATH).exists():
        try:
            check_config_file()
        except HTTPException:
            pytest.fail("HTTPException raised unexpectedly!")
    else:
        with pytest.raises(HTTPException) as excinfo:
            check_config_file()
        assert excinfo.value.status_code == 404
        assert excinfo.value.detail == "The file datasets.json doesnt exist in existe pas dans src/config"

# Test case for the scenario where the config file exists and can be loaded successfully
@patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"key": "value"}))
@patch("pathlib.Path.exists", return_value=True)
def test_load_config_exists(mock_exists, mock_file):  # Renaming mock_open to mock_file to avoid conflict
    # The file exists and contains valid JSON data
    result = load_config()

    # Check that the returned result matches the expected data
    assert result == {"key": "value"}
    mock_file.assert_called_once_with(CONFIG_FILE_PATH, "r")  # Ensure the file was opened with the correct path

# Test case for the scenario where the config file does not exist
@patch("pathlib.Path.exists", return_value=False)
def test_load_config_file_not_found(mock_exists):
    # The file does not exist, so load_config should raise an HTTPException
    with pytest.raises(HTTPException) as excinfo:
        load_config()
    
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "The file datasets.json doesnt exist in existe pas dans src/config"


def test_process_dataset():
    #create a false dataset
    sample_data = """
    SepalLength,SepalWidth,PetalLength,PetalWidth,Species
    5.1,3.5,1.4,0.2,setosa
    4.9,NaN,1.4,0.2,setosa
    4.7,3.2,NaN,0.2,setosa
    4.6,3.1,1.5,NaN,versicolor
    NaN,3.6,1.4,0.2,virginica
    """

    #load fake data in a file
    file_path = "test_dataset.csv"
    output_path = "processed_dataset.csv"
    with open(file_path, "w") as f:
        f.write(sample_data)

    process_dataset(file_path, output_path)
    processed_df = pd.read_csv(output_path)

    assert not processed_df.isnull().values.any(), "Le fichier traité ne doit pas contenir de valeurs manquantes"
    assert "Species" in processed_df.columns, "La colonne cible 'Species' doit être présente"
    
    #verified
    original_df = pd.read_csv(file_path)
    assert processed_df.shape[0] == original_df.shape[0], "Le nombre de lignes doit rester constant"
    assert processed_df.shape[1] == original_df.shape[1], "Le nombre de colonnes doit rester constant"

    #delete temp file
    os.remove(file_path)
    os.remove(output_path)
    print("Test réussi!")