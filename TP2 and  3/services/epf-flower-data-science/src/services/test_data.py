import pytest
from fastapi import HTTPException
from pathlib import Path
from unittest.mock import mock_open, patch
import json
import os
from data import check_config_file,save_config, load_config, CONFIG_FILE_PATH

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

def test_save_config_success():
    """
    Test the successful saving of a JSON configuration file.
    """
    config_data = {"key1": "value1", "key2": "value2"}
    config_file_path = "src/config/test_save_config.json"

    os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
    save_config(config_data)
    with open(config_file_path, "r") as file:
        saved_data = json.load(file)

    assert saved_data == config_data
    os.remove(config_file_path)
