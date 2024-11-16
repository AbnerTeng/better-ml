from typing import Any, Dict
import pickle
import json

import yaml
import pandas as pd


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from file.

    Supported file types:
        - JSON
        - YAML

    Args:
        path: str, path to configuration file

    Returns:
        Dict[str, Any], configuration settings
    """
    file_type = path.split('.')[-1]

    if file_type == 'json':
        with open(path, 'r', encoding='utf-8') as json_file:
            config = json.load(json_file)

    elif file_type in ['yaml', 'yml']:
        with open(path, 'r', encoding='utf-8') as yaml_file:
            config = yaml.safe_load(yaml_file)

    else:
        raise ValueError(f"File type {file_type} not supported.")

    return config


def load_data(path: str) -> Any:
    """
    Load data from specific file type.

    Supported file types:
        - CSV
        - Parquet
        - Pickle

    Args:
        path: str, path to data file

    Returns:
        Any, loaded data
    """
    file_type = path.split('.')[-1]

    if file_type == 'csv':
        dt = pd.read_csv(path)

    elif file_type == 'parquet':
        dt = pd.read_parquet(path)

    elif file_type == "pkl":
        with open(path, 'rb') as pkl_file:
            dt = pickle.load(pkl_file)

    else:
        raise ValueError(f"File type {file_type} not supported.")

    return dt
