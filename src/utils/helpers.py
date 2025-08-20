"""
Helper utilities for the DataSci project.
"""

import os
import json
import pickle
import pandas as pd
from typing import Any, Dict, Union
from pathlib import Path


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        ValueError: If file format is not supported
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif file_path.suffix.lower() == '.json':
        return pd.read_json(file_path)
    elif file_path.suffix.lower() == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame to various file formats.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        file_path (str): Path where to save the file
        
    Raises:
        ValueError: If file format is not supported
    """
    file_path = Path(file_path)
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_path.suffix.lower() == '.csv':
        df.to_csv(file_path, index=False)
    elif file_path.suffix.lower() == '.xlsx':
        df.to_excel(file_path, index=False)
    elif file_path.suffix.lower() == '.json':
        df.to_json(file_path, orient='records', indent=2)
    elif file_path.suffix.lower() == '.parquet':
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_model(model: Any, file_path: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        file_path (str): Path where to save the model
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(file_path: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        file_path (str): Path to the saved model
        
    Returns:
        Any: Loaded model object
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_config(config: Dict[str, Any], file_path: str) -> None:
    """
    Save configuration dictionary to JSON file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        file_path (str): Path where to save the config
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        file_path (str): Path to the config file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Path to the project root
    """
    return Path(__file__).parent.parent.parent


def create_directory(dir_path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        dir_path (str): Directory path to create
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)
