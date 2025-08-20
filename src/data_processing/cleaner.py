"""
Data cleaning utilities for the DataSci project.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Union


class DataCleaner:
    """
    A class for cleaning and preprocessing data.
    """
    
    def __init__(self):
        """Initialize the DataCleaner."""
        pass
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        return df.drop_duplicates()
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: str = 'drop',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            strategy (str): Strategy for handling missing values ('drop', 'mean', 'median', 'mode')
            columns (List[str], optional): Specific columns to process
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        if columns is None:
            columns = df.columns.tolist()
        
        df_copy = df.copy()
        
        if strategy == 'drop':
            df_copy = df_copy.dropna(subset=columns)
        elif strategy == 'mean':
            for col in columns:
                if df_copy[col].dtype in ['int64', 'float64']:
                    df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in columns:
                if df_copy[col].dtype in ['int64', 'float64']:
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in columns:
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        
        return df_copy
    
    def remove_outliers(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        method: str = 'iqr'
    ) -> pd.DataFrame:
        """
        Remove outliers from specified columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to check for outliers
            method (str): Method for outlier detection ('iqr', 'zscore')
            
        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        df_copy = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                df_copy = df_copy[z_scores < 3]
        
        return df_copy
