"""
Data Preprocessing Utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for ML projects.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Strategy for handling missing values
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        df_processed = df.copy()
        
        for column in df_processed.columns:
            if df_processed[column].isnull().sum() > 0:
                if df_processed[column].dtype in ['int64', 'float64']:
                    if strategy == 'mean':
                        df_processed[column].fillna(df_processed[column].mean(), inplace=True)
                    elif strategy == 'median':
                        df_processed[column].fillna(df_processed[column].median(), inplace=True)
                    elif strategy == 'mode':
                        df_processed[column].fillna(df_processed[column].mode()[0], inplace=True)
                else:
                    df_processed[column].fillna(df_processed[column].mode()[0], inplace=True)
                    
        return df_processed
    
    def encode_categorical_features(self, df, categorical_columns):
        """
        Encode categorical features using label encoding.
        
        Args:
            df (pd.DataFrame): Input dataframe
            categorical_columns (list): List of categorical column names
            
        Returns:
            pd.DataFrame: Encoded dataframe
        """
        df_encoded = df.copy()
        
        for column in categorical_columns:
            if column not in self.encoders:
                self.encoders[column] = LabelEncoder()
                df_encoded[column] = self.encoders[column].fit_transform(df_encoded[column])
            else:
                df_encoded[column] = self.encoders[column].transform(df_encoded[column])
                
        return df_encoded
    
    def scale_features(self, df, numerical_columns, method='standard'):
        """
        Scale numerical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            numerical_columns (list): List of numerical column names
            method (str): Scaling method ('standard' or 'minmax')
            
        Returns:
            pd.DataFrame: Scaled dataframe
        """
        df_scaled = df.copy()
        
        if method == 'standard':
            scaler_class = StandardScaler
        elif method == 'minmax':
            scaler_class = MinMaxScaler
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
            
        for column in numerical_columns:
            if column not in self.scalers:
                self.scalers[column] = scaler_class()
                df_scaled[column] = self.scalers[column].fit_transform(df_scaled[[column]])
            else:
                df_scaled[column] = self.scalers[column].transform(df_scaled[[column]])
                
        return df_scaled
    
    def create_train_test_split(self, X, y, test_size=0.2, random_state=42):
        """
        Create train-test split.
        
        Args:
            X: Features
            y: Target variable
            test_size (float): Test set size
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def preprocess_dataset(df, target_column, categorical_columns=None, numerical_columns=None):
    """
    Complete preprocessing pipeline for a dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of target column
        categorical_columns (list): List of categorical columns
        numerical_columns (list): List of numerical columns
        
    Returns:
        tuple: Processed features and target
    """
    preprocessor = DataPreprocessor()
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle missing values
    X = preprocessor.handle_missing_values(X)
    
    # Encode categorical features
    if categorical_columns:
        X = preprocessor.encode_categorical_features(X, categorical_columns)
    
    # Scale numerical features
    if numerical_columns:
        X = preprocessor.scale_features(X, numerical_columns)
    
    return X, y, preprocessor