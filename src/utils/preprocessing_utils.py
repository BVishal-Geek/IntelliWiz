import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
from scipy import stats

def detect_data_type(series: pd.Series) -> str:
    """
    Automatically detect the data type of a series
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    elif pd.api.types.is_numeric_dtype(series):
        if series.nunique() / len(series) < 0.05:  # Low cardinality
            return 'categorical'
        return 'numeric'
    elif pd.api.types.is_string_dtype(series):
        if series.str.contains(r'\d{4}-\d{2}-\d{2}').any():
            return 'datetime'
        elif series.nunique() / len(series) < 0.05:  # Low cardinality
            return 'categorical'
        elif series.str.len().mean() > 50:
            return 'text'
        return 'categorical'
    return 'unknown'

def calculate_feature_importance(df: pd.DataFrame, target: str) -> Dict[str, float]:
    """
    Calculate feature importance using correlation for numeric features
    and mutual information for categorical features
    """
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    
    importance_dict = {}
    X = df.drop(columns=[target])
    y = df[target]
    
    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            importance = abs(X[column].corr(y))
        else:
            if pd.api.types.is_numeric_dtype(y):
                importance = mutual_info_regression(
                    X[column].values.reshape(-1, 1), 
                    y
                )[0]
            else:
                importance = mutual_info_classif(
                    X[column].values.reshape(-1, 1), 
                    y
                )[0]
        importance_dict[column] = importance
    
    return importance_dict

def detect_seasonality(series: pd.Series, freq: str = 'D') -> Dict[str, float]:
    """
    Detect seasonality in time series data
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    try:
        result = seasonal_decompose(series, period=pd.Timedelta(freq))
        seasonality_strength = np.std(result.seasonal) / np.std(result.resid)
        return {
            'has_seasonality': seasonality_strength > 0.5,
            'seasonality_strength': seasonality_strength,
            'period': freq
        }
    except:
        return {
            'has_seasonality': False,
            'seasonality_strength': 0.0,
            'period': freq
        }

def detect_anomalies(series: pd.Series, method: str = 'zscore', threshold: float = 3) -> np.ndarray:
    """
    Detect anomalies using various methods
    """
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(series))
        return z_scores > threshold
    elif method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < (Q1 - threshold * IQR)) | (series > (Q3 + threshold * IQR))
    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        return iso_forest.fit_predict(series.values.reshape(-1, 1)) == -1
    return np.zeros(len(series), dtype=bool)