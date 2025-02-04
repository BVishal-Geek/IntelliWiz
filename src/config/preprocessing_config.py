from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PreprocessingConfig:
    # Data loading
    file_path: str
    target_column: Optional[str] = None
    
    # Missing values
    missing_threshold: float = 0.7
    numeric_imputation_strategy: str = 'mean'
    categorical_imputation_strategy: str = 'most_frequent'
    
    # Feature engineering
    create_interactions: bool = False
    max_interactions: int = 10
    perform_binning: bool = False
    n_bins: int = 5
    binning_strategy: str = 'quantile'
    polynomial_degree: int = 2
    
    # Text processing
    min_text_length: int = 50
    text_cleaning_operations: List[str] = None
    max_tfidf_features: int = 100
    create_tfidf: bool = False
    
    # Time series
    time_features: bool = True
    seasonality_detection: bool = False
    
    # Outliers
    outlier_detection_method: str = 'iqr'
    outlier_threshold: float = 1.5
    
    # Feature selection
    correlation_threshold: float = 0.95
    importance_threshold: float = 0.01
    
    # Scaling
    scaling_method: str = 'standard'
    
    def __post_init__(self):
        if self.text_cleaning_operations is None:
            self.text_cleaning_operations = [
                'lowercase',
                'remove_special_chars',
                'remove_extra_spaces'
            ]