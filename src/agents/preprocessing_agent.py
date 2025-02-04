import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.preprocessing_config import PreprocessingConfig
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import json
from itertools import combinations
from scipy import stats

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, 
    RobustScaler, 
    MinMaxScaler,
    LabelEncoder, 
    OneHotEncoder,
    PolynomialFeatures
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

from src.config.preprocessing_config import PreprocessingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PreprocessingAgent:
    def __init__(self, config: Union[PreprocessingConfig, str, dict]):
        """Initialize the preprocessing agent."""
        self.config = self._load_config(config)
        self.data = None
        self.original_data = None
        self.data_profile = {}
        self.transformers = {}
        self.feature_importance = {}
        
        # Column type containers - Initializes lists to store different types of columns
        self.numeric_columns = []
        self.categorical_columns = []
        self.text_columns = []
        self.date_columns = []
        self.binary_columns = []
        
        # Set up logging
        self.log_dir = Path("logs/preprocessing")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
        # Initialize preprocessing
        self._initialize_preprocessing()

    def _load_config(self, config: Union[PreprocessingConfig, str, dict]) -> PreprocessingConfig:
        """Load configuration from various input types."""
        try:
            if isinstance(config, PreprocessingConfig):
                return config
            elif isinstance(config, dict):
                return PreprocessingConfig(**config)
            elif isinstance(config, str):
                with open(config, 'r') as f:
                    config_dict = json.load(f)
                return PreprocessingConfig(**config_dict)
            else:
                raise ValueError("Invalid configuration input type")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise

    def _setup_logging(self):
        """Set up logging with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"preprocessing_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)

    def _initialize_preprocessing(self):
        """Initialize the preprocessing pipeline."""
        try:
            self._load_data()
            self._backup_original_data()
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _load_data(self):
        """Load data from the specified file path."""
        try:
            file_path = Path(self.config.file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path.suffix == '.csv':
                self.data = pd.read_csv(file_path)
            elif file_path.suffix == '.parquet':
                self.data = pd.read_parquet(file_path)
            elif file_path.suffix == '.xlsx':
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
            logger.info(f"Successfully loaded data from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _backup_original_data(self):
        """Create a backup of the original data."""
        if self.data is not None:
            self.original_data = self.data.copy()

    def analyze_data_types(self):
        """Analyze and categorize columns by data type."""
        logger.info("Starting data type analysis...")
        
        for column in self.data.columns:
            if column == self.config.target_column:
                continue
                
            dtype = self._detect_data_type(self.data[column])
            
            if dtype == 'numeric':
                self.numeric_columns.append(column)
            elif dtype == 'categorical':
                self.categorical_columns.append(column)
            elif dtype == 'datetime':
                self.date_columns.append(column)
                self._convert_to_datetime(column)
            elif dtype == 'text':
                self.text_columns.append(column)
            
            # Detect binary columns
            if dtype in ['numeric', 'categorical']:
                if self.data[column].nunique() == 2:
                    self.binary_columns.append(column)
        
        self._log_column_types()

    def _detect_data_type(self, series: pd.Series) -> str:
        """Detect the data type of a series."""
        try:
            if pd.api.types.is_datetime64_any_dtype(series):
                return 'datetime' 
            elif pd.api.types.is_numeric_dtype(series):
                if series.nunique() / len(series) < 0.05:  # Low cardinality
                    return 'categorical'
                return 'numeric'
            elif pd.api.types.is_string_dtype(series):
                if series.str.contains(r'\d{4}-\d{2}-\d{2}').any():
                    return 'datetime'
                elif series.nunique() / len(series) < 0.05:
                    return 'categorical'
                elif series.str.len().mean() > self.config.min_text_length:
                    return 'text'
                return 'categorical'
            return 'unknown'
        except Exception as e:
            logger.warning(f"Error detecting data type: {str(e)}")
            return 'unknown'
        
    def _log_column_types(self):
        """Log the detected column types."""
        type_summary = {
            'numeric': len(self.numeric_columns),
            'categorical': len(self.categorical_columns),
            'datetime': len(self.date_columns),
            'text': len(self.text_columns),
            'binary': len(self.binary_columns)
        }
        logger.info(f"Column type summary: {type_summary}")

    def _convert_to_datetime(self, column: str):
        """Convert a column to datetime format."""
        try:
            self.data[column] = pd.to_datetime(self.data[column])
            logger.info(f"Converted {column} to datetime")
        except Exception as e:
            logger.warning(f"Failed to convert {column} to datetime: {str(e)}")

    
    def _remove_duplicates(self):
        """Remove duplicate rows."""
        initial_rows = len(self.data)
        self.data.drop_duplicates(inplace=True)
        dropped_rows = initial_rows - len(self.data)
        logger.info(f"Removed {dropped_rows} duplicate rows")

    def _clean_text_data(self):
        """Clean text data using specified operations."""
        try:
            for col in self.text_columns:
                if 'lowercase' in self.config.text_cleaning_operations:
                    self.data[col] = self.data[col].str.lower()
                if 'remove_special_chars' in self.config.text_cleaning_operations:
                    self.data[col] = self.data[col].str.replace(r'[^\w\s]', '', regex=True)
                if 'remove_extra_spaces' in self.config.text_cleaning_operations:
                    self.data[col] = self.data[col].str.strip()
                    self.data[col] = self.data[col].str.replace(r'\s+', ' ', regex=True)
            logger.info("Text data cleaning completed")
        except Exception as e:
            logger.error(f"Error cleaning text data: {str(e)}")
            raise

    def _clean_categorical_data(self):
        """Clean categorical data."""
        try:
            for col in self.categorical_columns:
                # Handle rare categories
                value_counts = self.data[col].value_counts()
                rare_categories = value_counts[value_counts < len(self.data) * 0.01].index
                if len(rare_categories) > 0:
                    self.data[col] = self.data[col].replace(rare_categories, 'Other')
            logger.info("Categorical data cleaning completed")
        except Exception as e:
            logger.error(f"Error cleaning categorical data: {str(e)}")
            raise

    def _clean_numeric_data(self):
        """Clean numeric data."""
        try:
            for col in self.numeric_columns:
                # Replace infinite values with NaN
                self.data[col] = self.data[col].replace([np.inf, -np.inf], np.nan)
                
                # Apply scaling if specified
                if hasattr(self.config, 'scaling_method'):
                    if self.config.scaling_method == 'standard':
                        scaler = StandardScaler()
                    elif self.config.scaling_method == 'robust':
                        scaler = RobustScaler()
                    elif self.config.scaling_method == 'minmax':
                        scaler = MinMaxScaler()
                    self.data[col] = scaler.fit_transform(self.data[[col]])
                    self.transformers[f'scaler_{col}'] = scaler
            logger.info("Numeric data cleaning completed")
        except Exception as e:
            logger.error(f"Error cleaning numeric data: {str(e)}")
            raise

    def _clean_datetime_data(self):
        """Clean datetime data."""
        try:
            for col in self.date_columns:
                # Handle invalid dates
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
            logger.info("Datetime data cleaning completed")
        except Exception as e:
            logger.error(f"Error cleaning datetime data: {str(e)}")
            raise

    def _create_polynomial_features(self):
        """Create polynomial features."""
        try:
            poly = PolynomialFeatures(
                degree=self.config.polynomial_degree,
                include_bias=False
            )
            numeric_data = self.data[self.numeric_columns]
            poly_features = poly.fit_transform(numeric_data)
            
            feature_names = poly.get_feature_names_out(self.numeric_columns)
            for i, name in enumerate(feature_names[len(self.numeric_columns):]):
                self.data[f'poly_{name}'] = poly_features[:, i + len(self.numeric_columns)]
            
            logger.info(f"Created polynomial features of degree {self.config.polynomial_degree}")
        except Exception as e:
            logger.error(f"Error creating polynomial features: {str(e)}")
            raise

    def _bin_numeric_features(self):
        """Bin numeric features."""
        try:
            for col in self.numeric_columns:
                if self.config.binning_strategy == 'quantile':
                    self.data[f'{col}_binned'] = pd.qcut(
                        self.data[col],
                        q=self.config.n_bins,
                        labels=False,
                        duplicates='drop'
                    )
                else:  # uniform binning
                    self.data[f'{col}_binned'] = pd.cut(
                        self.data[col],
                        bins=self.config.n_bins,
                        labels=False
                    )
            logger.info(f"Created binned features using {self.config.binning_strategy} strategy")
        except Exception as e:
            logger.error(f"Error binning numeric features: {str(e)}")
            raise

    def clean_data(self):
        """Execute comprehensive data cleaning pipeline."""
        logger.info("Starting data cleaning process...")
        
        try:
            # Remove duplicates
            self._remove_duplicates()
            
            # Handle missing values
            self._handle_missing_values()
            
            # Handle outliers
            self._handle_outliers()
            
            # Clean text data
            if self.text_columns:
                self._clean_text_data()
            
            # Clean categorical data
            if self.categorical_columns:
                self._clean_categorical_data()
            
            # Clean numeric data
            if self.numeric_columns:
                self._clean_numeric_data()
            
            # Clean datetime data
            if self.date_columns:
                self._clean_datetime_data()
            
            logger.info("Data cleaning completed successfully")
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {str(e)}")
            raise

    def _handle_missing_values(self):
        """Handle missing values using configured strategies."""
        try:
            # Handle columns with too many missing values
            missing_prop = self.data.isnull().mean()
            cols_to_drop = missing_prop[missing_prop > self.config.missing_threshold].index
            
            if len(cols_to_drop) > 0:
                self.data.drop(columns=cols_to_drop, inplace=True)
                logger.info(f"Dropped {len(cols_to_drop)} columns with high missing values")
            
            # Impute numeric columns
            for col in self.numeric_columns:
                if self.data[col].isnull().any():
                    imputer = SimpleImputer(strategy=self.config.numeric_imputation_strategy)
                    self.data[col] = imputer.fit_transform(self.data[[col]])
                    self.transformers[f'imputer_{col}'] = imputer
            
            # Impute categorical columns
            for col in self.categorical_columns:
                if self.data[col].isnull().any():
                    imputer = SimpleImputer(strategy=self.config.categorical_imputation_strategy)
                    self.data[col] = imputer.fit_transform(self.data[[col]])
                    self.transformers[f'imputer_{col}'] = imputer
            
            logger.info("Missing value handling completed")
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise

    def _handle_outliers(self):
        """Handle outliers using configured method."""
        try:
            for col in self.numeric_columns:
                if self.config.outlier_detection_method == 'iqr':
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.config.outlier_threshold * IQR
                    upper_bound = Q3 + self.config.outlier_threshold * IQR
                    self.data[col] = self.data[col].clip(lower_bound, upper_bound)
                
                elif self.config.outlier_detection_method == 'zscore':
                    z_scores = np.abs(stats.zscore(self.data[col]))
                    outliers = z_scores > self.config.outlier_threshold
                    self.data.loc[outliers, col] = self.data[col].median()
            
            logger.info("Outlier handling completed")
            
        except Exception as e:
            logger.error(f"Error handling outliers: {str(e)}")
            raise

    def engineer_features(self):
        """Execute feature engineering pipeline."""
        logger.info("Starting feature engineering process...")
        
        try:
            # Create interaction features
            if self.config.create_interactions:
                self._create_interaction_features()
            
            # Create polynomial features
            if hasattr(self.config, 'polynomial_degree'):
                self._create_polynomial_features()
            
            # Bin numeric features
            if self.config.perform_binning:
                self._bin_numeric_features()
            
            # Create text features
            if self.text_columns and self.config.create_tfidf:
                self._create_text_features()
            
            # Create datetime features
            if self.date_columns and self.config.time_features:
                self._create_datetime_features()
            
            logger.info("Feature engineering completed successfully")
            
        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            raise

    def _create_interaction_features(self):
        """Create interaction features between numeric columns."""
        try:
            interactions = 0
            for col1, col2 in combinations(self.numeric_columns, 2):
                if interactions >= self.config.max_interactions:
                    break
                
                # Multiplication interaction
                self.data[f'{col1}_{col2}_mult'] = self.data[col1] * self.data[col2]
                
                # Division interaction (with error handling)
                if (self.data[col2] != 0).all():
                    self.data[f'{col1}_{col2}_div'] = self.data[col1] / self.data[col2]
                
                interactions += 2
            
            logger.info(f"Created {interactions} interaction features")
            
        except Exception as e:
            logger.error(f"Error creating interaction features: {str(e)}")
            raise

    def _create_datetime_features(self):
        """Create features from datetime columns."""
        try:
            for col in self.date_columns:
                # Extract basic time components
                self.data[f'{col}_year'] = self.data[col].dt.year
                self.data[f'{col}_month'] = self.data[col].dt.month
                self.data[f'{col}_day'] = self.data[col].dt.day
                self.data[f'{col}_dayofweek'] = self.data[col].dt.dayofweek
                self.data[f'{col}_quarter'] = self.data[col].dt.quarter
                self.data[f'{col}_is_weekend'] = self.data[col].dt.dayofweek.isin([5, 6]).astype(int)
            
            logger.info("Created datetime features")
            
        except Exception as e:
            logger.error(f"Error creating datetime features: {str(e)}")
            raise

    def _create_text_features(self):
        """Create features from text data."""
        try:
            for col in self.text_columns:
                # Basic text features
                self.data[f'{col}_length'] = self.data[col].str.len()
                self.data[f'{col}_word_count'] = self.data[col].str.split().str.len()
                
                # TF-IDF features
                if self.config.create_tfidf:
                    tfidf = TfidfVectorizer(
                        max_features=self.config.max_tfidf_features,
                        stop_words='english'
                    )
                    tfidf_matrix = tfidf.fit_transform(self.data[col].fillna(''))
                    
                    # Add top TF-IDF features
                    feature_names = tfidf.get_feature_names_out()
                    for i, name in enumerate(feature_names):
                        self.data[f'{col}_tfidf_{name}'] = tfidf_matrix[:, i].toarray()
                    
                    self.transformers[f'tfidf_{col}'] = tfidf
            
            logger.info("Created text features")
            
        except Exception as e:
            logger.error(f"Error creating text features: {str(e)}")
            raise

    def select_features(self):
        """Perform feature selection."""
        logger.info("Starting feature selection process...")
        
        try:
            selected_features = set(self.data.columns)
            
            # Keep target column if specified and exists in the DataFrame
            if self.config.target_column in self.data.columns:
                selected_features.add(self.config.target_column)
            
            # Remove low variance features
            if len(self.numeric_columns) > 0:
                selector = VarianceThreshold(threshold=0.01)
                selector.fit(self.data[self.numeric_columns])
                selected_numeric = np.array(self.numeric_columns)[selector.get_support()].tolist()
                selected_features &= set(selected_numeric)
            
            # Remove highly correlated features
            if len(self.numeric_columns) > 0:
                selected_features &= set(self._remove_correlated_features())
            
            # Update dataframe
            self.data = self.data[list(selected_features)]
            logger.info(f"Selected {len(selected_features)} features")
            
            return list(selected_features)
            
        except Exception as e:
            logger.error(f"Error during feature selection: {str(e)}")
            raise

    def _remove_correlated_features(self) -> List[str]:
        """Remove highly correlated features."""
        try:
            corr_matrix = self.data[self.numeric_columns].corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [
                column for column in upper.columns 
                if any(upper[column] > self.config.correlation_threshold)
            ]
            
            return [col for col in self.numeric_columns if col not in to_drop]
            
        except Exception as e:
            logger.error(f"Error removing correlated features: {str(e)}")
            raise

    def run_pipeline(self):
        """Execute the complete preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline...")
        
        try:
            # Initial data analysis
            self.analyze_data_types()
            
            # Data cleaning
            self.clean_data()
            
            # Feature engineering
            self.engineer_features()
            
            # Feature selection
            self.select_features()
            
            # Final validation
            self._validate_processed_data()
            
            logger.info("Preprocessing pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _validate_processed_data(self):
        """Validate the processed dataset."""
        try:
            # Check for missing values
            if self.data.isnull().any().any():
                logger.warning("Processed data contains missing values")
            
            # Check for infinite values
            if np.isinf(self.data.select_dtypes(include=np.number)).any().any():
                logger.warning("Processed data contains infinite values")
            
            # Log validation results
            logger.info(f"Final dataset shape: {self.data.shape}")
            logger.info(f"Memory usage: {self.data.memory_usage().sum() / 1024**2:.2f} MB")
            
        except Exception as e:
            logger.error(f"Error during data validation: {str(e)}")
            raise

    def save_results(self, output_path: str):
        """Save processed data and preprocessing artifacts."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save processed data
            if output_path.suffix == '.csv':
                self.data.to_csv(output_path, index=False)
            elif output_path.suffix == '.parquet':
                self.data.to_parquet(output_path, index=False)
            
            # Save preprocessing report
            report = {
                'data_profile': self.data_profile,
                'feature_importance': self.feature_importance,
                'transformers': {k: str(v) for k, v in self.transformers.items()},
                'column_types': {
                    'numeric': self.numeric_columns,
                    'categorical': self.categorical_columns,
                    'text': self.text_columns,
                    'date': self.date_columns,
                    'binary': self.binary_columns
                }
            }
            
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(report, f, indent=4, default=str)
                
            logger.info(f"Saved processed data and report to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    config = PreprocessingConfig(
        file_path="data/raw/sample_data.csv",
        target_column="target",
        create_interactions=True,
        perform_binning=True
    )
    
    agent = PreprocessingAgent(config)
    agent.run_pipeline()
    agent.save_results("data/processed/cleaned_data.csv")
