import pandas as pd
from typing import Dict, Any
from langchain.pydantic_v1 import BaseModel, Field

class EDAAgent(BaseModel):
    file_path: str = Field(description="Path to the data file")
    df: Any = Field(default=None, description="Pandas DataFrame")
    numerical_columns: list = Field(default_factory=list, description="List of numerical columns")
    categorical_columns: list = Field(default_factory=list, description="List of categorical columns")

    def load_data(self) -> Dict[str, Any]:
        try:
            if self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            elif self.file_path.endswith('.xlsx'):
                self.df = pd.read_excel(self.file_path)
            elif self.file_path.endswith('.json'):
                self.df = pd.read_json(self.file_path)
            else:
                raise ValueError("Unsupported file format")

            self.numerical_columns = self.df.select_dtypes(include=["number"]).columns.tolist()
            self.categorical_columns = self.df.select_dtypes(include=["object"]).columns.tolist()

            return {"status": "success", "message": "Data loaded successfully"}
        except Exception as e:
            return {"status": "error", "message": f"Error loading file: {str(e)}"}

    def get_data_description(self) -> str:
        numerical_stats = self.df[self.numerical_columns].describe().to_dict() if self.numerical_columns else {}
        categorical_stats = {col: self.df[col].value_counts().to_dict() for col in self.categorical_columns}
        
        description = f"""
Dataset Overview:
- Total Records: {len(self.df)}
- Total Features: {len(self.df.columns)}
- Features: {', '.join(self.df.columns)}

Numerical Features:
{self._format_numerical_stats(numerical_stats)}

Categorical Features:
{self._format_categorical_stats(categorical_stats)}

Sample Data (First 5 rows):
{self._format_sample_data(self.df.head())}
"""
        return description

    def _format_numerical_stats(self, stats: Dict) -> str:
        formatted = []
        for col, stat in stats.items():
            formatted.append(f"\n{col}:")
            for metric, value in stat.items():
                formatted.append(f"  - {metric}: {value:.2f}")
        return "\n".join(formatted)

    def _format_categorical_stats(self, stats: Dict) -> str:
        formatted = []
        for col, values in stats.items():
            formatted.append(f"\n{col}:")
            for category, count in values.items():
                formatted.append(f"  - {category}: {count}")
        return "\n".join(formatted)

    def _format_sample_data(self, sample: pd.DataFrame) -> str:
        return sample.to_string()
