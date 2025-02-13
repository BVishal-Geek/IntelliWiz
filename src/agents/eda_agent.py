import pandas as pd

class EDAAgent:
    def __init__(self, file):
        try:
            self.df = self.load_data(file)
            self.numerical_columns = self.df.select_dtypes(include=["number"]).columns.tolist()
            self.categorical_columns = self.df.select_dtypes(include=["object"]).columns.tolist()
        except Exception as e:
            raise Exception(f"Failed to initialize EDA Agent: {str(e)}")

    def load_data(self, file):
        try:
            if hasattr(file, "name"):
                file_name = file.name
                if file_name.endswith('.csv'):
                    return pd.read_csv(file)
                elif file_name.endswith('.xlsx'):
                    return pd.read_excel(file)
                elif file_name.endswith('.json'):
                    return pd.read_json(file)
                else:
                    raise ValueError("Unsupported file format")
            else:
                raise ValueError("Invalid file input")
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")

    def get_data_description(self):
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

    def _format_numerical_stats(self, stats):
        formatted = []
        for col, stat in stats.items():
            formatted.append(f"\n{col}:")
            for metric, value in stat.items():
                formatted.append(f"  - {metric}: {value:.2f}")
        return "\n".join(formatted)

    def _format_categorical_stats(self, stats):
        formatted = []
        for col, values in stats.items():
            formatted.append(f"\n{col}:")
            for category, count in values.items():
                formatted.append(f"  - {category}: {count}")
        return "\n".join(formatted)

    def _format_sample_data(self, sample):
        return sample.to_string()