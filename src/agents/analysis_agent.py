import pandas as pd
from typing import Dict, Any

class AnalysisAgent:
    
    def prepare_analysis_prompt(self, df: pd.DataFrame) -> str:
        prompt = f"""As a data analysis expert, analyze the following dataset:

Dataset Overview:
- Total Records: {len(df)}
- Total Features: {len(df.columns)}
- Features: {', '.join(df.columns)}

Numerical Summary:
{df.describe().to_dict()}

Categorical Summary:
{ {col: df[col].value_counts().to_dict() for col in df.select_dtypes(include=['object']).columns} }

Provide a comprehensive analysis, including:
1. Key patterns and trends in the data
2. Statistical insights from numerical features
3. Distribution patterns in categorical features
4. Notable relationships between features
5. Data quality observations
6. Suggestions for further analysis

Structure your response clearly and focus on actionable insights."""

        return prompt

