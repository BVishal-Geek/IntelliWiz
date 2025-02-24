# eda_agent.py
import pandas as pd
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from llamaapi import LlamaAPI

# ✅ Define a structured data schema using Pydantic
class DataDescriptionInput(BaseModel):
    columns: List[str] = Field(..., description="List of column names in the dataset")
    rows: List[Dict[str, Any]] = Field(..., description="List of rows, each represented as a dictionary")

@tool
def get_data_description(data: DataDescriptionInput) -> str:
    """Tool to generate an overview of the dataset including numerical and categorical statistics."""
    df = pd.DataFrame(data.rows, columns=data.columns)  # Convert dict back to DataFrame

    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    numerical_stats = df[numerical_columns].describe().to_dict() if numerical_columns else {}
    categorical_stats = {col: df[col].value_counts().to_dict() for col in categorical_columns}

    description = f"""
Dataset Overview:
- Total Records: {len(df)}
- Total Features: {len(df.columns)}
- Features: {', '.join(df.columns)}

Numerical Features:
{numerical_stats}

Categorical Features:
{categorical_stats}

Sample Data (First 5 rows):
{df.head().to_string()}
"""
    return description

@tool
def analyze_data(data_description: str, api_key: str) -> str:
    """Tool to analyze a dataset overview and generate insights using Llama API."""
    llama = LlamaAPI(api_key)

    prompt = f"""As a data analysis expert, analyze this dataset and provide insights:

{data_description}

Please provide a comprehensive analysis including:
1. Key patterns and trends in the data
2. Statistical insights from numerical features
3. Distribution patterns in categorical features
4. Notable relationships between features
5. Data quality observations
6. Suggestions for further analysis

Structure your response clearly and focus on actionable insights.
"""

    api_request = {
        "model": "llama3.1-70b",
        "messages": [
            {
                "role": "system",
                "content": "You are a data analysis expert. Provide clear, actionable insights from data analysis."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }

    try:
        response = llama.run(api_request)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Analysis failed: {str(e)}"
