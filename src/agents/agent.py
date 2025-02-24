# agent.py
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any
from llamaapi import LlamaAPI

# Load environment variables
load_dotenv(dotenv_path="C:\\Users\\swath\\GWU-Workspace\\CAPSTONE\\IntelliWiz\\src\\agents\\.env")

# ✅ Define structured input schema for AI tool
class DataInput(BaseModel):
    columns: List[str] = Field(..., description="List of column names in the dataset")
    rows: List[Dict[str, Any]] = Field(..., description="Dataset records as dictionaries")

@tool
def analyze_data(data: DataInput) -> str:
    """AI-powered tool to analyze a dataset and generate insights."""
    if isinstance(data, dict):
        data = DataInput(**data)  # Deserialize dictionary into DataInput model
    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key:
        return "❌ No API Key found. Please set LLAMA_API_KEY in .env file."

    llama = LlamaAPI(api_key)
    df = pd.DataFrame(data.rows, columns=data.columns)  # Convert dict back to DataFrame

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

    api_request = {
        "model": "llama3.1-70b",
        "messages": [
            {"role": "system", "content": "You are an expert data analyst. Provide clear, actionable insights."},
            {"role": "user", "content": prompt}
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

# # 🌟 LangGraph Workflow
# def create_analysis_workflow(df):
#     # ✅ Convert DataFrame to Pydantic-compatible input
#     df_dict = DataInput(columns=df.columns.tolist(), rows=df.to_dict(orient="records"))

#     # Define the node function for analysis
#     def analyze_node(state: dict):
#         """Node: Run AI-powered analysis on dataset."""
#         state["analysis_result"] = analyze_data(df_dict)
#         return state

#     # 🌐 Define LangGraph workflow
#     graph = StateGraph(dict)

#     # Add analysis node
#     graph.add_node("analyze_data", analyze_node)

#     # Set edges (Start → Analyze → End)
#     graph.add_edge(START, "analyze_data")
#     graph.add_edge("analyze_data", END)

#     # Compile into a runnable agent
#     return graph.compile()
from pydantic import ValidationError

def create_analysis_workflow(df):
    try:
        # ✅ Convert DataFrame to a valid DataInput instance
        df_input = DataInput(
            columns=df.columns.tolist(),
            rows=df.to_dict(orient="records")
        )
        
        # Serialize DataInput to a dictionary
        serialized_input = df_input.model_dump()  # Use .dict() if using Pydantic <2.0

    except ValidationError as e:
        raise ValueError(f"Invalid input for DataInput: {e}")

    # Define the node function for analysis
    def analyze_node(state: dict):
        """Node: Run AI-powered analysis on dataset."""
        try:
            print("Serialized Input for analyze_data:", {"data": serialized_input})
            # Pass serialized input (dictionary) to analyze_data
            state["analysis_result"] = analyze_data({"data": serialized_input})
        except ValidationError as e:
            raise ValueError(f"Validation error in analyze_data: {e}")
        return state

    # 🌐 Define LangGraph workflow
    graph = StateGraph(dict)
    graph.add_node("analyze_data", analyze_node)
    graph.add_edge(START, "analyze_data")
    graph.add_edge("analyze_data", END)

    # Compile into a runnable agent
    return graph.compile()
