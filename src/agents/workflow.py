import pandas as pd
import os
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, TypedDict, Optional
from llamaapi import LlamaAPI

from analysis_agent import AnalysisAgent
from visualization_agent import VisualizationAgent

# Define state type for the workflow
class AnalysisState(TypedDict):
    df: pd.DataFrame
    analysis_result: str
    visualization_data: Optional[Dict[str, Any]]

def create_analysis_workflow(df: pd.DataFrame):
    
    # Check for API key
    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key:
        raise ValueError("No API Key found. Please set LLAMA_API_KEY in .env file.")
    
    llama = LlamaAPI(api_key)
    
    analysis_agent = AnalysisAgent()
    visualization_agent = VisualizationAgent()
    
    # Define the analysis node that processes the dataframe and calls the API
    def analyze_node(state: AnalysisState):
        """Node: Run AI-powered analysis on dataset."""
        # Get prompt from analysis agent
        prompt = analysis_agent.prepare_analysis_prompt(state["df"])
        
        # Prepare API request
        api_request = {
            "model": "llama3.1-70b",
            "messages": [
                {"role": "system", "content": "You are an expert data analyst. Provide clear, actionable insights."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }
        
        # Call API
        try:
            response = llama.run(api_request)
            if response.status_code == 200:
                state["analysis_result"] = response.json()["choices"][0]["message"]["content"]
            else:
                state["analysis_result"] = f"❌ API Error: {response.status_code} - {response.text}"
        except Exception as e:
            state["analysis_result"] = f"❌ Analysis failed: {str(e)}"
            
        return state
    
    # Define the visualization node
    def visualize_node(state: AnalysisState):
        """Node: Create visualizations based on dataset."""
        state["visualization_data"] = visualization_agent.create_visualization(state["df"])
        return state
    
    # Define the graph
    graph = StateGraph(AnalysisState)
    
    # Add nodes
    graph.add_node("analyze_data", analyze_node)
    graph.add_node("create_visualizations", visualize_node)
    
    # Define edges - sequential flow to avoid concurrent updates
    graph.add_edge(START, "analyze_data")
    graph.add_edge("analyze_data", "create_visualizations")
    graph.add_edge("create_visualizations", END)
    
    # Compile the workflow
    workflow = graph.compile()
    
    def run_analysis():
        initial_state = {
            "df": df, 
            "analysis_result": "",
            "visualization_data": None
        }
        result = workflow.invoke(initial_state)
        return {
            "analysis_result": result["analysis_result"],
            "visualization_data": result["visualization_data"]
        }
    
    return run_analysis
