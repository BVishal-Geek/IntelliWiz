import pandas as pd
import os
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from typing import Dict, Any, TypedDict, Optional
from llamaapi import LlamaAPI
import json
import ast

from analysis_agent import AnalysisAgent
from visualization_tools import VisualizationTools
from viz_agent import VizAgent


# Define state type for the workflow
class AnalysisState(TypedDict):
    df: pd.DataFrame
    analysis_result: str
    visualization_data: Optional[Dict[str, Any]]
    messages: list

def create_analysis_workflow(df: pd.DataFrame):
    
    # Check for API key
    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key:
        raise ValueError("No API Key found. Please set LLAMA_API_KEY in .env file.")
    
    llama = LlamaAPI(api_key)
    
    analysis_agent = AnalysisAgent()
    
    
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
    
    # Define the suggest plots node
    def suggest_plots_node(state: AnalysisState):
        """Node: Get LLM-suggested visualizations."""
        
        visualization_agent = VizAgent()
        prompt = visualization_agent.suggest_visual_columns(state["df"])
        api_request = {
            "model": "llama3.1-70b",
            "messages": [
                {"role": "system", "content": "You are an expert data analyst. Provide clear, actionable insights."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 1500
        }

        try:
            response = llama.run(api_request)
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                state["visualization_data"] = {
                    "success": True,
                    "llm_visual_prompt_output": content
                }
                state["messages"] = [
                    HumanMessage(content=prompt),
                    AIMessage(content=content)
                ]
            else:
                state["visualization_data"] = {
                    "success": False,
                    "message": f"❌ API Error: {response.status_code} - {response.text}"
                }
        except Exception as e:
            state["visualization_data"] = {
                "success": False,
                "message": f"❌ Visualization generation failed: {str(e)}"
            }

        return state
    
    Visualization_Tools = VisualizationTools(df)

    # Define the graph
    graph = StateGraph(AnalysisState)
    
    # Add nodes
    graph.add_node("analyze_data", analyze_node)
    graph.add_node("suggest_plots", suggest_plots_node)
    graph.add_node("visualization_tool", ToolNode(Visualization_Tools.get_tools()))
    
    # Define edges - sequential flow to avoid concurrent updates
    graph.add_edge(START, "analyze_data")
    graph.add_edge("analyze_data", "suggest_plots")
    graph.add_edge("suggest_plots", "visualization_tool")
    graph.add_edge("visualization_tool", END)
    
    # Compile the workflow
    workflow = graph.compile()
    
    def run_analysis():
        initial_state = {
            "df": df, 
            "analysis_result": "",
            "visualization_data": None,
            "messages": []
        }
        result = workflow.invoke(initial_state)
        return {
            "analysis_result": result["analysis_result"],
            "visualization_data": result["visualization_data"]
        }
    
    return run_analysis
