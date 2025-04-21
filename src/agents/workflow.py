import pandas as pd
import os
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, TypedDict, Optional, List
from llamaapi import LlamaAPI

from analysis_agent import AnalysisAgent
# Import our fixed visualization agent
from ai_visualization_agent import VisualizationAgent  
from data_cleaning_agent import DataCleaningAgent

# Enhanced state type that includes debug information
class AnalysisState(TypedDict):
    original_df: pd.DataFrame                  # Original dataframe
    cleaned_df: Optional[pd.DataFrame]         # Cleaned dataframe
    original_analysis_result: str              # Analysis of original data
    original_visualization_data: Optional[Dict[str, Any]]  # Visualizations of original data
    cleaning_results: Optional[Dict[str, Any]] # Results from cleaning process
    cleaned_visualization_data: Optional[Dict[str, Any]]  # Visualizations of cleaned data
    cleaned_analysis_result: str               # Analysis of cleaned data
    current_df: pd.DataFrame                   # Current working dataframe
    debug_logs: List[str]                      # Debug log messages

def create_analysis_workflow(df: pd.DataFrame, debug_mode=True, api_key=None):
    
    # Use provided API key instead of environment variable
    if not api_key:
        raise ValueError("No API Key provided. Please enter your LLAMA API Key.")
    
    llama = LlamaAPI(api_key)
    
    # Initialize all agents - use the enhanced visualization agent
    cleaning_agent = DataCleaningAgent()
    analysis_agent = AnalysisAgent()
    visualization_agent = VisualizationAgent(debug=debug_mode, api_key=api_key)
    
    # Helper function to add debug logs
    def add_debug_log(state, message):
        if debug_mode:
            if "debug_logs" not in state:
                state["debug_logs"] = []
            state["debug_logs"].append(message)
        return state
    
    # Define the initial analysis node that analyzes the original data
    def analyze_original_data_node(state: AnalysisState):
        """Node: Run AI-powered analysis on the original dataset."""
        state = add_debug_log(state, "Starting analysis of original data")
        
        # Get prompt from analysis agent
        prompt = analysis_agent.prepare_analysis_prompt(state["original_df"])
        
        # Prepare API request
        api_request = {
            "model": "llama3.1-70b",
            "messages": [
                {"role": "system", "content": "You are an expert data analyst. Provide clear, actionable insights based on the raw, uncleaned data."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }
        
        # Call API
        try:
            response = llama.run(api_request)
            if response.status_code == 200:
                state["original_analysis_result"] = response.json()["choices"][0]["message"]["content"]
                state = add_debug_log(state, "Successfully completed original data analysis")
            else:
                state["original_analysis_result"] = f"❌ API Error: {response.status_code} - {response.text}"
                state = add_debug_log(state, f"Error in original data analysis: API returned {response.status_code}")
        except Exception as e:
            state["original_analysis_result"] = f"❌ Initial analysis failed: {str(e)}"
            state = add_debug_log(state, f"Exception in original data analysis: {str(e)}")
            
        return state
    
    # Define the visualization node for original data
    def visualize_original_data_node(state: AnalysisState):
        """Node: Create AI-powered visualizations based on original dataset."""
        state = add_debug_log(state, "Starting visualization of original data")
        
        try:
            # Log some basic info about the dataframe
            state = add_debug_log(state, f"Original dataframe shape: {state['original_df'].shape}")
            
            # Call the enhanced visualization agent
            viz_data = visualization_agent.create_visualization(state["original_df"])
            state["original_visualization_data"] = viz_data
            
            # Log visualization results
            if viz_data.get("success", False):
                state = add_debug_log(state, f"Successfully created {len(viz_data.get('plots', []))} visualizations for original data")
                if viz_data.get("is_fallback", False):
                    state = add_debug_log(state, "Used fallback visualizations for original data")
            else:
                state = add_debug_log(state, "Failed to create visualizations for original data")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            state = add_debug_log(state, f"Exception in original data visualization: {str(e)}")
            state["original_visualization_data"] = {
                "success": False,
                "error": str(e),
                "error_details": error_details,
                "plots": []
            }
            
        return state
    
    # Define the cleaning node that processes the raw dataframe
    def clean_data_node(state: AnalysisState):
        """Node: Clean the dataset using AI-powered data cleaning agent."""
        state = add_debug_log(state, "Starting data cleaning")
        
        try:
            # Call the cleaning agent
            cleaned_df, cleaning_results = cleaning_agent.clean_data(state["original_df"])
            
            # Verify the cleaned dataframe is valid
            if cleaned_df is None or len(cleaned_df) == 0 or len(cleaned_df.columns) == 0:
                state = add_debug_log(state, "Cleaning returned empty dataframe, using original instead")
                cleaned_df = state["original_df"].copy()
                cleaning_results["warning"] = "Cleaning returned empty dataframe, using original instead"
            
            # Add data types information to debug log
            if debug_mode:
                num_cols = cleaned_df.select_dtypes(include=['number']).columns.tolist()
                cat_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns.tolist()
                state = add_debug_log(state, f"Cleaned data - numeric columns: {num_cols}")
                state = add_debug_log(state, f"Cleaned data - categorical columns: {cat_cols}")
            
            # Update state with cleaned data and results
            state["cleaned_df"] = cleaned_df
            state["current_df"] = cleaned_df  # Update current working dataframe
            state["cleaning_results"] = cleaning_results
            
            # Log cleaning summary
            state = add_debug_log(state, f"Data cleaning completed: {len(cleaning_results.get('cleaning_actions', []))} actions performed")
            state = add_debug_log(state, f"Original shape: {state['original_df'].shape}, Cleaned shape: {cleaned_df.shape}")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            state = add_debug_log(state, f"❌ Data cleaning failed: {str(e)}")
            state = add_debug_log(state, error_details)
            
            # If cleaning fails, use the original dataframe
            state["cleaned_df"] = state["original_df"].copy()
            state["current_df"] = state["original_df"].copy()
            state["cleaning_results"] = {
                "success": False,
                "error": str(e),
                "error_details": error_details,
                "cleaning_actions": []  # Ensure this is a list, not None
            }
            
        return state
    
    # Define the visualization node for cleaned data
    def visualize_cleaned_data_node(state: AnalysisState):
        """Node: Create AI-powered visualizations based on cleaned dataset."""
        state = add_debug_log(state, "Starting visualization of cleaned data")
        
        try:
            # Ensure cleaned_df exists and is not None
            if "cleaned_df" not in state or state["cleaned_df"] is None:
                state = add_debug_log(state, "Warning: cleaned_df not found in state, using original_df")
                state["cleaned_df"] = state["original_df"].copy()
            
            # Log detailed info about the cleaned dataframe
            state = add_debug_log(state, f"Cleaned dataframe shape: {state['cleaned_df'].shape}")
            
            # Log column types
            if debug_mode:
                col_types = {col: str(state["cleaned_df"][col].dtype) for col in state["cleaned_df"].columns[:5]}
                state = add_debug_log(state, f"Sample column types: {col_types}")
                
                # Check for null values
                null_counts = {col: int(state["cleaned_df"][col].isna().sum()) 
                              for col in state["cleaned_df"].columns 
                              if state["cleaned_df"][col].isna().sum() > 0}
                if null_counts:
                    state = add_debug_log(state, f"Columns with null values: {null_counts}")
            
            # Create a fresh copy of the dataframe to avoid reference issues
            clean_df_copy = state["cleaned_df"].copy()
            
            # Call the enhanced visualization agent with explicit debugging
            try:
                viz_data = visualization_agent.create_visualization(clean_df_copy)
                state = add_debug_log(state, "Visualization agent successfully called")
            except Exception as e:
                state = add_debug_log(state, f"Error calling visualization agent: {str(e)}")
                import traceback
                state = add_debug_log(state, traceback.format_exc())
                # Create a minimal viz_data in case of error
                viz_data = {
                    "success": False,
                    "error": str(e),
                    "plots": [],
                    "debug_info": {"exception": str(e)}
                }
            
            state["cleaned_visualization_data"] = viz_data
            
            # Log visualization results
            if viz_data.get("success", False):
                state = add_debug_log(state, f"Successfully created {len(viz_data.get('plots', []))} visualizations for cleaned data")
                if viz_data.get("is_fallback", False):
                    state = add_debug_log(state, "Used fallback visualizations for cleaned data")
            else:
                state = add_debug_log(state, "Failed to create visualizations for cleaned data")
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            state = add_debug_log(state, f"Exception in cleaned data visualization: {str(e)}")
            state = add_debug_log(state, error_details)
            state["cleaned_visualization_data"] = {
                "success": False,
                "error": str(e),
                "error_details": error_details,
                "plots": []
            }
            
        return state
    
    # Define the analysis node for cleaned data
    def analyze_cleaned_data_node(state: AnalysisState):
        """Node: Run AI-powered analysis on cleaned dataset."""
        state = add_debug_log(state, "Starting analysis of cleaned data")
        
        # Get prompt from analysis agent
        prompt = analysis_agent.prepare_analysis_prompt(state["cleaned_df"])
        
        # Add information about cleaning to the prompt
        cleaning_summary = ""
        if state.get("cleaning_results", {}).get("success", False):
            cleaning_actions = state["cleaning_results"].get("cleaning_actions", [])
            quality_issues = state["cleaning_results"].get("quality_issues", [])
            
            cleaning_summary = f"""
The data has been cleaned with the following actions:
- {len(quality_issues) if isinstance(quality_issues, list) else 0} quality issues were identified
- {len(cleaning_actions) if isinstance(cleaning_actions, list) else 0} cleaning actions were performed
"""
            
            # Make sure cleaning_actions is a list and contains dictionaries
            if isinstance(cleaning_actions, list) and len(cleaning_actions) > 0:
                cleaning_summary += "\nTop cleaning actions performed:\n"
                count = 0
                for action in cleaning_actions:
                    if isinstance(action, dict):
                        cleaning_summary += f"- {action.get('column', 'dataset')}: {action.get('description', 'N/A')}\n"
                        count += 1
                        if count >= 5:
                            break
                    elif isinstance(action, str):
                        # Handle case where actions might be strings
                        cleaning_summary += f"- {action}\n"
                        count += 1
                        if count >= 5:
                            break
            
            prompt = cleaning_summary + "\n\n" + prompt
        
        # Prepare API request
        api_request = {
            "model": "llama3.1-70b",
            "messages": [
                {"role": "system", "content": "You are an expert data analyst. Provide clear, actionable insights based on the cleaned data."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }
        
        # Call API
        try:
            response = llama.run(api_request)
            if response.status_code == 200:
                state["cleaned_analysis_result"] = response.json()["choices"][0]["message"]["content"]
                state = add_debug_log(state, "Successfully completed cleaned data analysis")
            else:
                state["cleaned_analysis_result"] = f"❌ API Error: {response.status_code} - {response.text}"
                state = add_debug_log(state, f"Error in cleaned data analysis: API returned {response.status_code}")
        except Exception as e:
            state["cleaned_analysis_result"] = f"❌ Cleaned data analysis failed: {str(e)}"
            state = add_debug_log(state, f"Exception in cleaned data analysis: {str(e)}")
            
        return state
    
    # Define the graph
    graph = StateGraph(AnalysisState)
    
    # Add nodes
    graph.add_node("analyze_original_data", analyze_original_data_node)
    graph.add_node("visualize_original_data", visualize_original_data_node)
    graph.add_node("clean_data", clean_data_node)
    graph.add_node("visualize_cleaned_data", visualize_cleaned_data_node)
    graph.add_node("analyze_cleaned_data", analyze_cleaned_data_node)
    
    # Define edges - flow according to specified sequence
    graph.add_edge(START, "analyze_original_data")
    graph.add_edge("analyze_original_data", "visualize_original_data")
    graph.add_edge("visualize_original_data", "clean_data")
    graph.add_edge("clean_data", "visualize_cleaned_data")
    graph.add_edge("visualize_cleaned_data", "analyze_cleaned_data")
    graph.add_edge("analyze_cleaned_data", END)
    
    # Compile the workflow
    workflow = graph.compile()
    
    def run_analysis():
        initial_state = {
            "original_df": df,                  # Store original dataframe
            "current_df": df.copy(),            # Working dataframe (will be updated)
            "cleaned_df": None,                 # Will store cleaned dataframe
            "original_analysis_result": "",     # Analysis of original data
            "original_visualization_data": None, # Visualizations of original data
            "cleaning_results": None,           # Results from cleaning
            "cleaned_visualization_data": None, # Visualizations of cleaned data
            "cleaned_analysis_result": "",      # Analysis of cleaned data
            "debug_logs": []                    # Debug logs
        }
        result = workflow.invoke(initial_state)
        
        # Return comprehensive results
        return {
            "original_analysis": {
                "result": result["original_analysis_result"],
                "visualizations": result["original_visualization_data"]
            },
            "cleaning": result["cleaning_results"],
            "cleaned_analysis": {
                "result": result["cleaned_analysis_result"],
                "visualizations": result["cleaned_visualization_data"]
            },
            "debug_logs": result.get("debug_logs", []),
            "cleaned_df": result.get("cleaned_df")
        }
    
    return run_analysis