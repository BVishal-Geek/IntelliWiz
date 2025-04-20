import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io
from typing import Dict, Any, List, Optional
import os
from llamaapi import LlamaAPI
import traceback

class VisualizationAgent:
    """
    Enhanced AI-powered visualization agent with improved error handling, 
    debugging, and fallback visualization capabilities.
    """
    
    def __init__(self, debug=False):
        """Initialize the visualization agent with API client"""
        self.api_key = os.getenv("LLAMA_API_KEY")
        self.debug = debug
        if not self.api_key:
            raise ValueError("No API Key found. Please set LLAMA_API_KEY in .env file.")
        
        self.llama = LlamaAPI(self.api_key)
    
    def create_visualization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create visualizations based on dataset analysis with enhanced error handling.
        """
        viz_data = {"success": False, "plots": [], "debug_info": {}}
        
        # Early validation - if dataframe is empty or None, return with error
        if df is None or len(df) == 0 or len(df.columns) == 0:
            viz_data["debug_info"]["error"] = "Empty or invalid dataframe provided"
            return viz_data
            
        try:
            # Add debug info about the dataframe
            if self.debug:
                viz_data["debug_info"]["df_shape"] = df.shape
                viz_data["debug_info"]["df_columns"] = df.columns.tolist()
                viz_data["debug_info"]["df_dtypes"] = {col: str(df[col].dtype) for col in df.columns}
                viz_data["debug_info"]["df_null_counts"] = {col: int(df[col].isna().sum()) for col in df.columns}
            
            # Convert the entire dataframe to CSV string for the prompt
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Provide basic dataset shape info
            dataset_info = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": {col: str(df[col].dtype) for col in df.columns}
            }
            
            # Create a comprehensive prompt for the LLM
            prompt = f"""As a data visualization expert, I need your help to create effective visualizations for my dataset.

Here is information about the dataset:
```json
{json.dumps(dataset_info, indent=2)}
```

And here is the complete dataset:
```csv
{csv_string}
```

Please analyze this data and perform the following tasks:

1. Analyze the dataset characteristics:
   - Identify the data types of each column
   - Detect numerical, categorical, and date/time columns
   - Identify potential relationships between variables
   - Calculate basic statistics for numerical columns
   - Identify columns with missing values
   - Check for high cardinality in categorical columns

2. Based on your analysis, recommend the 3-5 most informative visualizations that would reveal important patterns, trends, or relationships.

3. For each recommended visualization, provide:
   a) The exact Plotly Express or Plotly Graph Objects code to create the visualization
   b) A descriptive title
   c) A brief explanation of what insights this visualization might reveal
   d) Any data preprocessing steps needed before creating the visualization

IMPORTANT: Make sure all code is valid Python using Plotly, and assume the dataframe is available as 'df'. 
Ensure preprocessing handles any null values or data type issues.
Return your response in the following JSON format:
```json
{{
  "data_analysis": {{
    "column_types": {{}},
    "statistics": {{}},
    "relationships": [],
    "missing_values": [],
    "high_cardinality_columns": []
  }},
  "visualizations": [
    {{
      "title": "Distribution of X",
      "insight": "This shows...",
      "preprocessing": "df['column'] = df['column'].fillna('Unknown')",
      "plot_code": "px.histogram(df, x='column', title='Distribution of X')",
      "plot_type": "histogram"
    }},
    ...
  ]
}}
```
"""
            
            # Prepare API request
            api_request = {
                "model": "llama3.1-70b",
                "messages": [
                    {"role": "system", "content": "You are an expert data scientist and visualization specialist who excels at Plotly visualizations and data analysis. You return only valid JSON with working Python code that handles null values and data type issues."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            }
            
            # Call the LLM API
            response = self.llama.run(api_request)
            
            if response.status_code != 200:
                viz_data["debug_info"]["api_error"] = f"API Error: {response.status_code} - {response.text}"
                # Call fallback visualization if LLM fails
                return self._create_fallback_visualizations(df, viz_data)
            
            # Parse the LLM response
            result = response.json()["choices"][0]["message"]["content"]
            llm_response = json.loads(result)
            
            if self.debug:
                viz_data["debug_info"]["llm_response"] = llm_response
            
            # Extract data analysis and visualization recommendations
            data_analysis = llm_response.get("data_analysis", {})
            vis_recommendations = llm_response.get("visualizations", [])
            
            if not vis_recommendations:
                viz_data["debug_info"]["error"] = "No visualization recommendations from LLM"
                return self._create_fallback_visualizations(df, viz_data)
            
            # Generate the visualizations based on LLM recommendations
            plots = []
            for idx, viz_rec in enumerate(vis_recommendations):
                try:
                    # Apply any preprocessing steps recommended by the LLM
                    processed_df = df.copy()
                    preprocessing_code = viz_rec.get("preprocessing")
                    if preprocessing_code:
                        # Use exec to run the preprocessing code on the dataframe
                        # This is safe since the code comes from our LLM, not external users
                        local_vars = {"df": processed_df, "pd": pd}
                        exec(preprocessing_code, globals(), local_vars)
                        processed_df = local_vars["df"]
                    
                    # Execute the plot code provided by the LLM
                    plot_code = viz_rec.get("plot_code")
                    if not plot_code:
                        continue
                        
                    # Create a local context with the processed dataframe and plotting libraries
                    local_vars = {
                        "df": processed_df, 
                        "px": px, 
                        "go": go, 
                        "pd": pd
                    }
                    
                    # Execute the plot code and get the figure
                    exec(f"fig = {plot_code}", globals(), local_vars)
                    fig = local_vars["fig"]
                    
                    # Set common layout properties
                    fig.update_layout(
                        template='plotly_white',
                        height=500,
                        margin=dict(t=80, b=50, l=50, r=50),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        title=dict(
                            text=viz_rec.get("title", f"Visualization {idx+1}"),
                            font=dict(size=16)
                        )
                    )
                    
                    # Verify the figure can be converted to JSON and back
                    fig_json = fig.to_json()
                    # Test parsing it back to ensure it's valid
                    _ = json.loads(fig_json)
                    
                    # Add plot data to result
                    plot_data = {
                        "id": f"plot_{idx+1}",
                        "figure": fig_json,
                        "title": viz_rec.get("title", f"Visualization {idx+1}"),
                        "insight": viz_rec.get("insight", ""),
                        "plot_type": viz_rec.get("plot_type", "custom")
                    }
                    
                    plots.append(plot_data)
                    
                except Exception as e:
                    error_trace = traceback.format_exc()
                    error_info = {
                        "viz_index": idx,
                        "error": str(e),
                        "traceback": error_trace
                    }
                    if self.debug:
                        if "visualization_errors" not in viz_data["debug_info"]:
                            viz_data["debug_info"]["visualization_errors"] = []
                        viz_data["debug_info"]["visualization_errors"].append(error_info)
                    continue
            
            viz_data["plots"] = plots
            viz_data["success"] = len(plots) > 0
            viz_data["analysis"] = data_analysis
            
            # If no plots were successfully created, use fallback
            if not viz_data["success"]:
                return self._create_fallback_visualizations(df, viz_data)
                
            return viz_data
            
        except Exception as e:
            error_trace = traceback.format_exc()
            viz_data["debug_info"]["error"] = str(e)
            viz_data["debug_info"]["traceback"] = error_trace
            
            # Use fallback visualizations if the primary method fails
            return self._create_fallback_visualizations(df, viz_data)
    
    def _create_fallback_visualizations(self, df: pd.DataFrame, viz_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create basic fallback visualizations when the LLM-based approach fails.
        This ensures that we always return some visualizations.
        """
        try:
            plots = []
            
            # Get numerical and categorical columns
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # 1. Basic histogram or bar chart for the first numeric column
            if num_cols:
                try:
                    col = num_cols[0]
                    fig = px.histogram(
                        df, 
                        x=col, 
                        title=f"Distribution of {col}",
                        labels={col: col.replace('_', ' ').title()},
                        template='plotly_white'
                    )
                    fig.update_layout(height=500)
                    
                    plot_data = {
                        "id": "plot_fallback_1",
                        "figure": fig.to_json(),
                        "title": f"Distribution of {col}",
                        "insight": f"This histogram shows the distribution of values for {col}.",
                        "plot_type": "histogram"
                    }
                    plots.append(plot_data)
                except Exception as e:
                    if self.debug:
                        viz_data["debug_info"]["fallback_errors"] = viz_data.get("fallback_errors", []) + [str(e)]
            
            # 2. Bar chart for the first categorical column
            if cat_cols:
                try:
                    col = cat_cols[0]
                    top_values = df[col].value_counts().head(10)
                    fig = px.bar(
                        x=top_values.index, 
                        y=top_values.values,
                        title=f"Top values for {col}",
                        labels={'x': col.replace('_', ' ').title(), 'y': 'Count'},
                        template='plotly_white'
                    )
                    fig.update_layout(height=500)
                    
                    plot_data = {
                        "id": "plot_fallback_2",
                        "figure": fig.to_json(),
                        "title": f"Top values for {col}",
                        "insight": f"This bar chart shows the most common values for {col}.",
                        "plot_type": "bar"
                    }
                    plots.append(plot_data)
                except Exception as e:
                    if self.debug:
                        viz_data["debug_info"]["fallback_errors"] = viz_data.get("fallback_errors", []) + [str(e)]
            
            # 3. Scatter plot if we have at least 2 numeric columns
            if len(num_cols) >= 2:
                try:
                    x_col, y_col = num_cols[0], num_cols[1]
                    color_col = cat_cols[0] if cat_cols and df[cat_cols[0]].nunique() <= 10 else None
                    
                    if color_col:
                        fig = px.scatter(
                            df, 
                            x=x_col, 
                            y=y_col, 
                            color=color_col,
                            title=f"{y_col} vs {x_col}",
                            labels={
                                x_col: x_col.replace('_', ' ').title(),
                                y_col: y_col.replace('_', ' ').title(),
                                color_col: color_col.replace('_', ' ').title()
                            },
                            template='plotly_white'
                        )
                    else:
                        fig = px.scatter(
                            df, 
                            x=x_col, 
                            y=y_col,
                            title=f"{y_col} vs {x_col}",
                            labels={
                                x_col: x_col.replace('_', ' ').title(),
                                y_col: y_col.replace('_', ' ').title()
                            },
                            template='plotly_white'
                        )
                        
                    fig.update_layout(height=500)
                    
                    plot_data = {
                        "id": "plot_fallback_3",
                        "figure": fig.to_json(),
                        "title": f"Relationship between {y_col} and {x_col}",
                        "insight": f"This scatter plot shows the relationship between {y_col} and {x_col}.",
                        "plot_type": "scatter"
                    }
                    plots.append(plot_data)
                except Exception as e:
                    if self.debug:
                        viz_data["debug_info"]["fallback_errors"] = viz_data.get("fallback_errors", []) + [str(e)]
            
            # Add plots to the viz_data
            viz_data["plots"] = plots
            viz_data["success"] = len(plots) > 0
            viz_data["is_fallback"] = True
            
            return viz_data
            
        except Exception as e:
            # Ultimate fallback - if even the basic fallback fails
            viz_data["debug_info"]["ultimate_fallback_error"] = str(e)
            viz_data["success"] = False
            viz_data["plots"] = []
            viz_data["is_fallback"] = True
            return viz_data