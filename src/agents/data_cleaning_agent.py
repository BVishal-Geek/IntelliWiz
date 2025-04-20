import pandas as pd
import numpy as np
import json
import io
from typing import Dict, Any, List, Optional, Tuple, Union
import os
from llamaapi import LlamaAPI

class DataCleaningAgent:
    """
    AI-powered data cleaning agent that automatically handles data preprocessing
    including missing values, outliers, type conversion, and data integrity issues.
    """
    
    def __init__(self):
        """Initialize the cleaning agent with API client"""
        self.api_key = os.getenv("LLAMA_API_KEY")
        if not self.api_key:
            raise ValueError("No API Key found. Please set LLAMA_API_KEY in .env file.")
        
        self.llama = LlamaAPI(self.api_key)
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main method to clean a dataset:
        1. Profile the data to understand its characteristics
        2. Identify data quality issues
        3. Clean the data by fixing identified issues
        4. Return cleaned data and cleaning report
        
        Args:
            df: Input pandas DataFrame
            
        Returns:
            Tuple containing:
                - Cleaned pandas DataFrame
                - Dictionary with cleaning report details
        """
        result = {
            "success": False,
            "original_data": {
                "shape": df.shape,
                "columns": df.columns.tolist()
            },
            "data_profile": None,
            "quality_issues": None,
            "cleaning_actions": None,
            "warning_flags": [],
        }
        
        try:
            # Step 1: Generate data profile
            print("Generating data profile...")
            result["data_profile"] = self._generate_data_profile(df)
            
            # Step 2: Clean the data based on profile
            print("Cleaning data...")
            cleaned_df, cleaning_result = self._apply_cleaning(df, result["data_profile"])
            
            # Step 3: Record cleaning actions and quality issues
            result["quality_issues"] = cleaning_result.get("quality_issues", [])
            result["cleaning_actions"] = cleaning_result.get("actions", [])
            
            # Step 4: Validate cleaned data
            print("Validating cleaned data...")
            is_valid, validation_issues = self._validate_cleaned_data(df, cleaned_df)
            if not is_valid:
                result["warning_flags"].extend(validation_issues)
            
            # Step 5: Set success flag and add cleaned data stats
            result["success"] = True
            result["cleaned_data"] = {
                "shape": cleaned_df.shape,
                "columns": cleaned_df.columns.tolist()
            }
            
            return cleaned_df, result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            result["success"] = False
            result["error"] = str(e)
            result["error_details"] = error_details
            print(f"Error during data cleaning: {str(e)}")
            return df, result  # Return original data in case of error
    
    def _generate_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data profile to identify data types,
        distributions, missing values, and potential quality issues.
        
        Args:
            df: Input pandas DataFrame
            
        Returns:
            Dictionary containing data profile information
        """
        try:
            # Convert to CSV for the prompt
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Basic dataset info
            dataset_info = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": {col: str(df[col].dtype) for col in df.columns}
            }
            
            # Prompt for data profiling
            prompt = f"""As a data quality expert, analyze this dataset and provide a comprehensive data profile.

Dataset Information:
```json
{json.dumps(dataset_info, indent=2)}
```

Dataset Content:
```csv
{csv_string}
```

Please provide a detailed profile of this dataset covering:

1. For each column:
   - Verify the correct data type
   - Calculate percentage of missing values
   - Identify potential invalid values or outliers
   - For numerical columns: analyze range, distribution, outliers
   - For categorical columns: identify cardinality, frequency distribution, invalid categories
   - For date columns: identify format issues, invalid dates, temporal range

2. Data quality issues to look for:
   - Inconsistent formatting (e.g., different date formats)
   - Typos in categorical data
   - Duplicate records
   - Values outside valid ranges (negative values for naturally positive quantities)
   - Impossible values (future dates in historical data)
   - Inconsistent units or scales
   - Unexpected relationships (e.g., end dates before start dates)

Return your analysis in this JSON format:
```json
{{
  "column_profiles": [
    {{
      "name": "column_name",
      "inferred_type": "current data type",
      "recommended_type": "recommended data type",
      "missing_count": 0,
      "missing_percentage": 0.0,
      "unique_count": 0,
      "unique_percentage": 0.0,
      "stats": {{}},
      "quality_issues": [
        {{"issue_type": "issue type", "description": "issue description", "severity": "high/medium/low"}}
      ],
      "cleaning_recommendations": [
        {{"action": "action type", "reason": "reason for action", "priority": "high/medium/low"}}
      ]
    }}
  ],
  "dataset_issues": [
    {{"issue_type": "issue type", "description": "issue description", "severity": "high/medium/low"}}
  ],
  "recommendations": [
    {{"action": "recommended action", "reason": "reason for recommendation", "priority": "high/medium/low"}}
  ]
}}
```

For numerical columns, include min, max, mean, median, std in the stats object.
For categorical columns, include top 5 categories and their frequencies.
Provide specific, actionable cleaning recommendations.
"""
            
            # API request for data profiling
            api_request = {
                "model": "llama3.1-70b",
                "messages": [
                    {"role": "system", "content": "You are an expert data scientist specializing in data quality assessment and data cleaning. You provide detailed, accurate analysis of datasets with specific, actionable recommendations."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            
            # Call the LLM API
            response = self.llama.run(api_request)
            
            if response.status_code != 200:
                return {
                    "error": f"API Error: {response.status_code} - {response.text}",
                    "column_profiles": [],
                    "dataset_issues": [],
                    "recommendations": []
                }
            
            # Parse the LLM response
            result = response.json()["choices"][0]["message"]["content"]
            profile_result = json.loads(result)
            
            return profile_result
            
        except Exception as e:
            print(f"Error in data profiling: {str(e)}")
            return {
                "error": str(e),
                "column_profiles": [],
                "dataset_issues": [],
                "recommendations": []
            }
    
    def _apply_cleaning(self, df: pd.DataFrame, profile: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply cleaning operations based on the data profile.
        
        Args:
            df: Input pandas DataFrame
            profile: Data profile dictionary from _generate_data_profile
            
        Returns:
            Tuple containing:
                - Cleaned pandas DataFrame
                - Dictionary with cleaning results
        """
        try:
            # Convert to CSV for the prompt
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Create a prompt for data cleaning
            prompt = f"""As a data cleaning expert, I need your help to clean this dataset based on the data profile.

Data Profile:
```json
{json.dumps(profile, indent=2)}
```

Dataset:
```csv
{csv_string}
```

Please provide Python code to clean this dataset addressing all the quality issues identified. 
The code should tackle these common cleaning tasks:

1. Data type corrections:
   - Convert columns to appropriate types
   - Parse dates consistently
   - Fix numeric fields stored as text

2. Missing value handling:
   - Impute missing values using appropriate methods for each column
   - Flag or drop rows with critical missing values
   - Use domain-specific logic when possible

3. Outlier treatment:
   - Identify and handle outliers using appropriate methods
   - Distinguish between errors and valid extreme values

4. Standardization:
   - Standardize text fields (case, leading/trailing spaces)
   - Normalize categorical values
   - Fix inconsistent formatting

5. Deduplication:
   - Identify and handle duplicate records

6. Value validation:
   - Fix values outside valid ranges
   - Correct impossible values
   - Address inconsistencies between related fields

Return your response in the following JSON format:
```json
{{
  "quality_issues": [
    {{"column": "column_name", "issue_type": "issue type", "description": "detailed description", "severity": "high/medium/low"}}
  ],
  "actions": [
    {{"column": "column_name", "action_type": "action type", "description": "description of cleaning action", "justification": "why this action was taken"}}
  ],
  "cleaning_code": "# Complete Python code to clean the dataset\\n..."
}}
```

The cleaning_code should be comprehensive Python code that takes a dataframe 'df' as input and returns a cleaned dataframe.
Include detailed comments explaining each cleaning step.
Your code should be robust and handle potential errors gracefully.
"""
            
            # API request for data cleaning
            api_request = {
                "model": "llama3.1-70b",
                "messages": [
                    {"role": "system", "content": "You are an expert data scientist specializing in data cleaning and preprocessing. You provide executable Python code that follows best practices for data cleaning."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            }
            
            # Call the LLM API
            response = self.llama.run(api_request)
            
            if response.status_code != 200:
                return df, {
                    "error": f"API Error: {response.status_code} - {response.text}",
                    "quality_issues": [],
                    "actions": []
                }
            
            # Parse the LLM response
            result = response.json()["choices"][0]["message"]["content"]
            cleaning_result = json.loads(result)
            
            # Execute the cleaning code
            cleaning_code = cleaning_result.get("cleaning_code", "")
            if cleaning_code:
                # Create a local context with the dataframe
                local_vars = {"df": df.copy(), "pd": pd, "np": np}
                
                # Add try/except block to the code to safely handle errors
                safe_code = f"""
try:
    # Original cleaning code
{cleaning_code}
    # Ensure the result is assigned to 'cleaned_df'
    if 'cleaned_df' not in locals():
        cleaned_df = df
except Exception as e:
    import traceback
    error_details = traceback.format_exc()
    print(f"Error in cleaning code: {{str(e)}}")
    print(error_details)
    cleaned_df = df
"""
                
                # Execute the cleaning code
                exec(safe_code, globals(), local_vars)
                
                # Get the cleaned dataframe
                cleaned_df = local_vars.get("cleaned_df", df)
            else:
                cleaned_df = df
            
            return cleaned_df, cleaning_result
            
        except Exception as e:
            print(f"Error in data cleaning: {str(e)}")
            return df, {
                "error": str(e),
                "quality_issues": [],
                "actions": []
            }
    
    def _validate_cleaned_data(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Tuple[bool, List[Dict[str, str]]]:
        """
        Validate the cleaned data to ensure the cleaning process was successful
        and didn't introduce new issues.
        
        Args:
            original_df: Original pandas DataFrame
            cleaned_df: Cleaned pandas DataFrame
            
        Returns:
            Tuple containing:
                - Boolean indicating validation success
                - List of validation issues found
        """
        validation_issues = []
        
        # Check if any columns were accidentally dropped
        missing_columns = set(original_df.columns) - set(cleaned_df.columns)
        actions = getattr(cleaned_df, "get", lambda x, y: [])("actions", [])
        if missing_columns and not any("drop" in action.get("action_type", "").lower() for action in actions):
            validation_issues.append({
                "issue_type": "missing_columns",
                "description": f"Columns were dropped without explicit drop action: {', '.join(missing_columns)}",
                "severity": "high"
            })
        
        # Check if row count decreased significantly (>10% without explanation)
        row_decrease_pct = (len(original_df) - len(cleaned_df)) / len(original_df) * 100 if len(original_df) > 0 else 0
        if row_decrease_pct > 10:
            validation_issues.append({
                "issue_type": "significant_row_decrease",
                "description": f"Row count decreased by {row_decrease_pct:.1f}% after cleaning",
                "severity": "medium"
            })
        
        # Check for introduction of new missing values in previously complete columns
        for col in set(original_df.columns) & set(cleaned_df.columns):
            if original_df[col].notna().all() and not cleaned_df[col].notna().all():
                validation_issues.append({
                    "issue_type": "new_missing_values",
                    "description": f"New missing values introduced in column '{col}'",
                    "severity": "medium"
                })
        
        # Check for data type issues (converting numeric to string, etc.)
        for col in set(original_df.columns) & set(cleaned_df.columns):
            orig_type = original_df[col].dtype
            new_type = cleaned_df[col].dtype
            
            # Check for problematic type conversions
            if pd.api.types.is_numeric_dtype(orig_type) and not pd.api.types.is_numeric_dtype(new_type):
                validation_issues.append({
                    "issue_type": "problematic_type_conversion",
                    "description": f"Column '{col}' converted from numeric ({orig_type}) to non-numeric ({new_type})",
                    "severity": "medium"
                })
        
        return len(validation_issues) == 0, validation_issues


class DataAnalysisPipeline:
    """
    Pipeline that combines data cleaning and visualization.
    """
    
    def __init__(self):
        """Initialize the pipeline with cleaning and visualization agents"""
        self.cleaning_agent = DataCleaningAgent()
        
        # The visualization agent is expected to be imported from elsewhere
        # and passed to the process_data method
    
    def process_data(self, df: pd.DataFrame, visualization_agent) -> Dict[str, Any]:
        """
        Process data through the complete pipeline:
        1. Clean data using the cleaning agent
        2. Create visualizations using the visualization agent
        
        Args:
            df: Input pandas DataFrame
            visualization_agent: Instance of VisualizationAgent class
            
        Returns:
            Dictionary with results from both cleaning and visualization
        """
        result = {
            "success": False,
            "cleaning_results": None,
            "visualization_results": None
        }
        
        try:
            # Step 1: Clean the data
            print("Cleaning data...")
            cleaned_df, cleaning_results = self.cleaning_agent.clean_data(df)
            result["cleaning_results"] = cleaning_results
            
            if not cleaning_results.get("success", False):
                print("Data cleaning failed. Using original data for visualization.")
                df_for_viz = df
            else:
                print(f"Data cleaned successfully. Rows before: {len(df)}, after: {len(cleaned_df)}")
                df_for_viz = cleaned_df
            
            # Step 2: Create visualizations with cleaned data
            print("Creating visualizations...")
            viz_results = visualization_agent.create_visualization(df_for_viz)
            result["visualization_results"] = viz_results
            
            result["success"] = viz_results.get("success", False)
            
            return result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            result["success"] = False
            result["error"] = str(e)
            result["error_details"] = error_details
            print(f"Error in data processing pipeline: {str(e)}")
            return result