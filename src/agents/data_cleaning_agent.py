import pandas as pd
import numpy as np
import json
import io
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import os
from llamaapi import LlamaAPI
import traceback

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
            "quality_issues": [],
            "cleaning_actions": [],
            "warning_flags": [],
        }
        
        try:
            # Step 1: Generate data profile
            print("Generating data profile...")
            result["data_profile"] = self._generate_data_profile(df)
            
            if "error" in result["data_profile"]:
                # If data profiling failed, add to warning flags and continue with basic cleaning
                result["warning_flags"].append(f"Data profiling failed: {result['data_profile']['error']}")
                print(f"Data profiling failed: {result['data_profile']['error']}")
                
                # Perform basic cleaning without the profile
                cleaned_df = self._perform_basic_cleaning(df)
                result["cleaning_actions"].append({
                    "column": "all",
                    "action_type": "basic_cleaning",
                    "description": "Applied basic cleaning due to profiling failure",
                    "justification": "Fallback cleaning when AI profiling fails"
                })
            else:
                # Step 2: Clean the data based on profile
                print("Cleaning data based on profile...")
                cleaned_df, cleaning_result = self._apply_cleaning(df, result["data_profile"])
                
                # Step 3: Record cleaning actions and quality issues
                result["quality_issues"] = cleaning_result.get("quality_issues", [])
                result["cleaning_actions"] = cleaning_result.get("actions", [])
                
                # If cleaning failed, use basic cleaning as fallback
                if "error" in cleaning_result:
                    result["warning_flags"].append(f"Advanced cleaning failed: {cleaning_result['error']}")
                    print(f"Advanced cleaning failed: {cleaning_result['error']}")
                    cleaned_df = self._perform_basic_cleaning(df)
                    result["cleaning_actions"].append({
                        "column": "all",
                        "action_type": "basic_cleaning",
                        "description": "Applied basic cleaning due to advanced cleaning failure",
                        "justification": "Fallback cleaning when AI cleaning fails"
                    })
            
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
            error_details = traceback.format_exc()
            result["success"] = False
            result["error"] = str(e)
            result["error_details"] = error_details
            print(f"Error during data cleaning: {str(e)}")
            print(error_details)
            # Return original data in case of error
            return df, result
    
    def _perform_basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic cleaning operations as a fallback when AI-based cleaning fails
        
        Args:
            df: Input pandas DataFrame
            
        Returns:
            Cleaned pandas DataFrame
        """
        print("Performing basic cleaning as fallback...")
        try:
            # Create a copy to avoid modifying the original
            cleaned_df = df.copy()
            
            # 1. Handle missing values
            # Fill numeric columns with median (more robust than mean)
            num_cols = df.select_dtypes(include=['number']).columns
            for col in num_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            
            # Fill categorical columns with mode or 'Unknown'
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                if cleaned_df[col].nunique() > 0:
                    # Use mode if available, otherwise use 'Unknown'
                    mode_value = cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else 'Unknown'
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                else:
                    cleaned_df[col] = cleaned_df[col].fillna('Unknown')
            
            # 2. Handle outliers in numeric columns
            for col in num_cols:
                # Calculate Q1, Q3 and IQR
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds for outliers (using 3*IQR for conservative approach)
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Cap outliers to bounds rather than removing them
                cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
            
            # 3. Standardize text fields
            for col in cat_cols:
                # Convert to string if not already
                cleaned_df[col] = cleaned_df[col].astype(str)
                
                # Strip whitespace
                cleaned_df[col] = cleaned_df[col].str.strip()
                
                # Convert to title case (often improves readability)
                # Only if the column appears to be a name or category (not a long text)
                if cleaned_df[col].str.len().mean() < 20:  # Heuristic for short text fields
                    cleaned_df[col] = cleaned_df[col].str.title()
            
            # 4. Handle duplicate rows (drop exactly identical rows)
            cleaned_df = cleaned_df.drop_duplicates()
            
            return cleaned_df
            
        except Exception as e:
            print(f"Error in basic cleaning: {str(e)}")
            print(traceback.format_exc())
            # If even basic cleaning fails, return the original
            return df
    
    def _fix_json_string(self, json_str: str) -> str:
        """
        Attempt to fix common JSON formatting issues in API responses
        
        Args:
            json_str: Potentially malformed JSON string
            
        Returns:
            Fixed JSON string or original if no fixes could be applied
        """
        if not json_str or not isinstance(json_str, str):
            return json_str
            
        # Fix 1: Try to extract valid JSON from markdown code blocks
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        json_blocks = re.findall(json_block_pattern, json_str)
        if json_blocks:
            for block in json_blocks:
                try:
                    # Try to parse the extracted block
                    json.loads(block)
                    return block  # Return the first valid JSON block
                except json.JSONDecodeError:
                    continue  # Try the next block if this one fails
        
        # Fix 2: Try to extract JSON from the beginning of the string to a potential end marker
        try:
            # Find the last closing brace or bracket
            last_brace_pos = json_str.rstrip().rfind('}')
            last_bracket_pos = json_str.rstrip().rfind(']')
            last_pos = max(last_brace_pos, last_bracket_pos)
            
            if last_pos > 0:
                # Check if we have valid JSON up to this point
                potential_json = json_str[:last_pos+1]
                json.loads(potential_json)
                return potential_json
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fix 3: Try to fix unescaped quotes in strings
        try:
            # This is a simple heuristic for fixing common JSON errors
            # Replace unescaped quotes within string values
            fixed_json = re.sub(r'(?<!")(")(?!")\s*:', r'\":', json_str)
            fixed_json = re.sub(r':\s*"([^"]*?)(?<!\\)"(\s*[,}])', r':"\1"\2', fixed_json)
            json.loads(fixed_json)
            return fixed_json
        except (json.JSONDecodeError, ValueError):
            pass
            
        # If no fixes worked, return the original
        return json_str
    
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
            # For large dataframes, use a sample to reduce API request size
            sample_size = min(1000, len(df))
            df_sample = df.sample(sample_size) if len(df) > sample_size else df
            
            # Convert to CSV for the prompt
            csv_buffer = io.StringIO()
            df_sample.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Basic dataset info
            dataset_info = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": {col: str(df[col].dtype) for col in df.columns}
            }
            
            # Add basic statistics to help the model understand the data better
            numeric_stats = {}
            for col in df.select_dtypes(include=['number']).columns:
                numeric_stats[col] = {
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "null_count": int(df[col].isna().sum())
                }
            
            # Categorical column info
            categorical_stats = {}
            for col in df.select_dtypes(include=['object', 'category']).columns:
                categorical_stats[col] = {
                    "unique_values": int(df[col].nunique()),
                    "null_count": int(df[col].isna().sum()),
                    "top_values": df[col].value_counts().head(3).to_dict()
                }
            
            # Create a more structured and simpler prompt to reduce JSON parsing issues
            prompt = f"""As a data quality expert, analyze this dataset and provide a comprehensive data profile.

Dataset Information:
```json
{json.dumps(dataset_info, indent=2)}
```

I'm providing you with a sample of the data (first {sample_size} rows):
```csv
{csv_string}
```

I've also calculated basic statistics for the dataset:
1. Numeric columns: {json.dumps(numeric_stats, indent=2)}
2. Categorical columns: {json.dumps(categorical_stats, indent=2)}

Please analyze this data and create a profile with the following structure:

1. "column_profiles": An array of objects, each containing:
   - "name": Column name
   - "inferred_type": Current data type
   - "recommended_type": Recommended data type
   - "missing_percentage": Percentage of missing values
   - "quality_issues": Array of issue objects with "issue_type", "description", and "severity"
   - "cleaning_recommendations": Array of recommendation objects with "action", "reason", and "priority"

2. "dataset_issues": Array of overall dataset issues

3. "recommendations": Array of overall recommendations

Return ONLY valid JSON with this structure. Do not include any explanatory text outside the JSON.
"""
            
            # API request for data profiling with reduced temperature for more consistent results
            api_request = {
                "model": "llama3.1-70b",
                "messages": [
                    {"role": "system", "content": "You are an expert data scientist specializing in data quality assessment. You provide detailed, accurate analysis in valid JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,  # Use zero temperature for most consistent results
                "response_format": {"type": "json_object"}  # Enforce JSON format
            }
            
            # Call the LLM API with better error handling
            try:
                response = self.llama.run(api_request)
                
                if response.status_code != 200:
                    return {
                        "error": f"API Error: {response.status_code} - {response.text}",
                        "column_profiles": [],
                        "dataset_issues": [],
                        "recommendations": []
                    }
                
                # Parse the LLM response with extensive error handling
                try:
                    api_response = response.json()
                    
                    # Check if required fields exist
                    if "choices" not in api_response or len(api_response["choices"]) == 0:
                        return {
                            "error": "API response missing 'choices' field",
                            "column_profiles": [],
                            "dataset_issues": [],
                            "recommendations": []
                        }
                        
                    choice = api_response["choices"][0]
                    if "message" not in choice or "content" not in choice["message"]:
                        return {
                            "error": "API response missing message content",
                            "column_profiles": [],
                            "dataset_issues": [],
                            "recommendations": []
                        }
                    
                    content = choice["message"]["content"]
                    
                    # Check if content is empty or not valid JSON
                    if not content or content.isspace():
                        return {
                            "error": "API returned empty content",
                            "column_profiles": [],
                            "dataset_issues": [],
                            "recommendations": []
                        }
                    
                    # Attempt to fix and parse JSON
                    try:
                        # First try direct parsing
                        profile_result = json.loads(content)
                    except json.JSONDecodeError as e:
                        # If direct parsing fails, try to fix the JSON
                        print(f"JSON parsing error: {str(e)}")
                        print(f"Attempting to fix JSON...")
                        
                        fixed_content = self._fix_json_string(content)
                        try:
                            profile_result = json.loads(fixed_content)
                            print("Successfully fixed JSON!")
                        except json.JSONDecodeError as e2:
                            # If fixing fails, return error with details
                            print(f"Could not fix JSON: {str(e2)}")
                            return {
                                "error": f"Invalid JSON response: {str(e2)}",
                                "column_profiles": [],
                                "dataset_issues": [],
                                "recommendations": []
                            }
                    
                    # Validate that the profile has the expected structure
                    if not isinstance(profile_result, dict):
                        return {
                            "error": "API returned non-object JSON",
                            "column_profiles": [],
                            "dataset_issues": [],
                            "recommendations": []
                        }
                    
                    # Ensure minimal required fields exist
                    if "column_profiles" not in profile_result:
                        profile_result["column_profiles"] = []
                    if "dataset_issues" not in profile_result:
                        profile_result["dataset_issues"] = []
                    if "recommendations" not in profile_result:
                        profile_result["recommendations"] = []
                    
                    return profile_result
                    
                except Exception as e:
                    return {
                        "error": f"Error processing API response: {str(e)}",
                        "column_profiles": [],
                        "dataset_issues": [],
                        "recommendations": []
                    }
                
            except Exception as e:
                return {
                    "error": f"API request failed: {str(e)}",
                    "column_profiles": [],
                    "dataset_issues": [],
                    "recommendations": []
                }
            
        except Exception as e:
            print(f"Error in data profiling: {str(e)}")
            print(traceback.format_exc())
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
            # Check if profile has error or is empty
            if "error" in profile or not profile.get("column_profiles"):
                return self._perform_basic_cleaning(df), {
                    "error": "Invalid or empty data profile",
                    "quality_issues": [],
                    "actions": []
                }
            
            # Extract cleaning recommendations from profile to create a smarter cleaning plan
            cleaning_plan = []
            
            # Add column-specific recommendations
            for col_profile in profile.get("column_profiles", []):
                col_name = col_profile.get("name")
                if not col_name or col_name not in df.columns:
                    continue
                    
                for rec in col_profile.get("cleaning_recommendations", []):
                    if not rec:
                        continue
                    cleaning_plan.append({
                        "column": col_name,
                        "action": rec.get("action", ""),
                        "reason": rec.get("reason", ""),
                        "priority": rec.get("priority", "low")
                    })
            
            # Add dataset-level recommendations
            for rec in profile.get("recommendations", []):
                if not rec:
                    continue
                cleaning_plan.append({
                    "column": "all",
                    "action": rec.get("action", ""),
                    "reason": rec.get("reason", ""),
                    "priority": rec.get("priority", "low")
                })
            
            # If no cleaning recommendations were found, use basic cleaning
            if not cleaning_plan:
                print("No cleaning recommendations found in profile, using basic cleaning")
                return self._perform_basic_cleaning(df), {
                    "error": "No cleaning recommendations in profile",
                    "quality_issues": [],
                    "actions": [{
                        "column": "all",
                        "action_type": "basic_cleaning",
                        "description": "Applied basic cleaning due to lack of recommendations",
                        "justification": "No specific cleaning recommendations in profile"
                    }]
                }
            
            # Convert to CSV for the prompt (limit size for large datasets)
            sample_size = min(1000, len(df))
            df_sample = df.sample(sample_size) if len(df) > sample_size else df
            
            csv_buffer = io.StringIO()
            df_sample.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Create a simpler, more structured prompt for data cleaning
            prompt = f"""As a data cleaning expert, I need your help to clean this dataset based on the cleaning plan.

Data Profile Summary:
```json
{json.dumps({"column_profiles": [{"name": cp["name"], "inferred_type": cp.get("inferred_type", ""), "recommended_type": cp.get("recommended_type", "")} for cp in profile.get("column_profiles", [])]}, indent=2)}
```

Cleaning Plan:
```json
{json.dumps(cleaning_plan, indent=2)}
```

Sample Dataset (first {sample_size} rows):
```csv
{csv_string}
```

Please provide Python code to clean this dataset addressing the issues in the cleaning plan.
Focus on these common cleaning tasks:

1. Data type corrections
2. Missing value handling
3. Outlier treatment
4. Text standardization
5. Deduplication
6. Value validation

Your code should be robust and handle potential errors gracefully.
Return only the JSON structure with these fields:
1. "quality_issues": Array of identified issues
2. "actions": Array of cleaning actions performed
3. "cleaning_code": Python code to clean the dataset

The cleaning_code must be valid Python that takes a dataframe 'df' as input and returns a cleaned dataframe 'cleaned_df'.
Do not include any markdown formatting in your code (no ```python tags).
"""
            
            # API request for data cleaning with reduced temperature for more consistent results
            api_request = {
                "model": "llama3.1-70b",
                "messages": [
                    {"role": "system", "content": "You are an expert data scientist specializing in data cleaning. You provide executable Python code that follows best practices for data cleaning. Return clean code without markdown tags."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # Lower temperature for more consistent results
                "response_format": {"type": "json_object"}  # Enforce JSON format
            }
            
            # Call the LLM API with better error handling
            try:
                response = self.llama.run(api_request)
                
                if response.status_code != 200:
                    return self._perform_basic_cleaning(df), {
                        "error": f"API Error: {response.status_code} - {response.text}",
                        "quality_issues": [],
                        "actions": []
                    }
                
                # Parse the LLM response with robust error handling
                try:
                    api_response = response.json()
                    
                    # Check if required fields exist
                    if "choices" not in api_response or len(api_response["choices"]) == 0:
                        return self._perform_basic_cleaning(df), {
                            "error": "API response missing 'choices' field",
                            "quality_issues": [],
                            "actions": []
                        }
                        
                    choice = api_response["choices"][0]
                    if "message" not in choice or "content" not in choice["message"]:
                        return self._perform_basic_cleaning(df), {
                            "error": "API response missing message content",
                            "quality_issues": [],
                            "actions": []
                        }
                    
                    content = choice["message"]["content"]
                    
                    # Check if content is empty or not valid JSON
                    if not content or content.isspace():
                        return self._perform_basic_cleaning(df), {
                            "error": "API returned empty content",
                            "quality_issues": [],
                            "actions": []
                        }
                    
                    # Attempt to fix and parse JSON
                    try:
                        # First try direct parsing
                        cleaning_result = json.loads(content)
                    except json.JSONDecodeError as e:
                        # If direct parsing fails, try to fix the JSON
                        print(f"JSON parsing error in cleaning response: {str(e)}")
                        print("Attempting to fix JSON...")
                        
                        fixed_content = self._fix_json_string(content)
                        try:
                            cleaning_result = json.loads(fixed_content)
                            print("Successfully fixed JSON!")
                        except json.JSONDecodeError as e2:
                            print(f"Could not fix JSON: {str(e2)}")
                            return self._perform_basic_cleaning(df), {
                                "error": f"Invalid JSON in cleaning response: {str(e2)}",
                                "quality_issues": [],
                                "actions": []
                            }
                except Exception as e:
                    return self._perform_basic_cleaning(df), {
                        "error": f"Error processing API cleaning response: {str(e)}",
                        "quality_issues": [],
                        "actions": []
                    }
                
            except Exception as e:
                return self._perform_basic_cleaning(df), {
                    "error": f"API request for cleaning failed: {str(e)}",
                    "quality_issues": [],
                    "actions": []
                }
            
            # Execute the cleaning code with enhanced error handling
            cleaning_code = cleaning_result.get("cleaning_code", "")
            if not cleaning_code:
                return self._perform_basic_cleaning(df), {
                    "error": "No cleaning code was provided by the API",
                    "quality_issues": cleaning_result.get("quality_issues", []),
                    "actions": cleaning_result.get("actions", [])
                }
            
            # Remove any markdown formatting if present and normalize indentation
            cleaning_code = cleaning_code.replace("```python", "").replace("```", "").strip()
            
            # Normalize indentation - ensure consistent 4-space indentation
            lines = cleaning_code.splitlines()
            # Find minimum indentation (ignoring empty lines)
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                # Remove common leading whitespace
                if min_indent > 0:
                    lines = [line[min_indent:] if line.strip() else line for line in lines]
                cleaning_code = "\n".join(lines)
            
            # Create a local context with the dataframe
            local_vars = {"df": df.copy(), "pd": pd, "np": np}
            
            # Make sure the cleaning code returns a cleaned_df
            if "cleaned_df" not in cleaning_code and "return" not in cleaning_code:
                cleaning_code += "\n\n# Ensure a cleaned_df is returned\ncleaned_df = df.copy()"
            
            # Create properly indented cleaning code
            import textwrap
            indented_cleaning_code = "\n".join(["    " + line for line in cleaning_code.splitlines()])
            safe_code = textwrap.dedent("""
                        try:
                            # Original cleaning code
                        {}
                            # Ensure the result is assigned to 'cleaned_df'
                            if 'cleaned_df' not in locals():
                                cleaned_df = df
                        except Exception as e:
                            import traceback
                            error_details = traceback.format_exc()
                            print(f"Error in cleaning code: {{str(e)}}")
                            print(error_details)
                            cleaned_df = df
                        """).format(indented_cleaning_code)
           
            
            # Execute the cleaning code
            try:
                exec(safe_code, globals(), local_vars)
                cleaned_df = local_vars.get("cleaned_df", df)
                
                # Verify the cleaned dataframe is valid
                if cleaned_df is None or len(cleaned_df) == 0:
                    print("Cleaning returned empty dataframe, using basic cleaning instead")
                    return self._perform_basic_cleaning(df), {
                        "error": "Cleaning code produced an empty dataframe",
                        "quality_issues": cleaning_result.get("quality_issues", []),
                        "actions": cleaning_result.get("actions", [])
                    }
                
                return cleaned_df, cleaning_result
                
            except Exception as e:
                error_details = traceback.format_exc()
                print(f"Error executing cleaning code: {str(e)}")
                print(error_details)
                return self._perform_basic_cleaning(df), {
                    "error": f"Error executing cleaning code: {str(e)}",
                    "quality_issues": cleaning_result.get("quality_issues", []),
                    "actions": cleaning_result.get("actions", [])
                }
            
        except Exception as e:
            print(f"Error in data cleaning: {str(e)}")
            print(traceback.format_exc())
            return self._perform_basic_cleaning(df), {
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



'''import pandas as pd
import numpy as np
import json
import io
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import os
from llamaapi import LlamaAPI
import traceback

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
            "quality_issues": [],
            "cleaning_actions": [],
            "warning_flags": [],
        }
        
        try:
            # Step 1: Generate data profile
            print("Generating data profile...")
            result["data_profile"] = self._generate_data_profile(df)
            
            if "error" in result["data_profile"]:
                # If data profiling failed, add to warning flags and continue with basic cleaning
                result["warning_flags"].append(f"Data profiling failed: {result['data_profile']['error']}")
                print(f"Data profiling failed: {result['data_profile']['error']}")
                
                # Perform basic cleaning without the profile
                cleaned_df = self._perform_basic_cleaning(df)
                result["cleaning_actions"].append({
                    "column": "all",
                    "action_type": "basic_cleaning",
                    "description": "Applied basic cleaning due to profiling failure",
                    "justification": "Fallback cleaning when AI profiling fails"
                })
            else:
                # Step 2: Clean the data based on profile
                print("Cleaning data based on profile...")
                cleaned_df, cleaning_result = self._apply_cleaning(df, result["data_profile"])
                
                # Step 3: Record cleaning actions and quality issues
                result["quality_issues"] = cleaning_result.get("quality_issues", [])
                result["cleaning_actions"] = cleaning_result.get("actions", [])
                
                # If cleaning failed, use basic cleaning as fallback
                if "error" in cleaning_result:
                    result["warning_flags"].append(f"Advanced cleaning failed: {cleaning_result['error']}")
                    print(f"Advanced cleaning failed: {cleaning_result['error']}")
                    cleaned_df = self._perform_basic_cleaning(df)
                    result["cleaning_actions"].append({
                        "column": "all",
                        "action_type": "basic_cleaning",
                        "description": "Applied basic cleaning due to advanced cleaning failure",
                        "justification": "Fallback cleaning when AI cleaning fails"
                    })
            
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
            error_details = traceback.format_exc()
            result["success"] = False
            result["error"] = str(e)
            result["error_details"] = error_details
            print(f"Error during data cleaning: {str(e)}")
            print(error_details)
            # Return original data in case of error
            return df, result
    
    def _perform_basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic cleaning operations as a fallback when AI-based cleaning fails
        
        Args:
            df: Input pandas DataFrame
            
        Returns:
            Cleaned pandas DataFrame
        """
        print("Performing basic cleaning as fallback...")
        try:
            # Create a copy to avoid modifying the original
            cleaned_df = df.copy()
            
            # 1. Handle missing values
            # Fill numeric columns with median (more robust than mean)
            num_cols = df.select_dtypes(include=['number']).columns
            for col in num_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            
            # Fill categorical columns with mode or 'Unknown'
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                if cleaned_df[col].nunique() > 0:
                    # Use mode if available, otherwise use 'Unknown'
                    mode_value = cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else 'Unknown'
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                else:
                    cleaned_df[col] = cleaned_df[col].fillna('Unknown')
            
            # 2. Handle outliers in numeric columns
            for col in num_cols:
                # Calculate Q1, Q3 and IQR
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds for outliers (using 3*IQR for conservative approach)
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Cap outliers to bounds rather than removing them
                cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
            
            # 3. Standardize text fields
            for col in cat_cols:
                # Convert to string if not already
                cleaned_df[col] = cleaned_df[col].astype(str)
                
                # Strip whitespace
                cleaned_df[col] = cleaned_df[col].str.strip()
                
                # Convert to title case (often improves readability)
                # Only if the column appears to be a name or category (not a long text)
                if cleaned_df[col].str.len().mean() < 20:  # Heuristic for short text fields
                    cleaned_df[col] = cleaned_df[col].str.title()
            
            # 4. Handle duplicate rows (drop exactly identical rows)
            cleaned_df = cleaned_df.drop_duplicates()
            
            return cleaned_df
            
        except Exception as e:
            print(f"Error in basic cleaning: {str(e)}")
            print(traceback.format_exc())
            # If even basic cleaning fails, return the original
            return df
    
    def _fix_json_string(self, json_str: str) -> str:
        """
        Attempt to fix common JSON formatting issues in API responses
        
        Args:
            json_str: Potentially malformed JSON string
            
        Returns:
            Fixed JSON string or original if no fixes could be applied
        """
        if not json_str or not isinstance(json_str, str):
            return json_str
            
        # Fix 1: Try to extract valid JSON from markdown code blocks
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        json_blocks = re.findall(json_block_pattern, json_str)
        if json_blocks:
            for block in json_blocks:
                try:
                    # Try to parse the extracted block
                    json.loads(block)
                    return block  # Return the first valid JSON block
                except json.JSONDecodeError:
                    continue  # Try the next block if this one fails
        
        # Fix 2: Try to extract JSON from the beginning of the string to a potential end marker
        try:
            # Find the last closing brace or bracket
            last_brace_pos = json_str.rstrip().rfind('}')
            last_bracket_pos = json_str.rstrip().rfind(']')
            last_pos = max(last_brace_pos, last_bracket_pos)
            
            if last_pos > 0:
                # Check if we have valid JSON up to this point
                potential_json = json_str[:last_pos+1]
                json.loads(potential_json)
                return potential_json
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fix 3: Try to fix unescaped quotes in strings
        try:
            # This is a simple heuristic for fixing common JSON errors
            # Replace unescaped quotes within string values
            fixed_json = re.sub(r'(?<!")(")(?!")\s*:', r'\":', json_str)
            fixed_json = re.sub(r':\s*"([^"]*?)(?<!\\)"(\s*[,}])', r':"\1"\2', fixed_json)
            json.loads(fixed_json)
            return fixed_json
        except (json.JSONDecodeError, ValueError):
            pass
            
        # If no fixes worked, return the original
        return json_str
    
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
            # For large dataframes, use a sample to reduce API request size
            sample_size = min(1000, len(df))
            df_sample = df.sample(sample_size) if len(df) > sample_size else df
            
            # Convert to CSV for the prompt
            csv_buffer = io.StringIO()
            df_sample.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Basic dataset info
            dataset_info = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": {col: str(df[col].dtype) for col in df.columns}
            }
            
            # Add basic statistics to help the model understand the data better
            numeric_stats = {}
            for col in df.select_dtypes(include=['number']).columns:
                numeric_stats[col] = {
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "null_count": int(df[col].isna().sum())
                }
            
            # Categorical column info
            categorical_stats = {}
            for col in df.select_dtypes(include=['object', 'category']).columns:
                categorical_stats[col] = {
                    "unique_values": int(df[col].nunique()),
                    "null_count": int(df[col].isna().sum()),
                    "top_values": df[col].value_counts().head(3).to_dict()
                }
            
            # Create a more structured and simpler prompt to reduce JSON parsing issues
            prompt = f"""As a data quality expert, analyze this dataset and provide a comprehensive data profile.

Dataset Information:
```json
{json.dumps(dataset_info, indent=2)}
```

I'm providing you with a sample of the data (first {sample_size} rows):
```csv
{csv_string}
```

I've also calculated basic statistics for the dataset:
1. Numeric columns: {json.dumps(numeric_stats, indent=2)}
2. Categorical columns: {json.dumps(categorical_stats, indent=2)}

Please analyze this data and create a profile with the following structure:

1. "column_profiles": An array of objects, each containing:
   - "name": Column name
   - "inferred_type": Current data type
   - "recommended_type": Recommended data type
   - "missing_percentage": Percentage of missing values
   - "quality_issues": Array of issue objects with "issue_type", "description", and "severity"
   - "cleaning_recommendations": Array of recommendation objects with "action", "reason", and "priority"

2. "dataset_issues": Array of overall dataset issues

3. "recommendations": Array of overall recommendations

Return ONLY valid JSON with this structure. Do not include any explanatory text outside the JSON.
"""
            
            # API request for data profiling with reduced temperature for more consistent results
            api_request = {
                "model": "llama3.1-70b",
                "messages": [
                    {"role": "system", "content": "You are an expert data scientist specializing in data quality assessment. You provide detailed, accurate analysis in valid JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,  # Use zero temperature for most consistent results
                "response_format": {"type": "json_object"}  # Enforce JSON format
            }
            
            # Call the LLM API with better error handling
            try:
                response = self.llama.run(api_request)
                
                if response.status_code != 200:
                    return {
                        "error": f"API Error: {response.status_code} - {response.text}",
                        "column_profiles": [],
                        "dataset_issues": [],
                        "recommendations": []
                    }
                
                # Parse the LLM response with extensive error handling
                try:
                    api_response = response.json()
                    
                    # Check if required fields exist
                    if "choices" not in api_response or len(api_response["choices"]) == 0:
                        return {
                            "error": "API response missing 'choices' field",
                            "column_profiles": [],
                            "dataset_issues": [],
                            "recommendations": []
                        }
                        
                    choice = api_response["choices"][0]
                    if "message" not in choice or "content" not in choice["message"]:
                        return {
                            "error": "API response missing message content",
                            "column_profiles": [],
                            "dataset_issues": [],
                            "recommendations": []
                        }
                    
                    content = choice["message"]["content"]
                    
                    # Check if content is empty or not valid JSON
                    if not content or content.isspace():
                        return {
                            "error": "API returned empty content",
                            "column_profiles": [],
                            "dataset_issues": [],
                            "recommendations": []
                        }
                    
                    # Attempt to fix and parse JSON
                    try:
                        # First try direct parsing
                        profile_result = json.loads(content)
                    except json.JSONDecodeError as e:
                        # If direct parsing fails, try to fix the JSON
                        print(f"JSON parsing error: {str(e)}")
                        print(f"Attempting to fix JSON...")
                        
                        fixed_content = self._fix_json_string(content)
                        try:
                            profile_result = json.loads(fixed_content)
                            print("Successfully fixed JSON!")
                        except json.JSONDecodeError as e2:
                            # If fixing fails, return error with details
                            print(f"Could not fix JSON: {str(e2)}")
                            return {
                                "error": f"Invalid JSON response: {str(e2)}",
                                "column_profiles": [],
                                "dataset_issues": [],
                                "recommendations": []
                            }
                    
                    # Validate that the profile has the expected structure
                    if not isinstance(profile_result, dict):
                        return {
                            "error": "API returned non-object JSON",
                            "column_profiles": [],
                            "dataset_issues": [],
                            "recommendations": []
                        }
                    
                    # Ensure minimal required fields exist
                    if "column_profiles" not in profile_result:
                        profile_result["column_profiles"] = []
                    if "dataset_issues" not in profile_result:
                        profile_result["dataset_issues"] = []
                    if "recommendations" not in profile_result:
                        profile_result["recommendations"] = []
                    
                    return profile_result
                    
                except Exception as e:
                    return {
                        "error": f"Error processing API response: {str(e)}",
                        "column_profiles": [],
                        "dataset_issues": [],
                        "recommendations": []
                    }
                
            except Exception as e:
                return {
                    "error": f"API request failed: {str(e)}",
                    "column_profiles": [],
                    "dataset_issues": [],
                    "recommendations": []
                }
            
        except Exception as e:
            print(f"Error in data profiling: {str(e)}")
            print(traceback.format_exc())
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
            # Check if profile has error or is empty
            if "error" in profile or not profile.get("column_profiles"):
                return self._perform_basic_cleaning(df), {
                    "error": "Invalid or empty data profile",
                    "quality_issues": [],
                    "actions": []
                }
            
            # Extract cleaning recommendations from profile to create a smarter cleaning plan
            cleaning_plan = []
            
            # Add column-specific recommendations
            for col_profile in profile.get("column_profiles", []):
                col_name = col_profile.get("name")
                if not col_name or col_name not in df.columns:
                    continue
                    
                for rec in col_profile.get("cleaning_recommendations", []):
                    if not rec:
                        continue
                    cleaning_plan.append({
                        "column": col_name,
                        "action": rec.get("action", ""),
                        "reason": rec.get("reason", ""),
                        "priority": rec.get("priority", "low")
                    })
            
            # Add dataset-level recommendations
            for rec in profile.get("recommendations", []):
                if not rec:
                    continue
                cleaning_plan.append({
                    "column": "all",
                    "action": rec.get("action", ""),
                    "reason": rec.get("reason", ""),
                    "priority": rec.get("priority", "low")
                })
            
            # If no cleaning recommendations were found, use basic cleaning
            if not cleaning_plan:
                print("No cleaning recommendations found in profile, using basic cleaning")
                return self._perform_basic_cleaning(df), {
                    "error": "No cleaning recommendations in profile",
                    "quality_issues": [],
                    "actions": [{
                        "column": "all",
                        "action_type": "basic_cleaning",
                        "description": "Applied basic cleaning due to lack of recommendations",
                        "justification": "No specific cleaning recommendations in profile"
                    }]
                }
            
            # Convert to CSV for the prompt (limit size for large datasets)
            sample_size = min(1000, len(df))
            df_sample = df.sample(sample_size) if len(df) > sample_size else df
            
            csv_buffer = io.StringIO()
            df_sample.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Create a simpler, more structured prompt for data cleaning
            prompt = f"""As a data cleaning expert, I need your help to clean this dataset based on the cleaning plan.

    Data Profile Summary:
    ```json
    {json.dumps({"column_profiles": [{"name": cp["name"], "inferred_type": cp.get("inferred_type", ""), "recommended_type": cp.get("recommended_type", "")} for cp in profile.get("column_profiles", [])]}, indent=2)}
    ```

    Cleaning Plan:
    ```json
    {json.dumps(cleaning_plan, indent=2)}
    ```

    Sample Dataset (first {sample_size} rows):
    ```csv
    {csv_string}
    ```

    Please provide Python code to clean this dataset addressing the issues in the cleaning plan.
    Focus on these common cleaning tasks:

    1. Data type corrections
    2. Missing value handling
    3. Outlier treatment
    4. Text standardization
    5. Deduplication
    6. Value validation

    Your code should be robust and handle potential errors gracefully.
    Return only the JSON structure with these fields:
    1. "quality_issues": Array of identified issues
    2. "actions": Array of cleaning actions performed
    3. "cleaning_code": Python code to clean the dataset

    The cleaning_code must be valid Python that takes a dataframe 'df' as input and returns a cleaned dataframe 'cleaned_df'.
    Do not include any markdown formatting in your code (no ```python tags).
    """
            
            # API request for data cleaning with reduced temperature for more consistent results
            api_request = {
                "model": "llama3.1-70b",
                "messages": [
                    {"role": "system", "content": "You are an expert data scientist specializing in data cleaning. You provide executable Python code that follows best practices for data cleaning. Return clean code without markdown tags."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # Lower temperature for more consistent results
                "response_format": {"type": "json_object"}  # Enforce JSON format
            }
            
            # Call the LLM API with better error handling
            try:
                response = self.llama.run(api_request)
                
                if response.status_code != 200:
                    return self._perform_basic_cleaning(df), {
                        "error": f"API Error: {response.status_code} - {response.text}",
                        "quality_issues": [],
                        "actions": []
                    }
                
                # Parse the LLM response with robust error handling
                try:
                    api_response = response.json()
                    
                    # Check if required fields exist
                    if "choices" not in api_response or len(api_response["choices"]) == 0:
                        return self._perform_basic_cleaning(df), {
                            "error": "API response missing 'choices' field",
                            "quality_issues": [],
                            "actions": []
                        }
                        
                    choice = api_response["choices"][0]
                    if "message" not in choice or "content" not in choice["message"]:
                        return self._perform_basic_cleaning(df), {
                            "error": "API response missing message content",
                            "quality_issues": [],
                            "actions": []
                        }
                    
                    content = choice["message"]["content"]
                    
                    # Check if content is empty or not valid JSON
                    if not content or content.isspace():
                        return self._perform_basic_cleaning(df), {
                            "error": "API returned empty content",
                            "quality_issues": [],
                            "actions": []
                        }
                    
                    # Attempt to fix and parse JSON
                    try:
                        # First try direct parsing
                        cleaning_result = json.loads(content)
                    except json.JSONDecodeError as e:
                        # If direct parsing fails, try to fix the JSON
                        print(f"JSON parsing error in cleaning response: {str(e)}")
                        print("Attempting to fix JSON...")
                        
                        fixed_content = self._fix_json_string(content)
                        try:
                            cleaning_result = json.loads(fixed_content)
                            print("Successfully fixed JSON!")
                        except json.JSONDecodeError as e2:
                            print(f"Could not fix JSON: {str(e2)}")
                            return self._perform_basic_cleaning(df), {
                                "error": f"Invalid JSON in cleaning response: {str(e2)}",
                                "quality_issues": [],
                                "actions": []
                            }
                except Exception as e:
                    return self._perform_basic_cleaning(df), {
                        "error": f"Error processing API cleaning response: {str(e)}",
                        "quality_issues": [],
                        "actions": []
                    }
                
            except Exception as e:
                return self._perform_basic_cleaning(df), {
                    "error": f"API request for cleaning failed: {str(e)}",
                    "quality_issues": [],
                    "actions": []
                }
            
            # Execute the cleaning code with enhanced error handling
            cleaning_code = cleaning_result.get("cleaning_code", "")
            if not cleaning_code:
                return self._perform_basic_cleaning(df), {
                    "error": "No cleaning code was provided by the API",
                    "quality_issues": cleaning_result.get("quality_issues", []),
                    "actions": cleaning_result.get("actions", [])
                }
            
            # Remove any markdown formatting if present and normalize indentation
            cleaning_code = cleaning_code.replace("```python", "").replace("```", "").strip()
            
            # Normalize indentation - ensure consistent 4-space indentation
            lines = cleaning_code.splitlines()
            # Find minimum indentation (ignoring empty lines)
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                # Remove common leading whitespace
                if min_indent > 0:
                    lines = [line[min_indent:] if line.strip() else line for line in lines]
                cleaning_code = "\n".join(lines)
            
            # Create a local context with the dataframe
            local_vars = {"df": df.copy(), "pd": pd, "np": np}
            
            # Make sure the cleaning code returns a cleaned_df
            if "cleaned_df" not in cleaning_code and "return" not in cleaning_code:
                cleaning_code += "\n\n# Ensure a cleaned_df is returned\ncleaned_df = df.copy()"
            
            # Fix the indentation in the safe_code template - remove extra spaces
            safe_code = """
    try:
        # Original cleaning code
    {}
        # Ensure the result is assigned to 'cleaned_df'
        if 'cleaned_df' not in locals():
            cleaned_df = df
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in cleaning code: {{str(e)}}")
        print(error_details)
        cleaned_df = df
    """.format(cleaning_code)
            
            # Execute the cleaning code
            try:
                exec(safe_code, globals(), local_vars)
                cleaned_df = local_vars.get("cleaned_df", df)
                
                # Verify the cleaned dataframe is valid
                if cleaned_df is None or len(cleaned_df) == 0:
                    print("Cleaning returned empty dataframe, using basic cleaning instead")
                    return self._perform_basic_cleaning(df), {
                        "error": "Cleaning code produced an empty dataframe",
                        "quality_issues": cleaning_result.get("quality_issues", []),
                        "actions": cleaning_result.get("actions", [])
                    }
                
                return cleaned_df, cleaning_result
                
            except Exception as e:
                error_details = traceback.format_exc()
                print(f"Error executing cleaning code: {str(e)}")
                print(error_details)
                return self._perform_basic_cleaning(df), {
                    "error": f"Error executing cleaning code: {str(e)}",
                    "quality_issues": cleaning_result.get("quality_issues", []),
                    "actions": cleaning_result.get("actions", [])
                }
            
        except Exception as e:
            print(f"Error in data cleaning: {str(e)}")
            print(traceback.format_exc())
            return self._perform_basic_cleaning(df), {
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
            return result'''



