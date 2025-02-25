import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple

class VisualizationAgent:
    """
    Agent responsible for creating meaningful visualizations from dataframes.
    No external API calls are made by this agent.
    """
    
    def _convert_to_datetime(self, series):
        """Helper method to convert a series to datetime with format inference"""
        # Try to infer a common date format from the first few non-null values
        sample_vals = series.dropna().head(5).astype(str).tolist()
        
        if sample_vals:
            # Check for common date patterns
            if all(len(val) == 10 and val.count('-') == 2 for val in sample_vals):
                format = '%Y-%m-%d'  # ISO format
            elif all(len(val) == 10 and val.count('/') == 2 for val in sample_vals):
                format = '%m/%d/%Y'  # US format
            else:
                format = None  # Let pandas infer
            
            return pd.to_datetime(series, errors='coerce', format=format)
        
        return pd.to_datetime(series, errors='coerce')
    
    def _is_date_column(self, series):
        """Helper method to check if a series contains date values"""
        return self._convert_to_datetime(series).notna().mean() > 0.8
        
    def identify_important_columns(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Identify the most important columns for visualization.
        Returns pairs of columns that would make meaningful visualizations.
        """
        # Initialize scores for columns
        column_scores = {}
        column_types = {}
        
        # Categorize columns and assign initial scores
        for col in df.columns:
            # Skip ID-like columns
            if col.lower() == 'id' or (col.lower().endswith('id') and len(col) <= 5):
                column_types[col] = "id"
                column_scores[col] = 0
                continue
                
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() > 10:  # Continuous numeric
                    column_types[col] = "numeric_continuous"
                    column_scores[col] = 3
                else:  # Discrete numeric
                    column_types[col] = "numeric_discrete"
                    column_scores[col] = 2
            elif df[col].nunique() <= max(10, len(df) * 0.3):  # Categorical with reasonable cardinality
                column_types[col] = "categorical"
                column_scores[col] = 2
            else:  # High cardinality categorical or other types
                column_types[col] = "other"
                column_scores[col] = 1
        
        # Adjust scores based on data quality
        for col in df.columns:
            if column_types.get(col) == "id":
                continue
                
            # Penalize columns with many missing values
            missing_ratio = df[col].isna().mean()
            column_scores[col] *= (1 - missing_ratio)
            
            # Boost scores for columns with good variation
            if column_types[col] in ["numeric_continuous", "numeric_discrete"]:
                try:
                    # For numeric columns, favor those with meaningful variation
                    variation = df[col].std() / (df[col].mean() if df[col].mean() != 0 else 1)
                    if 0.1 < variation < 10:  # Reasonable variation
                        column_scores[col] *= 1.2
                except:
                    pass
        
        # Find best column combinations
        viz_pairs = []
        
        # 1. Try to find a good categorical + numeric continuous pair
        categorical_cols = [col for col, type_ in column_types.items() if type_ == "categorical"]
        numeric_continuous_cols = [col for col, type_ in column_types.items() if type_ == "numeric_continuous"]
        
        if categorical_cols and numeric_continuous_cols:
            # Get top 2 categorical columns
            sorted_cat = sorted(categorical_cols, key=lambda x: column_scores[x], reverse=True)[:2]
            # Get top 2 numeric columns
            sorted_num = sorted(numeric_continuous_cols, key=lambda x: column_scores[x], reverse=True)[:2]
            
            # Create pairs between them
            for cat in sorted_cat:
                for num in sorted_num:
                    viz_pairs.append((cat, num))
        
        # 2. Try to find a good pair of numeric columns for scatter plot
        if len(numeric_continuous_cols) >= 2:
            sorted_nums = sorted(numeric_continuous_cols, key=lambda x: column_scores[x], reverse=True)
            # Create pairs from top 3 numeric columns
            for i in range(min(3, len(sorted_nums))):
                for j in range(i+1, min(3, len(sorted_nums))):
                    viz_pairs.append((sorted_nums[i], sorted_nums[j]))
        
        # 3. Add a time series if there's a date column and numeric column
        date_cols = []
        for col in df.columns:
            if column_types.get(col) == "id":
                continue
            try:
                # Try to infer a common date format from the first few non-null values
                sample_vals = df[col].dropna().head(5).astype(str).tolist()
                if sample_vals:
                    # Check for common date patterns
                    if all(len(val) == 10 and val.count('-') == 2 for val in sample_vals):
                        format = '%Y-%m-%d'  # ISO format
                    elif all(len(val) == 10 and val.count('/') == 2 for val in sample_vals):
                        format = '%m/%d/%Y'  # US format
                    else:
                        format = None  # Let pandas infer
                    
                    # Check if column can be converted to datetime
                    if pd.to_datetime(df[col], errors='coerce', format=format).notna().mean() > 0.8:
                        date_cols.append(col)
            except:
                continue
        
        if date_cols and numeric_continuous_cols:
            # Use all date columns with top numeric columns
            for date_col in date_cols[:2]:  # Limit to top 2 date columns
                for num_col in sorted(numeric_continuous_cols, key=lambda x: column_scores[x], reverse=True)[:2]:
                    viz_pairs.append((date_col, num_col))
        
        # 4. Add distributions of important columns
        # Add histogram for top numeric columns
        if numeric_continuous_cols:
            for col in sorted(numeric_continuous_cols, key=lambda x: column_scores[x], reverse=True)[:2]:
                viz_pairs.append((col, col))  # Same column for x and y indicates histogram
        
        # Add bar chart for top categorical columns
        if categorical_cols:
            for col in sorted(categorical_cols, key=lambda x: column_scores[x], reverse=True)[:2]:
                viz_pairs.append((col, col))  # Same column for x and y indicates bar chart
        
        # 5. If we still don't have enough pairs, add more combinations
        if len(viz_pairs) < 4:
            usable_cols = [col for col in df.columns if column_types.get(col) != "id"]
            if len(usable_cols) >= 2:
                for i in range(min(4, len(usable_cols))):
                    for j in range(i+1, min(4, len(usable_cols))):
                        pair = (usable_cols[i], usable_cols[j])
                        if pair not in viz_pairs and (usable_cols[j], usable_cols[i]) not in viz_pairs:
                            viz_pairs.append(pair)
                            if len(viz_pairs) >= 6:  # Cap at 6 total visualizations
                                break
                    if len(viz_pairs) >= 6:
                        break
        
        # Return at least one pair, at most six
        return viz_pairs[:6]

    def create_visualization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create visualizations based on important columns in the dataset.
        Returns a dictionary with visualization data.
        """
        viz_data = {"success": False, "plots": []}
        
        try:
            # Identify most important column pairs
            important_pairs = self.identify_important_columns(df)
            
            if not important_pairs:
                return {
                    "success": False, 
                    "message": "Couldn't identify suitable columns for visualization"
                }
            
            for idx, (col1, col2) in enumerate(important_pairs):
                plot_data = {"id": f"plot_{idx+1}"}
                
                # Special case: if same column is selected twice, we'll do a distribution plot
                if col1 == col2:
                    if pd.api.types.is_numeric_dtype(df[col1]):
                        # Histogram for numeric column
                        fig = px.histogram(
                            df, x=col1,
                            title=f'Distribution of {col1}',
                            nbins=min(30, max(10, df[col1].nunique())),
                        )
                    else:
                        # Bar chart for categorical column
                        value_counts = df[col1].value_counts().reset_index()
                        value_counts.columns = [col1, 'count']
                        fig = px.bar(
                            value_counts, 
                            x=col1, 
                            y='count',
                            title=f'Frequency of {col1} categories'
                        )
                
                # Check if first column looks like a date
                elif pd.api.types.is_datetime64_dtype(df[col1]) or (
                        not pd.api.types.is_numeric_dtype(df[col1]) and 
                        self._is_date_column(df[col1])
                    ):
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_dtype(df[col1]):
                        try:
                            temp_date = self._convert_to_datetime(df[col1])
                            if pd.api.types.is_numeric_dtype(df[col2]):
                                # Time series plot
                                temp_df = df.copy()
                                temp_df['temp_date'] = temp_date
                                temp_df = temp_df.dropna(subset=['temp_date'])
                                temp_df = temp_df.sort_values('temp_date')
                                
                                fig = px.line(
                                    temp_df, 
                                    x='temp_date', 
                                    y=col2,
                                    markers=True,
                                    title=f'Time series of {col2} over {col1}'
                                )
                            else:
                                # If second column isn't numeric, fallback to count by date
                                temp_df = df.copy()
                                temp_df['temp_date'] = temp_date
                                temp_df = temp_df.dropna(subset=['temp_date'])
                                date_counts = temp_df.groupby('temp_date').size().reset_index(name='count')
                                
                                fig = px.line(
                                    date_counts, 
                                    x='temp_date', 
                                    y='count',
                                    markers=True,
                                    title=f'Frequency over time for {col1}'
                                )
                        except:
                            # If date conversion fails completely, fall back to general case
                            fig = px.scatter(
                                df, x=col1, y=col2,
                                title=f'Relationship between {col1} and {col2}'
                            )
                    
                # General case based on column types
                elif pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                    # Scatter plot for two numeric columns
                    fig = px.scatter(
                        df, x=col1, y=col2,
                        trendline="ols" if len(df) < 10000 else None,
                        title=f'Relationship between {col1} and {col2}'
                    )
                    
                    # Add correlation coefficient in title
                    corr = df[col1].corr(df[col2])
                    fig.update_layout(
                        title=f'Relationship between {col1} and {col2} (Correlation: {corr:.2f})'
                    )
                    
                elif not pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                    # Bar chart or box plot
                    if df[col1].nunique() <= 15:  # Few enough categories
                        # First try grouped bar chart with means
                        grouped = df.groupby(col1)[col2].agg(['mean', 'count']).reset_index()
                        grouped.columns = [col1, 'mean', 'count']
                        
                        fig = px.bar(
                            grouped, 
                            x=col1, 
                            y='mean',
                            title=f'Average {col2} by {col1}',
                            text_auto='.2f'
                        )
                        
                        # Add box plot on top for distribution
                        for category in df[col1].unique():
                            subset = df[df[col1] == category][col2]
                            if len(subset) > 0:
                                fig.add_trace(
                                    go.Box(
                                        x=[category],
                                        y=subset,
                                        name=str(category),
                                        boxpoints='all',
                                        jitter=0.3,
                                        pointpos=0,
                                        marker=dict(size=3, opacity=0.6),
                                        showlegend=False
                                    )
                                )
                    else:
                        # Too many categories, take top N
                        top_cats = df[col1].value_counts().nlargest(15).index
                        df_filtered = df[df[col1].isin(top_cats)]
                        grouped = df_filtered.groupby(col1)[col2].mean().reset_index()
                        
                        fig = px.bar(
                            grouped, 
                            x=col1, 
                            y=col2,
                            title=f'Average {col2} by top 15 categories in {col1}',
                            text_auto='.2f'
                        )
                
                elif pd.api.types.is_numeric_dtype(df[col1]) and not pd.api.types.is_numeric_dtype(df[col2]):
                    # Numeric by categorical - box plots
                    if df[col2].nunique() <= 15:
                        fig = px.box(
                            df, 
                            x=col2, 
                            y=col1,
                            title=f'Distribution of {col1} by {col2}',
                            points="all"
                        )
                    else:
                        # Too many categories, take top N
                        top_cats = df[col2].value_counts().nlargest(15).index
                        df_filtered = df[df[col2].isin(top_cats)]
                        
                        fig = px.box(
                            df_filtered, 
                            x=col2, 
                            y=col1,
                            title=f'Distribution of {col1} by top 15 categories in {col2}',
                            points="all"
                        )
                
                else:
                    # Both categorical - heatmap or stacked bar
                    if df[col1].nunique() <= 15 and df[col2].nunique() <= 15:
                        # Create a contingency table
                        crosstab = pd.crosstab(df[col1], df[col2], normalize='index') * 100
                        
                        # Create heatmap
                        fig = px.imshow(
                            crosstab,
                            text_auto='.1f',
                            labels=dict(x=col2, y=col1, color="Percentage (%)"),
                            title=f'Relationship between {col1} and {col2}',
                            color_continuous_scale='Blues'
                        )
                    else:
                        # Too many categories, take top N for each
                        top_cats1 = df[col1].value_counts().nlargest(10).index
                        top_cats2 = df[col2].value_counts().nlargest(10).index
                        df_filtered = df[df[col1].isin(top_cats1) & df[col2].isin(top_cats2)]
                        
                        # Create a count-based bar chart
                        count_df = df_filtered.groupby([col1, col2]).size().reset_index(name='count')
                        
                        fig = px.bar(
                            count_df, 
                            x=col1, 
                            y='count', 
                            color=col2,
                            title=f'Counts of {col2} by {col1} (Top 10 categories each)'
                        )
                
                # Set common layout properties
                fig.update_layout(
                    template='plotly_white',
                    height=500,
                    margin=dict(t=80, b=50, l=50, r=50),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Add plot data to result
                plot_data["figure"] = fig.to_json()
                plot_data["title"] = fig.layout.title.text
                plot_data["x_column"] = col1
                plot_data["y_column"] = col2
                
                # Add descriptive insight about the visualization
                if col1 == col2:
                    if pd.api.types.is_numeric_dtype(df[col1]):
                        plot_data["insight"] = f"This histogram shows the distribution of {col1} values. The average is {df[col1].mean():.2f} with a standard deviation of {df[col1].std():.2f}."
                    else:
                        most_common = df[col1].value_counts().idxmax()
                        pct_most_common = df[col1].value_counts(normalize=True).max() * 100
                        plot_data["insight"] = f"This bar chart shows the frequency of each {col1} category. The most common value is '{most_common}' ({pct_most_common:.1f}% of data)."
                elif pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                    corr = df[col1].corr(df[col2])
                    corr_strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
                    corr_direction = "positive" if corr > 0 else "negative"
                    plot_data["insight"] = f"This scatter plot shows a {corr_strength} {corr_direction} correlation ({corr:.2f}) between {col1} and {col2}."
                else:
                    plot_data["insight"] = f"This visualization explores the relationship between {col1} and {col2}."
                
                viz_data["plots"].append(plot_data)
            
            viz_data["success"] = True
            return viz_data
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return {
                "success": False, 
                "message": f"Visualization failed: {str(e)}",
                "details": error_details
            }
