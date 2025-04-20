import streamlit as st
import pandas as pd
import os
import json
import plotly.graph_objects as go
from dotenv import load_dotenv
from workflow import create_analysis_workflow

load_dotenv()

@st.cache_data
def load_data(file):
    file_path = file  # Update path if needed
    return pd.read_csv(file_path)

def main():
    st.title("📊 AI-Powered Data Analysis")
    st.write("Using AI to analyze datasets and visualize key insights automatically")

    # Set debug mode in sidebar
    st.sidebar.title("Debug Options")
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=True)

    # Check API key
    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key:
        st.sidebar.error("❌ No API Key found. Please check your .env file.")
        st.stop()
    st.sidebar.success("✅ API Key loaded")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        # Load data
        try:
            df = load_data(uploaded_file)
            
            # Check if dataframe is valid
            if df is None or len(df) == 0:
                st.error("⚠️ The uploaded file appears to be empty.")
                st.stop()
                
            if len(df.columns) == 0:
                st.error("⚠️ The uploaded file doesn't have any columns.")
                st.stop()
                
        except Exception as e:
            st.error(f"⚠️ Error loading file: {str(e)}")
            st.stop()
    
        # Show dataset preview
        st.write("### Data Preview")
        st.dataframe(df.head())
    
        st.write(f"Dataset has {len(df)} rows and {len(df.columns)} columns")
        
        # Display column info
        if debug_mode:
            st.write("### Column Information")
            col_info = {
                "Column": df.columns.tolist(),
                "Type": [str(df[col].dtype) for col in df.columns],
                "Non-Null Count": [df[col].count() for col in df.columns],
                "Null Count": [df[col].isna().sum() for col in df.columns],
                "Unique Values": [df[col].nunique() for col in df.columns]
            }
            st.dataframe(pd.DataFrame(col_info).set_index("Column"))

        if st.button("Analyze Data"):
            with st.spinner("Analyzing data with AI..."):
                try:
                    # Create the analysis workflow with the dataframe and debug mode
                    analysis_runner = create_analysis_workflow(df, debug_mode=debug_mode)
                    
                    # Run the analysis workflow
                    results = analysis_runner()
                    
                    # Display debug logs if in debug mode
                    if debug_mode and "debug_logs" in results:
                        with st.expander("🔍 Debug Logs", expanded=False):
                            for i, log in enumerate(results["debug_logs"]):
                                st.text(f"{i+1}. {log}")
                    
                    # SECTION 1: Original Data Analysis
                    st.write("## Original Data Analysis")
                    st.write(results["original_analysis"]["result"])
                    
                    # Display original visualizations if available
                    original_viz_data = results["original_analysis"]["visualizations"]
                    display_visualizations(
                        original_viz_data, 
                        "Original Data Visualizations",
                        debug_mode
                    )
                    
                    # SECTION 2: Data Cleaning Results
                    cleaning_results = results["cleaning"]
                    if cleaning_results:
                        st.write("## Data Cleaning Results")
                        st.write(f"- Quality issues identified: {len(cleaning_results.get('quality_issues', []))}")
                        st.write(f"- Cleaning actions performed: {len(cleaning_results.get('cleaning_actions', []))}")
                        
                        if len(cleaning_results.get('cleaning_actions', [])) > 0:
                            st.write("### Top Cleaning Actions:")
                            for action in cleaning_results.get('cleaning_actions', [])[:5]:
                                st.write(f"- {action.get('column', 'dataset')}: {action.get('description', 'N/A')}")
                        
                        # Show data shape changes 
                        if 'original_data' in cleaning_results and 'cleaned_data' in cleaning_results:
                            orig_shape = cleaning_results['original_data'].get('shape', (0, 0))
                            clean_shape = cleaning_results['cleaned_data'].get('shape', (0, 0))
                            
                            if orig_shape != clean_shape:
                                st.write(f"Data shape changed from {orig_shape} to {clean_shape}")
                                
                        # Display any warnings in debug mode
                        if debug_mode and cleaning_results.get('warning_flags'):
                            st.warning("Cleaning Warnings:")
                            for warning in cleaning_results.get('warning_flags', []):
                                st.write(f"- {warning}")
                    
                    # SECTION 3: Cleaned Data Analysis
                    st.write("## Cleaned Data Analysis")
                    st.write(results["cleaned_analysis"]["result"])
                    
                    # Display cleaned visualizations if available
                    cleaned_viz_data = results["cleaned_analysis"]["visualizations"]
                    display_visualizations(
                        cleaned_viz_data, 
                        "Cleaned Data Visualizations",
                        debug_mode
                    )
                
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    import traceback
                    error_trace = traceback.format_exc()
                    if debug_mode:
                        st.code(error_trace)

def display_visualizations(viz_data, section_title, debug_mode=False):
    """
    Helper function to display visualizations with error handling and debug info
    """
    if not viz_data:
        st.write(f"### {section_title}")
        st.info("No visualization data available")
        return
        
    # Display debug info if enabled
    if debug_mode:
        with st.expander("🔍 Visualization Debug Info", expanded=False):
            if "debug_info" in viz_data:
                st.json(viz_data["debug_info"])
            
            st.write(f"Success: {viz_data.get('success', False)}")
            st.write(f"Fallback used: {viz_data.get('is_fallback', False)}")
            st.write(f"Number of plots: {len(viz_data.get('plots', []))}")
    
    # Check if visualizations were generated successfully
    if viz_data and viz_data.get("success"):
        st.write(f"### {section_title}")
        
        # If using fallback visualizations, show a notice
        if viz_data.get("is_fallback", False):
            st.info("⚠️ Using fallback visualizations due to issues with AI-generated visualizations")
        
        # Create tabs for multiple visualizations
        if len(viz_data["plots"]) > 1:
            tabs = st.tabs([f"Visualization {i+1}" for i in range(len(viz_data["plots"]))])
            
            for i, (tab, plot_data) in enumerate(zip(tabs, viz_data["plots"])):
                with tab:
                    try:
                        st.subheader(plot_data["title"])
                        
                        # Convert the JSON string back to a Plotly figure
                        fig_dict = json.loads(plot_data["figure"])
                        fig = go.Figure(fig_dict)
                        
                        # Display the figure
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add insightful description
                        if "insight" in plot_data and plot_data["insight"]:
                            st.write(plot_data["insight"])
                    except Exception as e:
                        st.error(f"Error displaying visualization: {str(e)}")
                        if debug_mode:
                            import traceback
                            st.code(traceback.format_exc())
        elif len(viz_data["plots"]) == 1:
            # Single visualization
            try:
                plot_data = viz_data["plots"][0]
                st.subheader(plot_data["title"])
                
                # Convert the JSON string back to a Plotly figure
                fig_dict = json.loads(plot_data["figure"])
                fig = go.Figure(fig_dict)
                
                # Display the figure
                st.plotly_chart(fig, use_container_width=True)
                
                # Add insightful description
                if "insight" in plot_data and plot_data["insight"]:
                    st.write(plot_data["insight"])
            except Exception as e:
                st.error(f"Error displaying visualization: {str(e)}")
                if debug_mode:
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("No visualizations were generated.")
    else:
        st.write(f"### {section_title}")
        
        # Show error message if visualizations failed
        if viz_data and not viz_data.get("success"):
            if "error" in viz_data:
                st.error(f"❌ Visualization failed: {viz_data['error']}")
            else:
                st.error("❌ Failed to generate visualizations")
            
            # Create fallback visualization on the fly if necessary
            st.info("Creating basic visualizations...")
            
            # Create a simple data summary
            st.write("#### Data Summary")
            if "current_df" in st.session_state:
                df = st.session_state["current_df"]
                
                # Display numeric columns summary
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    st.write("Summary of numeric columns:")
                    st.dataframe(df[numeric_cols].describe())
                    
                    # Create a simple histogram for the first numeric column
                    if len(numeric_cols) > 0:
                        col = numeric_cols[0]
                        st.write(f"Distribution of {col}")
                        st.bar_chart(df[col])
        else:
            st.info("No visualization data available")

if __name__ == "__main__":
    main()