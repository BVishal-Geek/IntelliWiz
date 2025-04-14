import streamlit as st
import pandas as pd
import os
import json
import plotly.graph_objects as go
from dotenv import load_dotenv
import workflow as wf

# Load environment variables
load_dotenv()

# Load dataset from file
@st.cache_data
def load_data(file):
    file_path = file  # Update path if needed
    return pd.read_csv(file_path)

def main():
    st.title("📊 AI-Powered Data Analysis")
    st.write("Using AI to analyze datasets and visualize key insights automatically")

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
        df = load_data(uploaded_file)
    
        # Show dataset preview
        st.write("### Data Preview")
        st.dataframe(df.head())
    
        st.write(f"Dataset has {len(df)} rows and {len(df.columns)} columns")

        if st.button("Analyze Data"):
            with st.spinner("Analyzing data with AI..."):
                try:
                    # Create the analysis workflow with the dataframe
                    analysis_runner = wf.create_analysis_workflow(df)
                    
                    # Run the analysis workflow
                    results = analysis_runner()
                    analysis_result = results["analysis_result"]
                    visualization_data = results["visualization_data"]
                    
                    # Display analysis results
                    st.write("### AI-Powered Analysis Results")
                    st.write(analysis_result)
                    
                    # Display visualizations if available
                    if visualization_data and visualization_data.get("success"):
                        st.write("#### LLM Visualization Suggestions")
                        st.json(visualization_data["llm_visual_prompt_output"])
                        
                        if visualization_data and visualization_data.get("plots"):
                            if len(visualization_data["plots"]) > 1:
                                tabs = st.tabs([f"Visualization {i+1}" for i in range(len(visualization_data["plots"]))])
                                
                                for i, (tab, plot_data) in enumerate(zip(tabs, visualization_data["plots"])):
                                    with tab:
                                        st.subheader(plot_data["title"])
                                        
                                        # Convert the JSON string back to a Plotly figure
                                        fig_dict = json.loads(plot_data["figure"])
                                        fig = go.Figure(fig_dict)
                                        
                                        # Display the figure
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Add insightful description
                                        if "insight" in plot_data:
                                            st.write(plot_data["insight"])
                            else:
                                # Single visualization
                                plot_data = visualization_data["plots"][0]
                                st.subheader(plot_data["title"])
                                
                                # Convert the JSON string back to a Plotly figure
                                fig_dict = json.loads(plot_data["figure"])
                                fig = go.Figure(fig_dict)
                                
                                # Display the figure
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add insightful description
                                if "insight" in plot_data:
                                    st.write(plot_data["insight"])
                    
                    elif visualization_data:
                        st.warning(f"Could not create visualizations: {visualization_data.get('message', 'Unknown error')}")
                        if "details" in visualization_data:
                            with st.expander("Error Details"):
                                st.code(visualization_data["details"])
                
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

if __name__ == "__main__":
    main()