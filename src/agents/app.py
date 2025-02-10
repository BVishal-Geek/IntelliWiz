import streamlit as st
import os
from dotenv import load_dotenv
from eda_agent import EDAAgent
from agent import AnalysisAgent

def main():
    load_dotenv(dotenv_path='C:/Users/yashk/Downloads/IntelliWiz/src/agents/.env')
    
    st.title("📊 Data Analysis System")
    st.write("Upload your data file for AI-powered analysis")
    
    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key:
        st.sidebar.error("❌ No API Key found")
        st.stop()
    st.sidebar.success("✅ API Key loaded")
    
    uploaded_file = st.file_uploader("Upload data file", type=["csv", "xlsx", "json"])
    
    if uploaded_file:
        try:
            with st.spinner("Loading data..."):
                eda_agent = EDAAgent(uploaded_file)
                st.success("✅ Data loaded successfully")
            
            st.write("### Data Preview")
            st.dataframe(eda_agent.df.head())
            
            if st.button("Analyze Data"):
                with st.spinner("Analyzing data..."):
                    try:
                        data_description = eda_agent.get_data_description()
                        analysis_agent = AnalysisAgent(api_key)
                        result = analysis_agent.analyze_data(data_description)
                        
                        st.write("### Analysis Results")
                        st.write(result)
                        
                        with st.expander("Show Data Description"):
                            st.text(data_description)
                            
                    except Exception as e:
                        st.error("Analysis failed")
                        st.error(str(e))
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()