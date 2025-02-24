# app.py
import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from agent import create_analysis_workflow

# Load environment variables
load_dotenv(dotenv_path="C:\\Users\\swath\\GWU-Workspace\\CAPSTONE\\IntelliWiz\\src\\agents\\.env")

# Load dataset from file
file_path = "C:\\Users\\swath\\GWU-Workspace\\CAPSTONE\\IntelliWiz\\src\\agents\\data.csv"  # Update path if needed
df = pd.read_csv(file_path)

def main():
    st.title("📊 AI-Powered Data Analysis")
    st.write("Using AI to analyze datasets automatically")

    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key:
        st.sidebar.error("❌ No API Key found. Please check your .env file.")
        st.stop()
    st.sidebar.success("✅ API Key loaded")

    # Show dataset preview
    st.write("### Data Preview")
    st.dataframe(df.head())

    if st.button("Analyze Data"):
        with st.spinner("Analyzing data..."):
            try:
                # Pass the preloaded dataset
                agent = create_analysis_workflow(df)
                result = agent.invoke({})  # Call LangGraph workflow
                analysis_result = result["analysis_result"]
                st.write("### AI-Powered Analysis Results")
                st.write(analysis_result)
            except ValueError as e:
                st.error(f"Input validation failed: {e}")
            except Exception as e:
                st.error("Analysis failed")
                st.error(str(e))

if __name__ == "__main__":
    main()
