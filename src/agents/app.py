import streamlit as st
import pandas as pd
import os
import json
import plotly.graph_objects as go
from dotenv import load_dotenv
import workflow as wf
from viz_agent import VizAgent


# Load environment variables
load_dotenv()


@st.cache_data
def load_data(file):
   return pd.read_csv(file)


def main():
   st.title("📊 AI-Powered Data Analysis")
   st.write("Upload your CSV file and let the AI analyze and visualize insights.")


   # API key check (can be removed if not needed)
   if not os.getenv("OPENAI_API_KEY"):
       st.sidebar.error("❌ OPENAI_API_KEY not found.")
       st.stop()
   st.sidebar.success("✅ API Key loaded")


   uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
   if uploaded_file:
       df = load_data(uploaded_file)
       st.write("### Data Preview")
       st.dataframe(df.head())


       if st.button("Analyze Data"):
           with st.spinner("Running AI analysis..."):
               try:
                   analysis_runner = wf.create_analysis_workflow(df)
                   results = analysis_runner()


                   analysis_result = results["analysis_result"]
                   visualization_data = results["visualization_data"]


                   st.write("### AI Insights")
                   st.markdown(analysis_result)


                   if visualization_data and visualization_data.get("success"):
                       st.write("### AI-Suggested Visualization Prompt")
                       prompt = visualization_data["llm_visual_prompt_output"]
                       st.code(prompt)


                       # Pass to agent to interpret prompt and call tool
                       visualization_agent = VizAgent(df)
                       response = visualization_agent.agent_executor.invoke({
                           "input": prompt,
                           "chat_history": [],
                           "agent_scratchpad": []
                       })


                       output = response.content if hasattr(response, "content") else response.get("output")


                       # Attempt to parse output as JSON (tool output)
                       try:
                           fig_dict = json.loads(output)
                           fig = go.Figure(fig_dict)
                           st.plotly_chart(fig, use_container_width=True)
                       except json.JSONDecodeError:
                           st.subheader("📝 AI Description")
                           st.markdown(output)
                           st.info("AI provided a suggestion but did not generate a chart. Try refining the prompt.")


                   elif visualization_data:
                       st.warning(f"Could not create visualizations: {visualization_data.get('message', 'Unknown error')}")


               except Exception as e:
                   st.error(f"Analysis failed: {str(e)}")
                   import traceback
                   st.error(traceback.format_exc())


if __name__ == "__main__":
   main()