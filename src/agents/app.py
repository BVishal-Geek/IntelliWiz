# import streamlit as st
# import pandas as pd
# import os
# import json
# import plotly.graph_objects as go
# from dotenv import load_dotenv
# import workflow as wf
# from visualization_tools import ColumnInput, ColumnPairInput

# # Load environment variables
# load_dotenv()

# # Load dataset from file
# @st.cache_data
# def load_data(file):
#     return pd.read_csv(file)

# def main():
#     st.title("📊 AI-Powered Data Analysis")
#     st.write("Using AI to analyze datasets and visualize key insights automatically")

#     # Check API key
#     api_key = os.getenv("LLAMA_API_KEY")
#     if not api_key:
#         st.sidebar.error("❌ No API Key found. Please check your .env file.")
#         st.stop()
#     st.sidebar.success("✅ API Key loaded")

#     # File uploader
#     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
#     if uploaded_file:
#         # Load data
#         df = load_data(uploaded_file)
        
#         # Show dataset preview
#         st.write("### Data Preview")
#         st.dataframe(df.head())
#         st.write(f"Dataset has {len(df)} rows and {len(df.columns)} columns")

#         if st.button("Analyze Data"):
#             with st.spinner("Analyzing data with AI..."):
#                 try:
#                     # Create and run analysis workflow
#                     analysis_runner = wf.create_analysis_workflow(df)
#                     results = analysis_runner()
#                     visualization_tools = results["visualization_tools"]
#                     analysis_result = results["analysis_result"]
#                     visualization_data = results["visualization_data"]

#                     # Display analysis results
#                     st.write("### AI-Powered Analysis Results")
#                     st.write(analysis_result)

#                     if visualization_data and visualization_data.get("success"):
#                         st.write("#### LLM Visualization Suggestions")
#                         st.json(visualization_data["llm_visual_prompt_output"])

#                         try:
#                             tool_suggestions = json.loads(visualization_data["llm_visual_prompt_output"])
#                             tool_calls = tool_suggestions.get("tool_calls", [])
#                             visualization_data["plots"] = []

#                             st.write(visualization_data["plots"])

#                             for tool in tool_calls:
#                                 tool_name = tool["name"]
#                                 st.write(f"### Tool: `{tool_name}`")
                                
#                                 # Extract arguments from the tool call
#                                 arguments = tool.get("arguments", {})
#                                 column = arguments.get("column", "❌ column not found")
#                                 st.write("#### Column Name:", column)
#                                 st.write("#### Column Name:", type(column))


#                                 # SOLUTION: PASS RAW DICTIONARY DIRECTLY
#                                 if hasattr(visualization_tools, tool_name):
#                                     try:
#                                         # This is the key change - pass the raw dictionary directly
#                                         # input_model = ColumnInput(**arguments)
#                                         # st.write(input_model)
#                                         plot_json = getattr(visualization_tools, tool_name)(df, arguments)
                                        
#                                         visualization_data["plots"].append({
#                                             "title": f"{tool_name.replace('_', ' ').title()}",
#                                             "figure": plot_json,
#                                             "insight": f"Visualization created using tool: `{tool_name}`"
#                                         })
                                        
#                                     except Exception as e:
#                                         st.warning(f"Tool `{tool_name}` failed: {str(e)}")
#                                         st.error(f"Error details: {str(e)}")
#                                 else:
#                                     st.warning(f"Tool `{tool_name}` not found in visualization tools")
                                        
#                             # Display generated plots
#                             if visualization_data.get("plots"):
#                                 tabs = st.tabs([f"Visualization {i+1}" for i in range(len(visualization_data["plots"]))])
                                
#                                 for i, (tab, plot_data) in enumerate(zip(tabs, visualization_data["plots"])):
#                                     with tab:
#                                         st.subheader(plot_data["title"])
#                                         fig_dict = json.loads(plot_data["figure"])
#                                         fig = go.Figure(fig_dict)
#                                         st.plotly_chart(fig, use_container_width=True)
#                                         if "insight" in plot_data:
#                                             st.write(plot_data["insight"])
                                            
#                         except Exception as e:
#                             st.warning(f"Failed to generate plots from LLM suggestions: {str(e)}")
#                             import traceback
#                             st.error(f"Error details: {traceback.format_exc()}")
                    
#                     elif visualization_data:
#                         st.warning(f"Could not create visualizations: {visualization_data.get('message', 'Unknown error')}")

#                 except Exception as e:
#                     st.error(f"Analysis failed: {str(e)}")
#                     import traceback
#                     st.error(f"Error details: {traceback.format_exc()}")

# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import os
import json
import plotly.graph_objects as go
from dotenv import load_dotenv
import workflow as wf
from langchain_core.messages import HumanMessage  # Needed to build prompt message
from viz_agent import VizAgent

# Load environment variables
load_dotenv()

# Load dataset from file
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

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
        visualization_agent = VizAgent()
        
        # Show dataset preview
        st.write("### Data Preview")
        st.dataframe(df.head())
        st.write(f"Dataset has {len(df)} rows and {len(df.columns)} columns")

        if st.button("Analyze Data"):
            with st.spinner("Analyzing data with AI..."):
                try:
                    # Create and run analysis workflow
                    analysis_runner = wf.create_analysis_workflow(df)
                    results = analysis_runner()

                    # visualization_agent = results["visualization_tools"]  # This now holds your VizAgent class
                    analysis_result = results["analysis_result"]
                    visualization_data = results["visualization_data"]

                    # Display analysis results
                    st.write("### AI-Powered Analysis Results")
                    st.write(analysis_result)
                    if visualization_data and visualization_data.get("success"):
                        st.write("#### LLM Visualization Suggestions")
                        st.write(visualization_data["llm_visual_prompt_output"])

                        try:
                            prompt = visualization_data["llm_visual_prompt_output"]
                            print("🧠 Prompt sent to agent_executor:")
                            print(prompt)
                            response = visualization_agent.agent_executor.invoke({
                                "input": prompt,
                                "chat_history": [],
                                "agent_scratchpad": []
                            })

                            plot_json = response.content if hasattr(response, "content") else response.get("output")

                            visualization_data["plots"] = [{
                                "title": "Visualization from AI",
                                "figure": plot_json,
                                "insight": "Visualization created based on AI suggestion"
                            }]
                        except Exception as e:
                            st.warning(f"Failed to generate plot from AI response: {str(e)}")
                            import traceback
                            st.error(f"Error details: {traceback.format_exc()}")
                    # if visualization_data and visualization_data.get("success"):
                    #     st.write("#### LLM Visualization Suggestions")
                    #     st.json(visualization_data["llm_visual_prompt_output"])

                    #     try:
                    #         prompt = visualization_data["llm_visual_prompt_output"]
                    #         visualization_data["plots"] = []

                    #         st.write("### Generated Visualizations:")

                    #         for tool in tool_calls:
                    #             tool_name = tool["name"]
                    #             arguments = tool.get("arguments", {})
                    #             column = arguments.get("column") or arguments.get("x_column")
                    #             st.write(f"### Tool: `{tool_name}`")
                    #             st.write("#### Selected Column(s):", arguments)

                    #             try:
                    #                 # Construct a natural prompt for the agent
                    #                 if "x_column" in arguments and "y_column" in arguments:
                    #                     prompt = f"Create a {tool_name.replace('_', ' ')} between {arguments['x_column']} and {arguments['y_column']}"
                    #                 elif "column" in arguments:
                    #                     prompt = f"Create a {tool_name.replace('_', ' ')} for column {arguments['column']}"
                    #                 else:
                    #                     prompt = f"Call the tool `{tool_name}` using: {arguments}"

                    #                 # Use agent_executor to invoke the tool
                    #                 response = visualization_agent.agent_executor.invoke({
                    #                     "input": prompt
                    #                 })

                    #                 # Extract plot JSON from response
                    #                 plot_json = response.content if hasattr(response, "content") else response.get("output")

                    #                 visualization_data["plots"].append({
                    #                     "title": f"{tool_name.replace('_', ' ').title()}",
                    #                     "figure": plot_json,
                    #                     "insight": f"Visualization created using tool: `{tool_name}`"
                    #                 })

                    #             except Exception as e:
                    #                 st.warning(f"Tool `{tool_name}` failed: {str(e)}")
                    #                 import traceback
                    #                 st.error(f"Error details: {traceback.format_exc()}")
                        except Exception as e:
                            st.warning(f"Failed to generate plots from LLM suggestions: {str(e)}")
                            import traceback
                            st.error(f"Error details: {traceback.format_exc()}")

                        # Display the generated plots
                        if visualization_data.get("plots"):
                            print('Sucessful plot generation')
                            print("🧪 Visualization data:", visualization_data["plots"])
                            tabs = st.tabs([f"Visualization {i+1}" for i in range(len(visualization_data["plots"]))])
                            for i, (tab, plot_data) in enumerate(zip(tabs, visualization_data["plots"])):
                                with tab:
                                    st.subheader(plot_data["title"])
                                    try:
                                        # First check if the response looks like JSON
                                        if isinstance(plot_data["figure"], str) and plot_data["figure"].strip().startswith("{"):
                                            # Looks like JSON, try to parse it
                                            fig_dict = json.loads(plot_data["figure"])
                                            # Process fig_dict normally
                                        else:
                                            # Not JSON, handle as Markdown/text
                                            st.subheader("Visualization from AI")
                                            st.markdown(plot_data["figure"])
                                            st.info("The AI provided explanation instead of a visualization. Try asking for a specific chart type.")
                                    except json.JSONDecodeError as e:
                                        # Handle parsing errors
                                        st.error(f"Error parsing visualization: {str(e)}")
                                        st.markdown(plot_data["figure"])
                                    print("🧪 Raw plot_data['figure']:", plot_data["figure"])
                                    print("🧪 Type of plot_data['figure']:", type(plot_data["figure"]))
                                    fig = go.Figure(fig_dict)
                                    st.plotly_chart(fig, use_container_width=True)
                                    if "insight" in plot_data:
                                        st.write(plot_data["insight"])

                    elif visualization_data:
                        st.warning(f"Could not create visualizations: {visualization_data.get('message', 'Unknown error')}")

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    import traceback
                    st.error(f"Error details: {traceback.format_exc()}")

if __name__ == "__main__":
    main()