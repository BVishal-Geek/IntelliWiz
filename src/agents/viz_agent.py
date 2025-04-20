# import pandas as pd
# from typing import Dict, Any

# class VizAgent:

#     def suggest_visual_columns(self, df: pd.DataFrame) -> str:
#       summary = df.describe(include='all', datetime_is_numeric=True).to_string()
#       columns = ', '.join(df.columns)

#       prompt = f'''
# You are a data visualization expert.

# Here is a summary of the dataset:
# {summary}

# Here are the columns in the dataset:
# {columns}

# Your task is to suggest only 1 histogram visualizations to help explore this dataset.

# Please return your response in the following JSON format, exactly:

# {{
#   "tool_calls": [
#     {{
#       "name": "scatter_plot",
#       "arguments": {{
#           "x_column": "Age",
#           "y_column": "Salary"
#       }}
#     }},
#     {{
#       "name": "box_plot_bivariate",
#       "arguments": {{
#           "x_column": "Gender",
#           "y_column": "Salary"
#       }}
#     }}
#   ]
# }}

# ### Rules ###
# - Only use from this list of tool names: 
#   scatter_plot, histogram, bar_plot, pie_chart, box_plot, violin_plot_univariate, violin_plot_bivariate, box_plot_bivariate, bar_grouped_plot, line_plot
# - Arguments must exactly match the tool definitions:
#   - Use `"column"` for univariate plots.
#   - Use `"x_column"` and `"y_column"` for bivariate plots.
# - Only one value for `"y_column"` — never a list.
# - Ensure that column names are present in the dataset and have the expected types.
# - Do not include extra text — just return the JSON block.
# - Avoid suggesting saving files or using specific libraries.
# '''

#       return prompt


import os
import json
import pandas as pd
from typing import Dict, Any
from langchain.agents import create_tool_calling_agent, AgentExecutor
from visualization_tools import get_all_tools
from llama_wrapper import LlamaLangChainWrapper  # <-- this is your custom wrapper
# from cleaning_workflow import get_llama_client     # <-- this is your LlamaAPI client factory
from llamaapi import LlamaAPI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder





class VizAgent:
    def __init__(self):
        # Step 1: Load tools
        self.tools = get_all_tools()

        # Step 2: Create LlamaAPI client directly
        api_key = os.getenv("LLAMA_API_KEY")
        if not api_key:
            raise ValueError("❌ Llama API Key not found in environment variables!")
        llama_client = LlamaAPI(api_key)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data visualization expert. 
        IMPORTANT: You MUST use one of the available visualization tools to create charts.
        DO NOT provide code samples or explanations - ONLY call tool functions directly.
        Available tools: histogram, scatter_plot, correlation_heatmap, box_plot"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.llm = LlamaLangChainWrapper(client=llama_client, model="llama3.1-70b")
        # Step 3: Create the agent and executor
        agent = create_tool_calling_agent(llm=self.llm, tools=self.tools, prompt=prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools)
    
    def suggest_visual_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        summary = df.describe(include='all', datetime_is_numeric=True).to_string()
        columns = ', '.join(df.columns)

        prompt = f"""
You are a data visualization expert.

Here is a summary of the dataset:
{summary}

Here are the columns in the dataset:
{columns}

Your task is to suggest 1 insightful visualization that helps explore this dataset.

Please follow these rules carefully:

Example responses (use this format):
- "Create a histogram of the Satisfaction_Score column to explore its distribution."
- "Use a scatter_plot to visualize the relationship between Experience and Salary."
- "Generate a correlation_heatmap of numeric columns to identify strong relationships."
- "Create a box_plot to compare Salary across different Department values."

Guidelines:
- ONLY use tool names from this list: histogram, scatter_plot, correlation_heatmap, box_plot
- Use plain English sentences. DO NOT include Python code, backticks, or return JSON
- Reference only the column names provided above
- Be specific about which chart to create and which columns to use
- Do NOT include any explanations about code, dictionary, list, libraries or saving files

Respond with clear and concise instructions using natural language only.
"""

        # # Step 4: Invoke the LLM agent with this prompt
        # response = self.agent_executor.invoke({
        #     "input": prompt,
        #     "chat_history": [],
        #     "agent_scratchpad": []
        # })
        # parsed = json.loads(response)
        # print("📦 Parsed Llama API response JSON:", type(parsed))
        # tool_calls = parsed["tool_calls"]

        # # (Optional) Debug print
        # print("✅ Parsed tool calls:", tool_calls)

        # # Now return tool_calls if that’s what workflow.py expects
        # return {
        #     "success": True,
        #     "tool_calls": tool_calls,
        #     "raw_response": response  # optional, for logging
        # }
        # return response

        response = self.agent_executor.invoke({
        "input": prompt,
        "chat_history": [],
        "agent_scratchpad": []
    })
        print("🧪 Response from agent_executor:", type(response), response) 

        # If response is an AIMessage object
        print("🧠 Agent Output:", response.content if hasattr(response, "content") else response)

        return {
            "success": True,
            "response": response
        }