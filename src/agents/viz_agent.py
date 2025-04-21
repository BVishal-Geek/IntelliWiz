import os
import pandas as pd
from typing import Dict, Any


from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI  # Use this if you installed langchain-openai
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from visualization_tools import get_all_tools




class VizAgent:
   def __init__(self, df: pd.DataFrame):
       # Load tools (scatter_plot, histogram, etc.)
       columns = ', '.join(df.columns)


       self.tools = get_all_tools(df)


       # Define reusable prompt template structure
       self.prompt_template = ChatPromptTemplate.from_messages([
           ("system", f"""You are a data visualization expert.
       IMPORTANT: You MUST call one of the available visualization tools directly. Do NOT return JSON, markdown, or code.


       Available tools: histogram, scatter_plot, correlation_heatmap, box_plot
       Available columns: {columns}
       """),
           MessagesPlaceholder(variable_name="chat_history"),
           ("human", "{input}"),
           MessagesPlaceholder(variable_name="agent_scratchpad"),
       ])


       # Load the LLM
       self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


       # Create agent + executor
       agent = create_tool_calling_agent(llm=self.llm, tools=self.tools, prompt=self.prompt_template)
       self.agent_executor = AgentExecutor(agent=agent, tools=self.tools)


   def generate_visualization_prompt(self, df: pd.DataFrame) -> str:
       summary = df.describe(include='all', datetime_is_numeric=True).to_string()
       self.columns = ', '.join(df.columns)


       prompt = f"""
   You are a data visualization expert.


   Here is a summary of the dataset:
   {summary}


   Here are the columns in the dataset:
   {self.columns}


   Your task:
- Select the most relevant chart to visualize insights from the dataset.
- Use only the available tools.
- Return a valid JSON object with "tool_calls".


- Do NOT use natural language explanation.
- Do NOT include markdown or backticks.


- ONLY output JSON in this format:


{{
 "tool_calls": [
   {{
     "name": "box_plot",
     "arguments": {{
       "x": "Department",
       "y": "Salary"
     }}
   }}
 ]
}}


Available tools: histogram, scatter_plot, box_plot.
"""
       return prompt


   def suggest_visual_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
       """Invoke the agent with a prompt built from the dataframe."""
       prompt = self.generate_visualization_prompt(df)


       # 🔍 DEBUG: show what's sent in
       print("🧠 Prompt sent to agent_executor:")
       print(prompt)


       # Invoke agent
       response = self.agent_executor.invoke({
           "input": "What is a good chart to visualize this data?",
           "chat_history": [],
           "agent_scratchpad": []
       })


       # 🔍 DEBUG: show what comes out
       print("🧪 Response from agent_executor:", type(response), response)
       print("🧠 Agent Output:", response.content if hasattr(response, "content") else response)


       return response #AIMessage(content=response.get("output", ""))
