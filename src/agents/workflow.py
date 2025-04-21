import pandas as pd
from typing import TypedDict, Optional, Any, Dict, List
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from analysis_agent import AnalysisAgent
from viz_agent import VizAgent
from dotenv import load_dotenv
load_dotenv()


class AnalysisState(TypedDict):
   df: pd.DataFrame
   analysis_result: Optional[str]
   messages: List[Any]
   visualization_data: Optional[Dict[str, Any]]


def analyze_node(state: AnalysisState) -> AnalysisState:
   llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
   agent = AnalysisAgent()


   prompt = agent.prepare_analysis_prompt(state["df"])
   state["messages"].append(HumanMessage(content=prompt))


   response = llm.invoke(prompt)
   state["messages"].append(response)
   state["analysis_result"] = response.content


   return state


def suggest_plots_node(state: AnalysisState) -> AnalysisState:
   viz_agent = VizAgent(state["df"])
   response = viz_agent.suggest_visual_columns(state["df"])


   state["messages"].append(response)
   state["visualization_data"] = {
       "success": True,
       "output": response.content,
       "llm_visual_prompt_output": response.content
}
   print("🧪 Inside suggest_plots_node, visualization_data:", state["visualization_data"])
   return state




def create_analysis_workflow(df: pd.DataFrame):
   graph = StateGraph(AnalysisState)


   graph.add_node("analyze_data", analyze_node)
   graph.add_node("suggest_plots", suggest_plots_node)


   graph.add_edge(START, "analyze_data")
   graph.add_edge("analyze_data", "suggest_plots")
   graph.add_edge("suggest_plots", END)


   # graph.set_entry_point(START)


   app = graph.compile()


   def run_analysis():
       initial_state = {
           "df": df,
           "analysis_result": None,
           "messages": []
       }
       result = app.invoke(initial_state)
       print("📦 Result of analysis:", result)
       return {
           "analysis_result": result["analysis_result"],
           "visualization_data": result["visualization_data"]
       }
   return run_analysis