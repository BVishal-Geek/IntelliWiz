import pandas as pd
import plotly.express as px
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.tools import tool, BaseTool
from typing import List
import inspect
import sys


# ------------------------ Input Schemas ------------------------


class HistogramInput(BaseModel):
   # df: pd.DataFrame
   x: str = Field(..., description="Column to plot histogram")
   model_config = ConfigDict(arbitrary_types_allowed=True)


class BoxPlotInput(BaseModel):
   # df: pd.DataFrame
   x: str = Field(..., description="Categorical column for X-axis")
   y: str = Field(..., description="Numeric column for Y-axis")
   model_config = ConfigDict(arbitrary_types_allowed=True)


class ScatterPlotInput(BaseModel):
   # df: pd.DataFrame
   x: str = Field(..., description="Numeric column for X-axis")
   y: str = Field(..., description="Numeric column for Y-axis")
   model_config = ConfigDict(arbitrary_types_allowed=True)


# class CorrelationInput(BaseModel):
#     # df: pd.DataFrame
#     model_config = ConfigDict(arbitrary_types_allowed=True)
# ------------------------ Tool Factories ------------------------


def make_histogram_tool(df: pd.DataFrame):
   @tool("histogram")
   def histogram(input: HistogramInput) -> str:
       """Histogram for a numeric or integer column."""
       print("📊 histogram tool was called with:", input)
       fig = px.histogram(df, x=input.x, title=f"Histogram of {input.x}")
       return fig.to_json()
   return histogram


def make_box_plot_tool(df: pd.DataFrame):
   @tool("box_plot")
   def box_plot(input: BoxPlotInput) -> str:
       """Box plot for a numeric column."""
       print("📊 box_plot tool was called with:", input)
       fig = px.box(df, x=input.x, y=input.y, title=f"Box Plot: {input.y} by {input.x}")
       return fig.to_json()
   return box_plot


def make_scatter_tool(df: pd.DataFrame):
   @tool("scatter_plot")
   def scatter_plot(input: ScatterPlotInput) -> str:
       """Scatter plot between two numeric columns."""
       print("📊 scatter_plot tool was called with:", input)
       fig = px.scatter(df, x=input.x, y=input.y, title=f"Scatter Plot: {input.y} vs {input.x}")
       return fig.to_json()
   return scatter_plot


# def make_corr_heatmap_tool(df: pd.DataFrame):
#     @tool("correlation_heatmap")
#     def correlation_heatmap(_: CorrelationInput) -> str:
#         """Heatmap showing correlation between numeric columns."""
#         corr = df.corr(numeric_only=True)
#         fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap")
#         return fig.to_json()
#     return correlation_heatmap




# ------------------------ Tool Collector ------------------------


def get_all_tools(df: pd.DataFrame) -> List[BaseTool]:
   return [
       make_histogram_tool(df),
       make_box_plot_tool(df),
       make_scatter_tool(df),
       # make_corr_heatmap_tool(df)
   ]

