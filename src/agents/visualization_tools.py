import pandas as pd
import plotly.express as px
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Dict, Any

# ----------------- Input Models -----------------

class ColumnInput(BaseModel):
    column: str

class ColumnPairInput(BaseModel):
    x_column: str
    y_column: str

# ----------------- Visualization Class -----------------

class VisualizationTools:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ---------- Univariate Plots ----------

    @tool
    def histogram(self, input: ColumnInput) -> str:
        """Histogram for a numeric column."""
        fig = px.histogram(self.df, x=input.column, title=f'Histogram of {input.column}')
        return fig.to_json()

    @tool
    def box_plot(self, input: ColumnInput) -> str:
        """Box plot for a numeric column."""
        fig = px.box(self.df, y=input.column, title=f'Box Plot of {input.column}')
        return fig.to_json()

    @tool
    def bar_plot(self, input: ColumnInput) -> str:
        """Bar plot for a categorical column."""
        value_counts = self.df[input.column].value_counts().reset_index()
        value_counts.columns = [input.column, 'count']
        fig = px.bar(value_counts, x=input.column, y='count', title=f'Bar Plot of {input.column}')
        return fig.to_json()

    @tool
    def pie_chart(self, input: ColumnInput) -> str:
        """Pie chart for a categorical column."""
        value_counts = self.df[input.column].value_counts().reset_index()
        value_counts.columns = [input.column, 'count']
        fig = px.pie(value_counts, names=input.column, values='count', title=f'Pie Chart of {input.column}')
        return fig.to_json()

    @tool
    def violin_plot_univariate(self, input: ColumnInput) -> str:
        """Violin plot for a numeric column."""
        fig = px.violin(self.df, y=input.column, box=True, title=f'Violin Plot of {input.column}')
        return fig.to_json()

    # ---------- Bivariate Plots ----------

    @tool
    def scatter_plot(self, input: ColumnPairInput) -> str:
        """Scatter plot between two numeric columns."""
        fig = px.scatter(self.df, x=input.x_column, y=input.y_column, title=f'Scatter Plot: {input.y_column} vs {input.x_column}')
        return fig.to_json()

    @tool
    def line_plot(self, input: ColumnPairInput) -> str:
        """Line plot for numeric or datetime x-axis."""
        fig = px.line(self.df, x=input.x_column, y=input.y_column, title=f'Line Plot: {input.y_column} over {input.x_column}')
        return fig.to_json()

    @tool
    def box_plot_bivariate(self, input: ColumnPairInput) -> str:
        """Box plot for categorical vs numeric."""
        fig = px.box(self.df, x=input.x_column, y=input.y_column, title=f'Box Plot: {input.y_column} by {input.x_column}')
        return fig.to_json()

    @tool
    def violin_plot_bivariate(self, input: ColumnPairInput) -> str:
        """Violin plot for categorical vs numeric."""
        fig = px.violin(self.df, x=input.x_column, y=input.y_column, box=True, title=f'Violin Plot: {input.y_column} by {input.x_column}')
        return fig.to_json()

    @tool
    def bar_grouped_plot(self, input: ColumnPairInput) -> str:
        """Grouped bar plot for categorical vs aggregated numeric."""
        grouped = self.df.groupby(input.x_column)[input.y_column].mean().reset_index()
        fig = px.bar(grouped, x=input.x_column, y=input.y_column, title=f'Average {input.y_column} by {input.x_column}')
        return fig.to_json()

    # ---------- New Additions ----------

    @tool
    def correlation_heatmap(self, _: Dict[str, Any]) -> str:
        """Heatmap showing correlation between numeric columns."""
        corr = self.df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap")
        return fig.to_json()

    @tool
    def stacked_bar_plot(self, input: ColumnPairInput) -> str:
        """Stacked bar plot for two categorical columns."""
        grouped = self.df.groupby([input.x_column, input.y_column]).size().reset_index(name='count')
        fig = px.bar(grouped, x=input.x_column, y="count", color=input.y_column, title=f"Stacked Bar: {input.x_column} vs {input.y_column}")
        return fig.to_json()

    # ---------- Expose All Tools ----------

    def get_tools(self):
        return [
            self.histogram,
            self.box_plot,
            self.bar_plot,
            self.pie_chart,
            self.violin_plot_univariate,
            self.scatter_plot,
            self.line_plot,
            self.box_plot_bivariate,
            self.violin_plot_bivariate,
            self.bar_grouped_plot,
            self.correlation_heatmap,
            self.stacked_bar_plot
        ]