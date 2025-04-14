import pandas as pd
from typing import Dict, Any

class VizAgent:

    def suggest_visual_columns(self, df: pd.DataFrame) -> str:
            summary = df.describe(include='all', datetime_is_numeric=True).to_string()
            prompt = f"""
You are a data visualization expert.

Here is a summary of the dataset:
{summary}

Your task is to suggest 2 to 4 meaningful visualizations** that can help explore this dataset.

For each suggestion, return:
- Column(s) to use (1 or 2)
- Type of visualization (e.g., line plot, box plot, histogram, bar chart, scatter plot, time series, box plot, heatmap etc.)
- A brief reason why this chart is useful

Respond in JSON format like this:

[
  {{
    "columns": ["age"],
    "chart_type": "histogram",
    "reason": "To understand the distribution of customer ages."
  }},
  {{
    "columns": ["region", "sales"],
    "chart_type": "bar chart",
    "reason": "To compare sales across regions."
  }}
]

Only include valid chart types and avoid suggesting saving files or using specific libraries.
"""
            return prompt