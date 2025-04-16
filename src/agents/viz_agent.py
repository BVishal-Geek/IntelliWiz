import pandas as pd
from typing import Dict, Any

class VizAgent:

    def suggest_visual_columns(self, df: pd.DataFrame) -> str:
      summary = df.describe(include='all', datetime_is_numeric=True).to_string()
      columns = ', '.join(df.columns)

      prompt = f'''
You are a data visualization expert.

Here is a summary of the dataset:
{summary}

Here are the columns in the dataset:
{columns}

Your task is to suggest 2 to 4 meaningful visualizations to help explore this dataset.

Please return your response in the following JSON format, exactly:

{{
  "tool_calls": [
    {{
      "name": "scatter_plot",
      "arguments": {{
        "input": {{
          "x_column": "Age",
          "y_column": "Salary"
        }}
      }}
    }},
    {{
      "name": "box_plot_bivariate",
      "arguments": {{
        "input": {{
          "x_column": "Gender",
          "y_column": "Salary"
        }}
      }}
    }}
  ]
}}

### Rules ###
- Only use from this list of tool names: 
  scatter_plot, histogram, bar_plot, pie_chart, box_plot, violin_plot_univariate, violin_plot_bivariate, box_plot_bivariate, bar_grouped_plot, line_plot
- Arguments must exactly match the tool definitions:
  - Use `"column"` for univariate plots.
  - Use `"x_column"` and `"y_column"` for bivariate plots.
- Only one value for `"y_column"` — never a list.
- Ensure that column names are present in the dataset and have the expected types.
- Do not include extra text — just return the JSON block.
- Avoid suggesting saving files or using specific libraries.
'''

      return prompt