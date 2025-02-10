import os
from llamaapi import LlamaAPI

class AnalysisAgent:
    def __init__(self, api_key):
        self.llama = LlamaAPI(api_key)

    def analyze_data(self, data_description):
        prompt = f"""As a data analysis expert, analyze this dataset and provide insights:

{data_description}

Please provide a comprehensive analysis including:
1. Key patterns and trends in the data
2. Statistical insights from numerical features
3. Distribution patterns in categorical features
4. Notable relationships between features
5. Data quality observations
6. Suggestions for further analysis

Structure your response clearly and focus on actionable insights."""

        api_request = {
            "model": "llama3.1-70b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a data analysis expert. Provide clear, actionable insights from data analysis."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }

        try:
            response = self.llama.run(api_request)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")