# IntelliWiz: AI-Powered Data Analysis Platform

IntelliWiz is a smart platform that automates data cleaning and visualizations using AI agents. It helps data analysts save time by automating the most tedious parts of data analysis.

## Problem Statement

- 80% of a data analyst's time is spent on data cleaning, preparation, and exploratory data analysis
- Manual data cleaning is tedious, error-prone, and doesn't scale
- Existing tools often require manual intervention at multiple stages

## Solution

IntelliWiz leverages AI agents to automate:
- Data profiling and analysis
- Intelligent data cleaning and preprocessing
- Visualization selection and generation
- Before/after comparisons of cleaned data

## 🔗 Live Demo

- Main Application: [https://intelliwiz.streamlit.app/](https://intelliwiz.streamlit.app/)
- Report Application: [https://intelliwiz-report.streamlit.app/](https://intelliwiz-report.streamlit.app/)

## 🔍 Features

- **AI-powered data cleaning** - Automatically handles missing values, outliers, and data type issues
- **Intelligent visualization generation** - Creates relevant visualizations based on dataset characteristics
- **Comparative analysis** - Shows before/after cleaning analysis to understand data quality improvements
- **Debug mode** - Provides detailed logs and insights into the AI decision-making process

## 🛠️ Technologies Used

- **Frontend:** Streamlit (Python)
- **Data Processing:** Pandas, NumPy
- **AI Integration:** Llama API (Llama 3.1)
- **Visualization:** Plotly Express, Graph Objects
- **Workflow Orchestration:** LangGraph

## 📋 Components

- **VisualizationAgent**: Creates meaningful visualizations from dataframes
- **DataCleaningAgent**: Handles data preprocessing including missing values, outliers, type conversion
- **AnalysisAgent**: Prepares comprehensive prompts for data analysis
- **LangGraph Workflow**: Orchestrates the entire process in a defined sequence

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/BVishal-Geek/IntelliWiz.git
cd IntelliWiz
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Enter your Llama API key in the application interface when prompted

## 💡 How It Works

1. **Data Ingestion**: Upload CSV files via drag-and-drop
2. **Original Data Analysis**: AI agents profile and analyze the original data
3. **Original Data Visualization**: AI selects and creates appropriate visualizations
4. **Data Cleaning**: AI identifies and fixes data quality issues
5. **Cleaned Data Analysis and Visualization**: AI analyzes and visualizes the improved data
6. **Comparative Analysis**: Compare before/after results to see improvements

## 📊 Workflow Architecture

The application uses LangGraph to orchestrate the workflow:
![IntelliWiz LangGraph Workflow](workflow_diagram.png)

1. `analyze_original_data_node` - Run AI-powered analysis on the original dataset
2. `visualize_original_data_node` - Create visualizations based on original dataset
3. `clean_data_node` - Clean the dataset using AI-powered data cleaning
4. `visualize_cleaned_data_node` - Create visualizations for cleaned dataset
5. `analyze_cleaned_data_node` - Run AI-powered analysis on cleaned dataset

## 🔮 Future Extensions

- **Human-in-the-Loop**: Implement feedback mechanisms allowing users to validate, correct, and refine AI-driven decisions
- **Memory-Enhanced Workflow**: Add persistent memory capabilities to retain context across sessions
- **LLM-Agnostic Architecture**: Refactor the system to work with multiple LLM providers
- **Agentic AI for Data Engineering and ML Modeling**: Develop specialized AI agents for more advanced tasks

## 👨‍💻 Contributors

- Yash Kattimani
- Swathi Murali Srinivasan
- Vishal Bakshi

## 📝 Repository

GitHub: [https://github.com/BVishal-Geek/IntelliWiz](https://github.com/BVishal-Geek/IntelliWiz)

---

*Note: This project requires a Llama API key to function. You'll need to enter your API key in the application interface after launching the app.*

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39291651/89750847-9847-44fa-8023-5072b53df076/AI-Powered-Data-Analysis-Platform_-Comprehensive-Documentation-2.pptx
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39291651/b3f8c1f7-92b8-4ca3-bd8a-1cb36490356e/analysis_agent.py
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39291651/584152eb-b90a-46eb-a55d-ab9c17875cfc/visualization_agent.py
[4] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39291651/0cd5c557-5bdc-456d-a49f-cbabb937602d/workflow.py
[5] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39291651/a0ad170b-d3b4-4910-9b41-664f88d4ae1b/app.py
[6] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39291651/9897490d-c03f-43d4-b262-47795ab67ba2/ai_visualization_agent.py
[7] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39291651/ccb4390e-70a4-4642-8cc5-fc0e980ed659/data_cleaning_agent.py

---
Answer from Perplexity: pplx.ai/share
