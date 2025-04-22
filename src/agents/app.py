import streamlit as st
import pandas as pd
import os
import json
import plotly.graph_objects as go
from workflow import create_analysis_workflow
from fpdf import FPDF
import tempfile
import base64
import dotenv

@st.cache_data
def load_data(file):
    file_path = file  # Update path if needed
    return pd.read_csv(file_path)

def create_pdf_report(report_sections, plot_image_paths, filename="report.pdf"):
    """
    Create a PDF report with text sections and images
    Args:
        report_sections: List of text sections to include in the report
        plot_image_paths: List of paths to plot images to include
        filename: Name of the output PDF file
    Returns:
        PDF file as bytes
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add text sections
    for section in report_sections:
        pdf.multi_cell(0, 10, section)
        pdf.ln(5)

    # Add plots
    for img_path in plot_image_paths:
        pdf.add_page()
        pdf.image(img_path, x=10, y=20, w=180)
    
    # Save PDF to bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        tmpfile.seek(0)
        pdf_bytes = tmpfile.read()
    return pdf_bytes

def generate_report_pdf(results, original_viz_data, cleaned_viz_data, original_fallback_paths=None, cleaned_fallback_paths=None):
    """
    Generate a PDF report from analysis results
    Args:
        results: Analysis results dictionary
        original_viz_data: Original data visualizations
        cleaned_viz_data: Cleaned data visualizations
        original_fallback_paths: Paths to fallback visualizations for original data
        cleaned_fallback_paths: Paths to fallback visualizations for cleaned data
    Returns:
        PDF file as bytes
    """
    # Prepare report sections
    quality_issues = results["cleaning"].get('quality_issues', [])
    cleaning_actions = results["cleaning"].get('cleaning_actions', [])
    quality_issues_count = len(quality_issues) if isinstance(quality_issues, list) else 0
    cleaning_actions_count = len(cleaning_actions) if isinstance(cleaning_actions, list) else 0
    
    # Get shape information
    orig_shape = results["cleaning"]['original_data'].get('shape', (0, 0)) if 'original_data' in results["cleaning"] else (0, 0)
    clean_shape = results["cleaning"]['cleaned_data'].get('shape', (0, 0)) if 'cleaned_data' in results["cleaning"] else (0, 0)
    
    report_sections = [
        "Original Data Analysis:\n" + results["original_analysis"]["result"],
        "\nData Cleaning Results:",
        f"- Quality issues identified: {quality_issues_count}",
        f"- Cleaning actions performed: {cleaning_actions_count}"
    ]
    
    # Add cleaning actions if available
    if isinstance(cleaning_actions, list) and len(cleaning_actions) > 0:
        report_sections.append("\nTop Cleaning Actions:")
        for i, action in enumerate(cleaning_actions[:5]):
            if isinstance(action, dict):
                report_sections.append(f"- {action.get('column', 'dataset')}: {action.get('description', 'N/A')}")
            elif isinstance(action, str):
                report_sections.append(f"- {action}")
    
    # Add shape change information
    if orig_shape != clean_shape:
        report_sections.append(f"\nData shape changed from {orig_shape} to {clean_shape}")
    
    # Add cleaned data analysis
    report_sections.append("\nCleaned Data Analysis:\n" + results["cleaned_analysis"]["result"])
    
    # Clean up any existing plot files before generating new ones
    for filename in os.listdir('.'):
        if filename.startswith('original_plot_') or filename.startswith('cleaned_plot_') or filename.startswith('fallback_plot_'):
            try:
                os.remove(filename)
            except:
                pass
    
    # Save plots as images
    plot_image_paths = []
    
    # Process original plots if successful
    if original_viz_data and original_viz_data.get("success"):
        for i, plot_data in enumerate(original_viz_data.get("plots", [])):
            try:
                fig_dict = json.loads(plot_data["figure"])
                fig = go.Figure(fig_dict)
                img_path = f"original_plot_{i}.png"
                fig.write_image(img_path, width=800, height=400)
                plot_image_paths.append(img_path)
            except Exception as e:
                print(f"Error saving plot: {e}")
    # Add fallback plots for original data if available
    elif original_fallback_paths:
        plot_image_paths.extend(original_fallback_paths)
        
    # Process cleaned plots if successful
    if cleaned_viz_data and cleaned_viz_data.get("success"):
        for i, plot_data in enumerate(cleaned_viz_data.get("plots", [])):
            try:
                fig_dict = json.loads(plot_data["figure"])
                fig = go.Figure(fig_dict)
                img_path = f"cleaned_plot_{i}.png"
                fig.write_image(img_path, width=800, height=400)
                plot_image_paths.append(img_path)
            except Exception as e:
                print(f"Error saving plot: {e}")
    # Add fallback plots for cleaned data if available
    elif cleaned_fallback_paths:
        plot_image_paths.extend(cleaned_fallback_paths)
    
    # Generate PDF
    pdf_bytes = create_pdf_report(report_sections, plot_image_paths)
    
    # Clean up temporary image files
    for img_path in plot_image_paths:
        if os.path.exists(img_path):
            os.remove(img_path)
    
    return pdf_bytes

def display_visualizations(viz_data, section_title, debug_mode=False, key_prefix="viz"):
    """
    Helper function to display visualizations with error handling and debug info
    Args:
        viz_data: Visualization data from agent
        section_title: Title for the visualization section
        debug_mode: Whether to show debug information
        key_prefix: Prefix for unique Streamlit keys to avoid duplicate ID errors
    Returns:
        List of fallback plot paths if fallback was used, otherwise empty list
    """
    fallback_plot_paths = []
    
    if not viz_data:
        st.write(f"### {section_title}")
        st.info("No visualization data available")
        return fallback_plot_paths

    # Display debug info if enabled
    if debug_mode:
        with st.expander("🔍 Visualization Debug Info", expanded=False):
            if "debug_info" in viz_data:
                st.json(viz_data["debug_info"])
            st.write(f"Success: {viz_data.get('success', False)}")
            st.write(f"Fallback used: {viz_data.get('is_fallback', False)}")
            st.write(f"Number of plots: {len(viz_data.get('plots', []))}")

    # Check if visualizations were generated successfully
    if viz_data and viz_data.get("success"):
        st.write(f"### {section_title}")
        if viz_data.get("is_fallback", False):
            st.info("⚠️ Using fallback visualizations due to issues with AI-generated visualizations")
        if len(viz_data["plots"]) > 1:
            tab_titles = [plot_data.get("title", f"Visualization {i+1}") for i, plot_data in enumerate(viz_data["plots"])]
            tabs = st.tabs(tab_titles)
            for i, (tab, plot_data) in enumerate(zip(tabs, viz_data["plots"])):
                with tab:
                    try:
                        st.subheader(plot_data["title"])
                        fig_dict = json.loads(plot_data["figure"])
                        fig = go.Figure(fig_dict)
                        unique_key = f"{key_prefix}_multi_{i}_{plot_data.get('id', '')}"
                        st.plotly_chart(fig, use_container_width=True, key=unique_key)
                        if "insight" in plot_data and plot_data["insight"]:
                            st.write(plot_data["insight"])
                    except Exception as e:
                        st.error(f"Error displaying visualization: {str(e)}")
                        if debug_mode:
                            import traceback
                            st.code(traceback.format_exc())
        elif len(viz_data["plots"]) == 1:
            try:
                plot_data = viz_data["plots"][0]
                st.subheader(plot_data["title"])
                fig_dict = json.loads(plot_data["figure"])
                fig = go.Figure(fig_dict)
                unique_key = f"{key_prefix}_single_{plot_data.get('id', '')}"
                st.plotly_chart(fig, use_container_width=True, key=unique_key)
                if "insight" in plot_data and plot_data["insight"]:
                    st.write(plot_data["insight"])
            except Exception as e:
                st.error(f"Error displaying visualization: {str(e)}")
                if debug_mode:
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("No visualizations were generated.")
    else:
        st.write(f"### {section_title}")
        if viz_data and not viz_data.get("success"):
            if "error" in viz_data:
                st.error(f"❌ Visualization failed: {viz_data['error']}")
            else:
                st.error("❌ Failed to generate visualizations")
        
        st.info("Creating basic visualizations...")
        st.write("#### Data Summary")
        
        if "current_df" in st.session_state:
            df = st.session_state["current_df"]
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) > 0:
                st.write("Summary of numeric columns:")
                st.dataframe(df[numeric_cols].describe())
                
                # Create fallback visualizations as Plotly figures and save them
                for i, col in enumerate(numeric_cols[:3]):  # Limit to first 3 numeric columns
                    st.write(f"Distribution of {col}")
                    
                    # Create and display a Plotly figure instead of using st.bar_chart
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=df[col], nbinsx=20))
                    fig.update_layout(title=f"Distribution of {col}")
                    
                    # Display the figure in Streamlit
                    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_fallback_{i}")
                    
                    # Save the figure for PDF
                    img_path = f"fallback_plot_{key_prefix}_{i}.png"
                    fig.write_image(img_path, width=800, height=400)
                    fallback_plot_paths.append(img_path)
            else:
                st.info("No numeric columns available for visualization")
        else:
            st.info("No visualization data available")
    
    return fallback_plot_paths

def main():
    # Initialize session state for analysis results
    if "analysis_complete" not in st.session_state:
        st.session_state["analysis_complete"] = False
    
    if "analysis_results" not in st.session_state:
        st.session_state["analysis_results"] = None
    
    if "original_viz_data" not in st.session_state:
        st.session_state["original_viz_data"] = None
    
    if "cleaned_viz_data" not in st.session_state:
        st.session_state["cleaned_viz_data"] = None
        
    if "original_fallback_paths" not in st.session_state:
        st.session_state["original_fallback_paths"] = []
        
    if "cleaned_fallback_paths" not in st.session_state:
        st.session_state["cleaned_fallback_paths"] = []
    
    st.title("📊 AI-Powered Data Analysis")
    st.write("Using AI to analyze datasets and visualize key insights automatically")

    # Set debug mode in sidebar
    st.sidebar.title("Debug Options")
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=True)

    # API Key input in sidebar with password masking
    st.sidebar.title("API Configuration")
    api_key = st.sidebar.text_input("Enter LLAMA API Key", type="password")

    # Store API key in session state if not already there
    if "api_key" not in st.session_state and api_key:
        st.session_state["api_key"] = ""
    elif api_key:  # Update if user changes it
        st.session_state["api_key"] = api_key

    # Check API key
    if not api_key:
        st.sidebar.warning("⚠️ Please enter your LLAMA API Key to proceed")
        st.info("To use this application, you need to provide your LLAMA API Key in the sidebar.")
        st.stop()
    st.sidebar.success("✅ API Key entered")

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        # Load data
        try:
            df = load_data(uploaded_file)
            if df is None or len(df) == 0:
                st.error("⚠️ The uploaded file appears to be empty.")
                st.stop()
            if len(df.columns) == 0:
                st.error("⚠️ The uploaded file doesn't have any columns.")
                st.stop()
        except Exception as e:
            st.error(f"⚠️ Error loading file: {str(e)}")
            st.stop()

        # Show dataset preview
        st.write("### Data Preview")
        st.dataframe(df.head())
        st.write(f"Dataset has {len(df)} rows and {len(df.columns)} columns")

        # Display column info
        if debug_mode:
            st.write("### Column Information")
            col_info = {
                "Column": df.columns.tolist(),
                "Type": [str(df[col].dtype) for col in df.columns],
                "Non-Null Count": [df[col].count() for col in df.columns],
                "Null Count": [df[col].isna().sum() for col in df.columns],
                "Unique Values": [df[col].nunique() for col in df.columns]
            }
            st.dataframe(pd.DataFrame(col_info).set_index("Column"))

        # Analysis button - only runs the analysis and stores results in session state
        if st.button("Analyze Data"):
            with st.spinner("Analyzing data with AI..."):
                try:
                    # Create the analysis workflow with the dataframe and debug mode
                    analysis_runner = create_analysis_workflow(df, debug_mode=debug_mode, api_key=st.session_state["api_key"])
                    results = analysis_runner()
                    
                    # Store results in session state
                    st.session_state["analysis_results"] = results
                    st.session_state["original_viz_data"] = results["original_analysis"]["visualizations"]
                    st.session_state["cleaned_viz_data"] = results["cleaned_analysis"]["visualizations"]
                    st.session_state["analysis_complete"] = True
                    
                    if "cleaned_df" in results.get("cleaning", {}):
                        st.session_state["current_df"] = results["cleaning"]["cleaned_df"]
                    else:
                        st.session_state["current_df"] = df
                    
                    # We'll generate the PDF after displaying visualizations to include fallback plots if needed
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    import traceback
                    error_trace = traceback.format_exc()
                    if debug_mode:
                        st.code(error_trace)

        # Display results from session state if analysis is complete
        # This section will always run after page reloads (including after download button clicks)
        if st.session_state["analysis_complete"]:
            results = st.session_state["analysis_results"]
            
            # Display debug logs if in debug mode
            if debug_mode and "debug_logs" in results:
                with st.expander("🔍 Debug Logs", expanded=False):
                    for i, log in enumerate(results["debug_logs"]):
                        st.text(f"{i+1}. {log}")
            
            # SECTION 1: Original Data Analysis
            st.write("## Original Data Analysis")
            st.write(results["original_analysis"]["result"])
            original_fallback_paths = display_visualizations(
                st.session_state["original_viz_data"],
                "Original Data Visualizations",
                debug_mode,
                "original"
            )
            # Store fallback paths in session state
            if original_fallback_paths:
                st.session_state["original_fallback_paths"] = original_fallback_paths

            # SECTION 2: Data Cleaning Results
            cleaning_results = results["cleaning"]
            if cleaning_results:
                st.write("## Data Cleaning Results")
                quality_issues = cleaning_results.get('quality_issues', [])
                cleaning_actions = cleaning_results.get('cleaning_actions', [])
                quality_issues_count = len(quality_issues) if isinstance(quality_issues, list) else 0
                cleaning_actions_count = len(cleaning_actions) if isinstance(cleaning_actions, list) else 0
                st.write(f"- Quality issues identified: {quality_issues_count}")
                st.write(f"- Cleaning actions performed: {cleaning_actions_count}")
                if isinstance(cleaning_actions, list) and len(cleaning_actions) > 0:
                    st.write("### Top Cleaning Actions:")
                    count = 0
                    for action in cleaning_actions:
                        if isinstance(action, dict):
                            st.write(f"- {action.get('column', 'dataset')}: {action.get('description', 'N/A')}")
                        elif isinstance(action, str):
                            st.write(f"- {action}")
                        count += 1
                        if count >= 5:
                            break
                if 'original_data' in cleaning_results and 'cleaned_data' in cleaning_results:
                    orig_shape = cleaning_results['original_data'].get('shape', (0, 0))
                    clean_shape = cleaning_results['cleaned_data'].get('shape', (0, 0))
                    if orig_shape != clean_shape:
                        st.write(f"Data shape changed from {orig_shape} to {clean_shape}")
                if debug_mode and cleaning_results.get('warning_flags'):
                    st.warning("Cleaning Warnings:")
                    for warning in cleaning_results.get('warning_flags', []):
                        st.write(f"- {warning}")

            # SECTION 3: Cleaned Data Analysis
            st.write("## Cleaned Data Analysis")
            st.write(results["cleaned_analysis"]["result"])
            cleaned_fallback_paths = display_visualizations(
                st.session_state["cleaned_viz_data"],
                "Cleaned Data Visualizations",
                debug_mode,
                "cleaned"
            )
            # Store fallback paths in session state
            if cleaned_fallback_paths:
                st.session_state["cleaned_fallback_paths"] = cleaned_fallback_paths

            # Show preview of cleaned data
            if "current_df" in st.session_state:
                cleaned_df = st.session_state["current_df"]
                if cleaned_df is not None and len(cleaned_df) > 0:
                    st.write("## Cleaned Data Preview")
                    st.dataframe(cleaned_df.head())
                    st.write(f"Cleaned dataset has {len(cleaned_df)} rows and {len(cleaned_df.columns)} columns")
                else:
                    st.warning("Cleaned data could not be displayed.")
            
            # Generate or update PDF with fallback visualizations if needed
            if "pdf_report" not in st.session_state or st.session_state["original_fallback_paths"] or st.session_state["cleaned_fallback_paths"]:
                pdf_bytes = generate_report_pdf(
                    st.session_state["analysis_results"],
                    st.session_state["original_viz_data"],
                    st.session_state["cleaned_viz_data"],
                    st.session_state.get("original_fallback_paths", []),
                    st.session_state.get("cleaned_fallback_paths", [])
                )
                st.session_state["pdf_report"] = pdf_bytes
            
            # Always show download button if analysis is complete
            if "pdf_report" in st.session_state:
                st.download_button(
                    label="📄 Download Report",
                    data=st.session_state["pdf_report"],
                    file_name="analysis_report.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
