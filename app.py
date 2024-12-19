import streamlit as st
import pandas as pd
import os
import base64

# Import evaluation modules
from phoenix_code import phoenix_eval
from ragas_code import ragas_eval
from traditional_metrics_score import RAGEvaluator

# Set page configuration
st.set_page_config(
    page_title="RAG Evaluation Toolkit",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function to create a more visually appealing file uploader
def custom_file_uploader():
    st.markdown("""
    <div class="file-upload-container">
        <div class="file-upload-icon">📂</div>
        <div class="file-upload-text">
            Drag and Drop or <span class="file-upload-browse">Browse Files</span>
        </div>
        <small>Supports CSV, XLS, XLSX</small>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Dataset", 
        type=["csv", "xls", "xlsx"], 
        label_visibility="collapsed"
    )
    return uploaded_file

# Main Streamlit App
def main():
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stTitle {
        color: #2C3E50;
        text-align: center;
        margin-bottom: 30px;
    }
    .stMarkdown {
        color: #34495E;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980B9;
        transform: scale(1.05);
    }
    .sidebar .sidebar-content {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .file-upload-container {
        border: 2px dashed #3498DB;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background-color: #FFFFFF;
        transition: all 0.3s ease;
    }
    .file-upload-container:hover {
        border-color: #2980B9;
        background-color: #F1F8FF;
    }
    .file-upload-icon {
        font-size: 50px;
        color: #3498DB;
        margin-bottom: 15px;
    }
    .file-upload-text {
        color: #2C3E50;
        font-size: 18px;
    }
    .file-upload-browse {
        color: #3498DB;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # App Title
    st.markdown("<h1 class='stTitle'>🔍 RAG Evaluation Toolkit</h1>", unsafe_allow_html=True)
    
    # Sidebar for Configuration
    st.sidebar.header("📋 Evaluation Configuration")
    
    # API Key Input with improved styling
    st.sidebar.subheader("OpenAI API Key")
    openai_api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key", 
        type="password", 
        help="Required for running evaluations"
    )
    
    # File Upload Section
    st.markdown("### 📊 Upload Your Dataset")
    uploaded_file = custom_file_uploader()
    
    # Evaluation Type Selection
    st.sidebar.subheader("🛠 Evaluation Methods")
    evaluation_methods = {
        "Phoenix Evaluation": [
            "hallucination", 
            "toxicity", 
            "relevance", 
            "Q&A"
        ],
        "RAGAS Evaluation": [
            "answer_correctness", 
            "answer_relevancy", 
            "faithfulness", 
            "context_precision", 
            "context_recall", 
            "context_relevancy", 
            "answer_similarity"
        ],
        "Traditional Metrics": [
            "BLEU", 
            "ROUGE-1", 
            "BERT Score", 
            "Perplexity", 
            "Diversity", 
            "Racial Bias"
        ]
    }
    
    # Multiselect for each evaluation method
    selected_metrics = {}
    for method, metrics in evaluation_methods.items():
        if st.sidebar.checkbox(method):
            selected_metrics[method] = st.sidebar.multiselect(
                f"Select {method} Metrics", 
                metrics
            )
    
    # Evaluation Button
    if uploaded_file and openai_api_key and selected_metrics:
        if st.button("🚀 Run Evaluation"):
            # Load data
            file_extension = os.path.splitext(uploaded_file.name)[1]
            if file_extension.lower() == ".csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension.lower() in [".xls", ".xlsx"]:
                df = pd.read_excel(uploaded_file)
            
            # Combine results
            combined_results = pd.DataFrame()
            
            # Progress bar
            progress_bar = st.progress(0)
            
            # Run evaluations
            with st.spinner("Processing evaluations..."):
                # Phoenix Evaluation
                if "Phoenix Evaluation" in selected_metrics:
                    progress_bar.progress(33)
                    phoenix_results = phoenix_eval(
                        selected_metrics.get("Phoenix Evaluation", []), 
                        openai_api_key, 
                        df.copy()
                    )
                    combined_results = pd.concat([combined_results, phoenix_results], axis=1)
                
                # RAGAS Evaluation
                if "RAGAS Evaluation" in selected_metrics:
                    progress_bar.progress(66)
                    ragas_results = ragas_eval(
                        selected_metrics.get("RAGAS Evaluation", []), 
                        openai_api_key, 
                        df.copy()
                    )
                    combined_results = pd.concat([combined_results, ragas_results], axis=1)
                
                # Traditional Metrics Evaluation
                if "Traditional Metrics" in selected_metrics:
                    progress_bar.progress(100)
                    traditional_results = RAGEvaluator(
                        df=df.copy(), 
                        selected_metrics=selected_metrics.get("Traditional Metrics", [])
                    )
                    combined_results = pd.concat([combined_results, traditional_results], axis=1)
                
                # Save results
                results_filename = "rag_evaluation_results.xlsx"
                combined_results.to_excel(results_filename, index=False)
                
                # Success message and download button
                st.success("Evaluation Completed Successfully!")
                
                # Create download button with improved styling
                with open(results_filename, "rb") as file:
                    btn = st.download_button(
                        label="📥 Download Evaluation Results",
                        data=file,
                        file_name=results_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Display results preview
                st.markdown("### 📊 Results Preview")
                st.dataframe(combined_results)

# Run the app
if __name__ == "__main__":
    main()