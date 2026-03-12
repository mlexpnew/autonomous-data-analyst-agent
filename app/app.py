"""
Autonomous Data Analyst Agent - Streamlit Application
Run with: streamlit run app/app.py
"""

import sys
import os

import os
import warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# Ensure project root is on the path regardless of where streamlit is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from agents.data_analyst_agent import DataAnalystAgent
from utils.dataset_summary import get_dataset_summary, get_column_info_table
from analysis.data_analysis_tools import detect_outliers, correlation_analysis
from visualization.chart_generator import correlation_heatmap

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autonomous Data Analyst Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

## ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>

    /* Main title gradient */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    /* Subtitle */
    .sub-title {
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #60a5fa;
        border-left: 4px solid #3b82f6;
        padding-left: 10px;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* AI insights / explanation card */
    .insight-box {
        background: linear-gradient(180deg,#0f172a,#020617);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #1e293b;
        margin-top: 1rem;
        color: #e5e7eb;
        line-height: 1.6;
        font-size: 15px;
    }

    /* Metric cards */
    .metric-card {
        background: #0f172a;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        border: 1px solid #1e293b;
    }

    </style>
    """,
    unsafe_allow_html=True,
)



# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Get your free key at https://aistudio.google.com/app/apikey",
    )
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        st.success("API key set ✅")
    else:
        st.warning("Enter your Gemini API key to enable AI analysis.")

    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "Show top 10 values by frequency",
        "Find trends over time",
        "Which category has the highest total?",
        "Detect correlations between variables",
        "Show the distribution of numeric columns",
        "Identify anomalies or outliers",
        "Create a summary statistics table",
        "Compare groups using a bar chart",
    ]
    for q in example_questions:
        st.markdown(f"• {q}")

    st.markdown("---")
    st.markdown("### 📋 About")
    st.info(
        "Autonomous Data Analyst Agent powered by **Google Gemini 1.5 Flash**. "
        "Upload a dataset, ask questions in plain English, and get instant "
        "Python analysis, visualizations, and business insights."
    )


# ── Main title ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🤖 Autonomous Data Analyst Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-powered data analysis · Visualizations · Business Insights · Powered by Google Gemini 1.5 Flash</div>',
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of {question, result} dicts
if "agent" not in st.session_state:
    st.session_state.agent = None

# ── Section 1: Dataset Upload ─────────────────────────────────────────────────
st.markdown('<div class="section-header">📂 Upload Your Dataset</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, Excel (.xlsx, .xls)",
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.df = df
        st.session_state.summary = get_dataset_summary(df)
        st.success(f"✅ Dataset loaded: **{uploaded_file.name}** ({df.shape[0]:,} rows × {df.shape[1]} columns)")
    except Exception as e:
        st.error(f"Error loading file: {e}")

# ── Section 2: Dataset Preview ────────────────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df
    summary = st.session_state.summary

    st.markdown('<div class="section-header">🔍 Dataset Overview</div>', unsafe_allow_html=True)

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{summary['n_rows']:,}")
    col2.metric("Columns", summary["n_cols"])
    col3.metric("Missing Values", sum(summary["missing_values"].values()) if summary["missing_values"] else 0)
    col4.metric("Duplicate Rows", summary["duplicate_rows"])

    tab1, tab2, tab3, tab4 = st.tabs(["📄 Preview", "📊 Column Info", "📈 Statistics", "🔎 Outliers"])

    with tab1:
        st.dataframe(df.head(50), use_container_width=True)

    with tab2:
        col_info = get_column_info_table(df)
        st.dataframe(col_info, use_container_width=True)

    with tab3:
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe().round(4), use_container_width=True)

            st.markdown("#### Correlation Heatmap")
            if numeric_df.shape[1] >= 2:
                fig = correlation_heatmap(df)
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No numeric columns found for statistics.")

    with tab4:
        outliers = detect_outliers(df)
        if outliers:
            for col, info in outliers.items():
                with st.expander(f"⚠️ {col} — {info['count']} outliers"):
                    st.write(f"**Bounds:** [{info['lower_bound']}, {info['upper_bound']}]")
                    st.write(f"**Sample outlier values:** {info['sample']}")
        else:
            st.success("No significant outliers detected.")

    # ── Section 3: AI Dataset Explanation ─────────────────────────────────────
    st.markdown('<div class="section-header">🧠 AI Dataset Overview</div>', unsafe_allow_html=True)
    if st.button("🔍 Generate AI Dataset Explanation", use_container_width=True):
        if not os.environ.get("GEMINI_API_KEY"):
            st.error("Please enter your Gemini API key in the sidebar.")
        else:
            with st.spinner("Gemini is analyzing your dataset..."):
                try:
                    if st.session_state.agent is None:
                        st.session_state.agent = DataAnalystAgent()
                    explanation = st.session_state.agent.explain_dataset(df)
                    st.markdown(
                        f'<div class="insight-box">{explanation}</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Error generating explanation: {e}")

    # ── Section 4: Ask a Question ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">💬 Ask the Data Analyst Agent</div>', unsafe_allow_html=True)
    st.markdown("Ask any data analysis question in plain English. The agent will write Python code, execute it, create charts, and generate insights.")

    question = st.text_area(
        "Your question:",
        placeholder="e.g. Show top 10 companies by frequency, or find trends in revenue over time...",
        height=80,
    )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        run_analysis = st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    with col_b:
        get_recs = st.button("📋 Get Recommendations", use_container_width=True)

    # ── Run Analysis ──────────────────────────────────────────────────────────
    if run_analysis:
        if not question.strip():
            st.warning("Please enter a question first.")
        elif not os.environ.get("GEMINI_API_KEY"):
            st.error("Please enter your Gemini API key in the sidebar.")
        else:
            with st.spinner("🤖 Agent is thinking and writing code..."):
                try:
                    if st.session_state.agent is None:
                        st.session_state.agent = DataAnalystAgent()

                    result = st.session_state.agent.analyze(question, df)

                    # Store in history
                    st.session_state.history.append({"question": question, "result": result})

                except Exception as e:
                    st.error(f"Agent error: {e}")
                    result = None

    # ── Get Recommendations ───────────────────────────────────────────────────
    if get_recs:
        if not os.environ.get("GEMINI_API_KEY"):
            st.error("Please enter your Gemini API key in the sidebar.")
        else:
            last_output = ""
            if st.session_state.history:
                last = st.session_state.history[-1]
                last_output = last["result"].get("output", "") + "\n" + last["result"].get("insights", "")

            with st.spinner("Generating business recommendations..."):
                try:
                    if st.session_state.agent is None:
                        st.session_state.agent = DataAnalystAgent()
                    recs = st.session_state.agent.generate_recommendations(df, last_output)
                    st.markdown('<div class="section-header">📋 Business Recommendations</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="insight-box">{recs}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")

    # ── Display Results ───────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)

        # Show most recent result first
        for entry in reversed(st.session_state.history):
            q = entry["question"]
            r = entry["result"]

            with st.expander(f"❓ {q}", expanded=(entry == st.session_state.history[-1])):

                # Generated code
                st.markdown("#### 🐍 Generated Python Code")
                st.code(r["code"], language="python")

                # Execution status
                if r["success"]:
                    st.success("✅ Code executed successfully")
                else:
                    st.error(f"❌ Execution error:\n{r['error']}")

                # Printed output
                if r["output"]:
                    st.markdown("#### 📝 Analysis Output")
                    st.text(r["output"])

                # Charts
                if r["figures"]:
                    st.markdown("#### 📊 Visualizations")
                    for fig in r["figures"]:
                        st.pyplot(fig)
                        plt.close(fig)

                # DataFrames produced during execution
                if r["dataframes"]:
                    st.markdown("#### 📋 Result Tables")
                    for name, result_df in r["dataframes"]:
                        st.markdown(f"**{name}**")
                        st.dataframe(result_df, use_container_width=True)

                # Insights
                if r["insights"]:
                    st.markdown("#### 💡 AI Insights")
                    st.markdown(
                        f'<div class="insight-box">{r["insights"]}</div>',
                        unsafe_allow_html=True,
                    )

        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

else:
    # Landing state – no file uploaded yet
    st.info("👆 Upload a CSV or Excel dataset above to get started.")

    st.markdown("---")
    st.markdown("### 🚀 What This Agent Can Do")
    capabilities = [
        ("🔍 Data Understanding", "Automatically detects column types, missing values, duplicates, and outliers."),
        ("🧹 Data Cleaning", "Handles missing values, removes duplicates, and fixes data types."),
        ("📊 Statistical Analysis", "Runs groupby, aggregations, frequency analysis, and descriptive statistics."),
        ("📈 Visualizations", "Generates bar charts, line charts, histograms, scatter plots, and heatmaps."),
        ("💡 AI Insights", "Explains what the analysis means in plain business language."),
        ("📋 Recommendations", "Provides data-driven business recommendations."),
    ]
    cols = st.columns(3)
    for i, (title, desc) in enumerate(capabilities):
        with cols[i % 3]:
            st.markdown(f"**{title}**")
            st.markdown(desc)
            st.markdown("")
