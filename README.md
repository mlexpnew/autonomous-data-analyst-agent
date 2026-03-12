# 🤖 Autonomous Data Analyst Agent

An AI-powered data analysis system built with **Google Gemini 1.5 Flash** and **Streamlit** that performs end-to-end data analysis automatically — from data exploration to business insight generation.

---

## ✨ Features

| Capability | Description |
|---|---|
| 🔍 **Data Understanding** | Auto-detects column types, missing values, duplicates, and outliers |
| 🧹 **Data Cleaning** | Handles missing values, removes duplicates, fixes data types |
| 📊 **Statistical Analysis** | Groupby, aggregations, frequency analysis, descriptive statistics |
| 📈 **Visualizations** | Bar, line, histogram, scatter, correlation heatmap |
| 🐍 **Code Generation** | Gemini writes and executes Python analysis code automatically |
| 💡 **AI Insights** | Plain-English explanations of what results mean for the business |
| 📋 **Recommendations** | Actionable, data-driven business recommendations |

---

## 🗂️ Project Structure

```
autonomous-data-analyst-agent/
├── app/
│   └── app.py                   # Main Streamlit application
├── agents/
│   └── data_analyst_agent.py    # Core AI agent orchestration
├── llm/
│   └── gemini_client.py         # Google Gemini 1.5 Flash client
├── execution/
│   └── python_executor.py       # Safe code execution engine
├── analysis/
│   └── data_analysis_tools.py   # Analytical helper functions
├── visualization/
│   └── chart_generator.py       # Chart generation functions
├── utils/
│   └── dataset_summary.py       # Dataset summarization utilities
├── sample_business_data.csv     # Example dataset for testing
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone / Download the Project

```bash
cd autonomous-data-analyst-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a Free Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key

### 5. Configure API Key

**Option A — .env file (recommended):**
```bash
cp .env.example .env
# Edit .env and add your key:
# GEMINI_API_KEY=your_key_here
```

**Option B — Enter in the app sidebar:**
You can also paste your key directly into the sidebar when the app is running.

### 6. Run the Application

```bash
streamlit run app/app.py
```

Open your browser to: **http://localhost:8501**

---

## 🧪 Testing with Sample Data

A sample dataset (`sample_business_data.csv`) is included with 500 rows of business data:

| Column | Type | Description |
|---|---|---|
| Company | String | Company name |
| Category | String | Business category |
| Region | String | Geographic region |
| Year | Integer | Fiscal year (2020–2024) |
| Revenue_USD_M | Float | Revenue in millions USD |
| Profit_USD_M | Float | Profit in millions USD |
| Employees | Integer | Number of employees |
| Customer_Count | Integer | Number of customers |
| Growth_Rate_Pct | Float | Revenue growth rate % |
| Customer_Satisfaction | Float | Satisfaction score 1–5 |
| Marketing_Spend_USD_M | Float | Marketing spend in millions USD |

### Example Questions to Try

```
Show top companies by total revenue
Find revenue trends over the years
Which category has the highest average profit?
Detect correlations between revenue and marketing spend
Show distribution of customer satisfaction scores
Compare regions by employee count
Identify companies with the fastest growth rate
Create a heatmap of numeric variable correlations
```

---

## 🔄 User Workflow

```
1. Upload CSV or Excel file
        ↓
2. View automatic dataset overview
   (preview, column info, statistics, outliers)
        ↓
3. Ask a question in plain English
        ↓
4. Agent generates Python code → executes it → renders charts
        ↓
5. View AI insights and business recommendations
```

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI/LLM**: Google Gemini 1.5 Flash (free tier)
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Orchestration**: LangChain (prompt management)

---

## ⚠️ Gemini Free Tier Limits

The free tier of Google Gemini 1.5 Flash includes:
- **15 requests per minute (RPM)**
- **1 million tokens per minute (TPM)**
- **1,500 requests per day**

This is more than sufficient for typical data analysis sessions.

---

## 🔒 Security Notes

- Generated code runs in a sandboxed Python namespace
- The DataFrame is passed by copy to prevent modification of original data
- No external network calls are made from generated code
- API keys are never logged or stored beyond the session

---

## 📄 License

MIT License — free to use and modify.
