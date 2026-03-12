"""
Data Analyst Agent - Orchestrates LLM prompting, code generation,
execution, and insight generation.
"""

import pandas as pd
from llm.gemini_client import get_gemini_client, call_gemini
from execution.python_executor import execute_analysis_code, clean_code
from utils.dataset_summary import get_dataset_summary


# ── System prompt injected into every analysis request ─────────────────────
SYSTEM_CONTEXT = """
You are an expert data analyst and Python developer. 
Your job is to help users analyze datasets by:
1. Writing clean, executable Python code using pandas, numpy, matplotlib, and seaborn.
2. The DataFrame is already loaded as `df`. Do NOT re-load or re-create it.
3. Always use plt.figure() or plt.subplots() when creating charts so they display correctly.
4. Print key results using print() so they appear in the output.
5. Never use plt.show() — Streamlit captures figures automatically.
6. Write production-quality code with comments.
7. After the code, always provide a brief insight block starting with "## Insights:" 
   that explains what the analysis reveals in plain business language.
"""


class DataAnalystAgent:
    """
    Autonomous agent that takes a user question + DataFrame,
    generates analysis code with Gemini, executes it, and returns
    structured results including figures, printed output, and AI insights.
    """

    def __init__(self):
        self.model = get_gemini_client()

    def analyze(self, question: str, df: pd.DataFrame) -> dict:
        """
        Full analysis pipeline:
          1. Build context prompt
          2. Ask Gemini for Python code + insights
          3. Parse and execute the code
          4. Return structured results

        Args:
            question: Natural language question from the user.
            df: The user's DataFrame.

        Returns:
            dict with keys: code, output, figures, dataframes,
                            insights, raw_response, success, error
        """
        dataset_summary = get_dataset_summary(df)
        prompt = self._build_prompt(question, dataset_summary)
        raw_response = call_gemini(prompt, self.model)

        # Split code block from insights block
        code_str, insights_str = self._parse_response(raw_response)
        clean = clean_code(code_str)

        # Execute
        exec_result = execute_analysis_code(clean, df)

        return {
            "code": clean,
            "output": exec_result["output"],
            "figures": exec_result["figures"],
            "dataframes": exec_result["dataframes"],
            "insights": insights_str,
            "raw_response": raw_response,
            "success": exec_result["success"],
            "error": exec_result["error"],
        }

    def generate_recommendations(self, df: pd.DataFrame, analysis_summary: str) -> str:
        """
        Ask Gemini to generate business recommendations based on the analysis.

        Args:
            df: Original DataFrame.
            analysis_summary: Text summary of what was found in the analysis.

        Returns:
            Recommendation text from Gemini.
        """
        dataset_summary = get_dataset_summary(df)
        prompt = f"""
{SYSTEM_CONTEXT}

Dataset Overview:
{dataset_summary['text_description']}

Analysis findings:
{analysis_summary}

Based on the dataset and the analysis findings above, provide 3-5 concrete, 
actionable business recommendations. Format each recommendation as:
- **Recommendation**: [title]
  **Rationale**: [why this matters based on the data]
  **Action**: [what the business should do]
"""
        return call_gemini(prompt, self.model)

    def explain_dataset(self, df: pd.DataFrame) -> str:
        """
        Ask Gemini to provide a plain-language overview of the dataset.
        """
        dataset_summary = get_dataset_summary(df)
        prompt = f"""
{SYSTEM_CONTEXT}

You have been given a dataset with the following properties:

{dataset_summary['text_description']}

In 3-5 paragraphs, explain:
1. What this dataset appears to contain (domain/topic).
2. Key patterns or characteristics visible in the summary statistics.
3. Data quality observations (missing values, duplicates, outliers risk).
4. What types of analyses would be most valuable for this dataset.

Write in a professional but approachable tone, as if briefing a business stakeholder.
"""
        return call_gemini(prompt, self.model)

    # ── Private helpers ────────────────────────────────────────────────────

    def _build_prompt(self, question: str, dataset_summary: dict) -> str:
        """Construct the full prompt sent to Gemini."""
        return f"""
{SYSTEM_CONTEXT}

=== DATASET OVERVIEW ===
{dataset_summary['text_description']}

=== USER QUESTION ===
{question}

=== YOUR TASK ===
1. Write Python code to answer the question above.
   - The DataFrame is already available as `df`.
   - Import only from: pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns).
   - Create clear, labeled charts.
   - Print important numeric results.
   - Do NOT call plt.show().

2. After the code block, write a section starting with:
   ## Insights:
   Explain what the results mean in 3-5 bullet points using plain English for a business audience.

Return ONLY the code block (inside ```python ... ```) followed by the ## Insights: section.
"""

    def _parse_response(self, raw: str) -> tuple[str, str]:
        """
        Split the raw Gemini response into (code, insights).
        Returns the code block and everything after ## Insights:.
        """
        insights = ""
        code = raw

        if "## Insights:" in raw:
            parts = raw.split("## Insights:", 1)
            code = parts[0].strip()
            insights = "## Insights:\n" + parts[1].strip()
        elif "## insights:" in raw.lower():
            idx = raw.lower().index("## insights:")
            code = raw[:idx].strip()
            insights = raw[idx:].strip()

        return code, insights
