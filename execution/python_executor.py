"""
Python Executor - Safely executes LLM-generated Python code and captures outputs.
"""

import sys
import io
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def execute_analysis_code(code: str, df: pd.DataFrame) -> dict:
    """
    Execute generated Python analysis code in a sandboxed namespace.

    Args:
        code: Python code string to execute.
        df: The user's DataFrame, injected into the execution namespace.

    Returns:
        dict with keys:
            - success (bool)
            - output (str): captured stdout
            - error (str): error message if failed
            - figures (list): list of matplotlib Figure objects
            - dataframes (list): list of any DataFrames created during execution
    """
    result = {
        "success": False,
        "output": "",
        "error": "",
        "figures": [],
        "dataframes": [],
    }

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    # Close any existing plots
    plt.close("all")

    # Build execution namespace with common libraries pre-imported
    namespace = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "print": print,
        "__builtins__": __builtins__,
    }

    try:
        exec(code, namespace)  # noqa: S102
        result["success"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    result["output"] = captured_output.getvalue()

    # Collect all open figures
    fig_nums = plt.get_fignums()
    for num in fig_nums:
        fig = plt.figure(num)
        result["figures"].append(fig)

    # Collect any DataFrames that were created or modified in the namespace
    for key, val in namespace.items():
        if key.startswith("_") or key in ("df", "pd", "np", "plt", "sns", "print"):
            continue
        if isinstance(val, pd.DataFrame) and not val.empty:
            result["dataframes"].append((key, val))

    return result


def clean_code(raw_code: str) -> str:
    """
    Strip markdown code fences and leading/trailing whitespace from LLM output.

    Args:
        raw_code: Raw string from LLM that may contain ```python ... ``` fences.

    Returns:
        Clean executable Python code string.
    """
    code = raw_code.strip()

    # Remove markdown code fences
    if code.startswith("```"):
        lines = code.split("\n")
        # Drop first line (```python or ```)
        lines = lines[1:]
        # Drop last line if it's just closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    return code.strip()
