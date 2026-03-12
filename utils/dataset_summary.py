"""
Dataset Summary Utilities - Generates structured summaries of uploaded datasets.
"""

import pandas as pd
import numpy as np


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive summary of a DataFrame.

    Returns a dict with shape, dtypes, missing values, duplicates,
    numeric stats, and a text description for the LLM prompt.
    """
    summary = {}

    # Basic shape
    summary["shape"] = df.shape
    summary["n_rows"] = df.shape[0]
    summary["n_cols"] = df.shape[1]

    # Column types
    summary["columns"] = list(df.columns)
    summary["dtypes"] = df.dtypes.astype(str).to_dict()

    # Numeric vs categorical split
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    summary["numeric_columns"] = numeric_cols
    summary["categorical_columns"] = categorical_cols

    # Missing values
    missing = df.isnull().sum()
    summary["missing_values"] = missing[missing > 0].to_dict()
    summary["missing_percent"] = (
        (df.isnull().sum() / len(df) * 100)
        .round(2)
        .to_dict()
    )

    # Duplicates
    summary["duplicate_rows"] = int(df.duplicated().sum())

    # Numeric stats
    if numeric_cols:
        desc = df[numeric_cols].describe().round(4)
        summary["numeric_stats"] = desc.to_dict()
    else:
        summary["numeric_stats"] = {}

    # Categorical value counts (top 5 per column)
    cat_info = {}
    for col in categorical_cols[:10]:  # limit to avoid huge prompts
        vc = df[col].value_counts().head(5).to_dict()
        cat_info[col] = vc
    summary["categorical_top_values"] = cat_info

    # Sample rows
    summary["sample_rows"] = df.head(5).to_string()

    # Build a compact text description for the LLM
    summary["text_description"] = _build_text_description(summary, df)

    return summary


def _build_text_description(summary: dict, df: pd.DataFrame) -> str:
    """Create a concise text description of the dataset for LLM context."""
    lines = [
        f"Dataset Shape: {summary['n_rows']} rows × {summary['n_cols']} columns",
        f"Columns: {', '.join(summary['columns'])}",
        f"Numeric columns: {', '.join(summary['numeric_columns']) or 'None'}",
        f"Categorical columns: {', '.join(summary['categorical_columns']) or 'None'}",
        f"Duplicate rows: {summary['duplicate_rows']}",
    ]

    if summary["missing_values"]:
        mv = ", ".join(
            f"{col}: {cnt}" for col, cnt in summary["missing_values"].items()
        )
        lines.append(f"Missing values: {mv}")
    else:
        lines.append("Missing values: None")

    if summary["numeric_stats"]:
        lines.append("\nNumeric column statistics (mean | std | min | max):")
        for col, stats in summary["numeric_stats"].items():
            mean = stats.get("mean", "N/A")
            std = stats.get("std", "N/A")
            mn = stats.get("min", "N/A")
            mx = stats.get("max", "N/A")
            lines.append(f"  {col}: mean={mean}, std={std}, min={mn}, max={mx}")

    if summary["categorical_top_values"]:
        lines.append("\nTop categorical values:")
        for col, vals in summary["categorical_top_values"].items():
            top = ", ".join(f"{k}({v})" for k, v in vals.items())
            lines.append(f"  {col}: {top}")

    lines.append(f"\nSample rows:\n{summary['sample_rows']}")

    return "\n".join(lines)


def get_column_info_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame showing column name, dtype, non-null count, and % missing."""
    info_rows = []
    for col in df.columns:
        non_null = df[col].notna().sum()
        missing_pct = round((df[col].isna().sum() / len(df)) * 100, 2)
        unique_vals = df[col].nunique()
        info_rows.append(
            {
                "Column": col,
                "Data Type": str(df[col].dtype),
                "Non-Null Count": non_null,
                "Missing %": missing_pct,
                "Unique Values": unique_vals,
            }
        )
    return pd.DataFrame(info_rows)
