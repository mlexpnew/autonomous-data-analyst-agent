"""
Data Analysis Tools - Helper functions for common analytical tasks.
"""

import pandas as pd
import numpy as np


def detect_outliers(df: pd.DataFrame) -> dict:
    """
    Detect outliers in numeric columns using the IQR method.

    Returns:
        Dict mapping column name → list of outlier values.
    """
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        out_vals = df[col][(df[col] < lower) | (df[col] > upper)]
        if not out_vals.empty:
            outliers[col] = {
                "count": len(out_vals),
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "sample": out_vals.head(5).tolist(),
            }

    return outliers


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply basic data cleaning: drop duplicates, fill or flag missing values,
    and attempt dtype inference.

    Returns:
        Tuple of (cleaned DataFrame, list of actions taken).
    """
    actions = []
    cleaned = df.copy()

    # Drop duplicate rows
    dupes = cleaned.duplicated().sum()
    if dupes > 0:
        cleaned = cleaned.drop_duplicates()
        actions.append(f"Removed {dupes} duplicate rows.")

    # Attempt to convert object columns to numeric where possible
    for col in cleaned.select_dtypes(include=["object"]).columns:
        converted = pd.to_numeric(cleaned[col], errors="coerce")
        if converted.notna().sum() > 0.5 * len(cleaned):
            cleaned[col] = converted
            actions.append(f"Converted column '{col}' to numeric.")

    # Fill numeric missing values with median
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing = cleaned[col].isna().sum()
        if missing > 0:
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)
            actions.append(
                f"Filled {missing} missing values in '{col}' with median ({median_val:.4f})."
            )

    # Fill categorical missing values with mode
    cat_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        missing = cleaned[col].isna().sum()
        if missing > 0:
            mode_val = cleaned[col].mode()
            if not mode_val.empty:
                cleaned[col] = cleaned[col].fillna(mode_val[0])
                actions.append(
                    f"Filled {missing} missing values in '{col}' with mode ('{mode_val[0]}')."
                )

    if not actions:
        actions.append("No cleaning actions needed — dataset is already clean.")

    return cleaned, actions


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Return the correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr().round(4)


def frequency_analysis(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    """
    Return value counts (frequency analysis) for a column.

    Args:
        df: Source DataFrame.
        column: Column to analyze.
        top_n: Number of top values to return.

    Returns:
        DataFrame with Value, Count, and Percentage columns.
    """
    if column not in df.columns:
        return pd.DataFrame()

    vc = df[column].value_counts().head(top_n).reset_index()
    vc.columns = ["Value", "Count"]
    vc["Percentage"] = (vc["Count"] / len(df) * 100).round(2)
    return vc


def groupby_aggregation(
    df: pd.DataFrame,
    group_col: str,
    agg_col: str,
    agg_func: str = "sum",
) -> pd.DataFrame:
    """
    Perform a groupby aggregation.

    Args:
        df: Source DataFrame.
        group_col: Column to group by.
        agg_col: Column to aggregate.
        agg_func: Aggregation function name ('sum', 'mean', 'count', 'max', 'min').

    Returns:
        Aggregated DataFrame sorted descending.
    """
    if group_col not in df.columns or agg_col not in df.columns:
        return pd.DataFrame()

    agg_map = {
        "sum": "sum",
        "mean": "mean",
        "count": "count",
        "max": "max",
        "min": "min",
    }
    func = agg_map.get(agg_func.lower(), "sum")
    result = df.groupby(group_col)[agg_col].agg(func).reset_index()
    result.columns = [group_col, f"{agg_func.capitalize()} of {agg_col}"]
    return result.sort_values(by=result.columns[1], ascending=False)
