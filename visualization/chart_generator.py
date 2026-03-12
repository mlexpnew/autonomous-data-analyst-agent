"""
Chart Generator - Creates standard visualizations for data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

# Consistent style across all charts
sns.set_theme(style="whitegrid", palette="muted")
FIGURE_SIZE = (10, 6)


def bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Bar Chart",
    top_n: int = 15,
) -> plt.Figure:
    """Generate a horizontal bar chart for top N values."""
    data = df[[x_col, y_col]].dropna().copy()
    data = data.sort_values(y_col, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    colors = sns.color_palette("Blues_r", len(data))
    bars = ax.barh(data[x_col].astype(str), data[y_col], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(y_col)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    plt.tight_layout()
    return fig


def line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Line Chart",
    hue_col: Optional[str] = None,
) -> plt.Figure:
    """Generate a line chart, optionally with a hue grouping."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    if hue_col and hue_col in df.columns:
        for label, group in df.groupby(hue_col):
            group = group.sort_values(x_col)
            ax.plot(group[x_col], group[y_col], marker="o", label=str(label))
        ax.legend(title=hue_col)
    else:
        data = df[[x_col, y_col]].dropna().sort_values(x_col)
        ax.plot(data[x_col], data[y_col], marker="o", color="#2196F3")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def histogram(
    df: pd.DataFrame,
    col: str,
    bins: int = 30,
    title: str = "Distribution",
) -> plt.Figure:
    """Generate a histogram with KDE overlay."""
    data = df[col].dropna()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.histplot(data, bins=bins, kde=True, ax=ax, color="#42A5F5")
    ax.set_xlabel(col)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = None,
    title: str = "Scatter Plot",
) -> plt.Figure:
    """Generate a scatter plot."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    scatter_kw = dict(data=df, x=x_col, y=y_col, ax=ax, alpha=0.7)
    if hue_col and hue_col in df.columns:
        scatter_kw["hue"] = hue_col
    sns.scatterplot(**scatter_kw)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap") -> plt.Figure:
    """Generate a correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "Not enough numeric columns", ha="center", va="center")
        return fig

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr)), max(6, len(corr) - 1)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def boxplot(
    df: pd.DataFrame,
    col: str,
    group_col: Optional[str] = None,
    title: str = "Box Plot",
) -> plt.Figure:
    """Generate a box plot to visualize distribution and outliers."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    if group_col and group_col in df.columns:
        # Limit to top 10 groups for readability
        top_groups = df[group_col].value_counts().head(10).index
        data = df[df[group_col].isin(top_groups)]
        sns.boxplot(data=data, x=group_col, y=col, ax=ax, palette="muted")
        plt.xticks(rotation=45)
    else:
        sns.boxplot(data=df, y=col, ax=ax, color="#42A5F5")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig
