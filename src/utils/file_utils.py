"""
Utility functions for Business Intelligence Dashboard.
Handles exports, validation, formatting, and other helper functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import os
from datetime import datetime
import config


def export_dataframe_to_csv(df: pd.DataFrame, filename: str = None) -> str:
    """
    Export DataFrame to CSV file.
    
    Args:
        df: DataFrame to export
        filename: Optional filename (auto-generated if not provided)
        
    Returns:
        Path to the exported file
    """
    try:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exported_data_{timestamp}.csv"
        
        # Ensure filename ends with .csv
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Export to temporary location
        filepath = os.path.join('/tmp', filename)
        df.to_csv(filepath, index=False, encoding=config.EXPORT_CSV_ENCODING)
        
        return filepath
    
    except Exception as e:
        print(f"Error exporting DataFrame to CSV: {str(e)}")
        return None


def save_figure_as_png(fig, filename: str = None) -> str:
    """
    Save matplotlib/plotly figure as PNG file.
    
    Args:
        fig: Matplotlib figure or plotly figure
        filename: Optional filename (auto-generated if not provided)
        
    Returns:
        Path to the saved file
    """
    try:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{timestamp}.png"
        
        # Ensure filename ends with .png
        if not filename.endswith('.png'):
            filename += '.png'
        
        # Export to temporary location
        filepath = os.path.join('/tmp', filename)
        
        # Check if it's a plotly figure
        if hasattr(fig, 'write_image'):
            fig.write_image(filepath, width=1200, height=800)
        else:
            # Matplotlib figure
            fig.savefig(filepath, dpi=config.EXPORT_PNG_DPI, 
                       bbox_inches=config.EXPORT_PNG_BBOX)
        
        return filepath
    
    except Exception as e:
        print(f"Error saving figure as PNG: {str(e)}")
        return None


def validate_file_upload(file_path: str) -> Tuple[bool, str]:
    """
    Validate uploaded file.
    
    Args:
        file_path: Path to the uploaded file
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not file_path:
        return False, "No file uploaded"
    
    # Check if file exists
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check file extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in config.SUPPORTED_FILE_TYPES:
        return False, f"Unsupported file type. Please upload {', '.join(config.SUPPORTED_FILE_TYPES)}"
    
    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > config.MAX_FILE_SIZE_MB:
        return False, f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({config.MAX_FILE_SIZE_MB} MB)"
    
    return True, "File validation successful"


def format_large_numbers(value: float) -> str:
    """
    Format large numbers with K, M, B suffixes.
    
    Args:
        value: Number to format
        
    Returns:
        Formatted string
    """
    try:
        if pd.isna(value):
            return "N/A"
        
        abs_value = abs(value)
        
        if abs_value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        elif abs_value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        elif abs_value >= 1_000:
            return f"{value / 1_000:.2f}K"
        else:
            return f"{value:.2f}"
    
    except Exception as e:
        return str(value)


def generate_color_palette(n_colors: int) -> List[str]:
    """
    Generate consistent color palette for charts.
    
    Args:
        n_colors: Number of colors needed
        
    Returns:
        List of color codes
    """
    if n_colors <= len(config.CHART_COLORS):
        return config.CHART_COLORS[:n_colors]
    else:
        # If more colors needed, repeat the palette
        repeats = (n_colors // len(config.CHART_COLORS)) + 1
        return (config.CHART_COLORS * repeats)[:n_colors]


def format_dataframe_for_display(df: pd.DataFrame, max_rows: int = None) -> pd.DataFrame:
    """
    Format DataFrame for better display in Gradio.
    
    Args:
        df: Input DataFrame
        max_rows: Maximum number of rows to display
        
    Returns:
        Formatted DataFrame
    """
    try:
        display_df = df.copy()
        
        # Limit rows if specified
        if max_rows and len(display_df) > max_rows:
            display_df = display_df.head(max_rows)
        
        # Format datetime columns
        for col in display_df.columns:
            if pd.api.types.is_datetime64_any_dtype(display_df[col]):
                display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Round numerical columns
        for col in display_df.select_dtypes(include=[np.number]).columns:
            if display_df[col].dtype in ['float64', 'float32']:
                display_df[col] = display_df[col].round(2)
        
        return display_df
    
    except Exception as e:
        print(f"Error formatting DataFrame: {str(e)}")
        return df


def create_summary_text(df: pd.DataFrame, column_types: dict) -> str:
    """
    Create a text summary of the dataset.
    
    Args:
        df: Input DataFrame
        column_types: Dictionary of column types
        
    Returns:
        Formatted summary text
    """
    try:
        summary = "## Dataset Summary\n\n"
        summary += f"**Total Rows:** {len(df):,}\n\n"
        summary += f"**Total Columns:** {len(df.columns)}\n\n"
        
        # Count by type
        type_counts = {}
        for col_type in column_types.values():
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
        
        summary += "**Column Types:**\n"
        for col_type, count in type_counts.items():
            summary += f"- {col_type.capitalize()}: {count}\n"
        
        # Missing values
        total_missing = df.isnull().sum().sum()
        missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
        summary += f"\n**Missing Values:** {total_missing:,} ({missing_pct:.2f}%)\n\n"
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        summary += f"**Memory Usage:** {memory_mb:.2f} MB\n\n"
        
        return summary
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect potential date columns in DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names that might contain dates
    """
    date_columns = []
    
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
        elif df[col].dtype == 'object':
            # Try to parse a sample
            try:
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    pd.to_datetime(sample, errors='raise')
                    date_columns.append(col)
            except (ValueError, TypeError):
                pass
    
    return date_columns


def get_sample_data(df: pd.DataFrame, n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get sample data from the beginning and end of DataFrame.
    
    Args:
        df: Input DataFrame
        n: Number of rows to sample
        
    Returns:
        Tuple of (head_sample, tail_sample)
    """
    try:
        head_sample = df.head(n)
        tail_sample = df.tail(n)
        return head_sample, tail_sample
    except Exception as e:
        print(f"Error getting sample data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


def clean_column_name(col_name: str) -> str:
    """
    Clean column name for display.
    
    Args:
        col_name: Original column name
        
    Returns:
        Cleaned column name
    """
    # Replace underscores with spaces
    cleaned = col_name.replace('_', ' ')
    # Capitalize first letter of each word
    cleaned = cleaned.title()
    return cleaned


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of numeric columns from DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of numeric column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame, max_unique: int = 50) -> List[str]:
    """
    Get list of categorical columns from DataFrame.
    
    Args:
        df: Input DataFrame
        max_unique: Maximum unique values to consider as categorical
        
    Returns:
        List of categorical column names
    """
    categorical_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            if df[col].nunique() <= max_unique:
                categorical_cols.append(col)
    
    return categorical_cols


def calculate_dataset_health_score(df: pd.DataFrame) -> Tuple[float, str]:
    """
    Calculate overall health score for the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (score, status_message)
    """
    try:
        score = 100.0
        issues = []
        
        # Deduct for missing values
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 0:
            score -= min(missing_pct, 30)
            if missing_pct > 10:
                issues.append(f"High missing values ({missing_pct:.1f}%)")
        
        # Deduct for columns with all same values
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            score -= len(constant_cols) * 5
            issues.append(f"{len(constant_cols)} constant column(s)")
        
        # Deduct for duplicate rows
        dup_pct = (df.duplicated().sum() / len(df)) * 100
        if dup_pct > 0:
            score -= min(dup_pct, 20)
            if dup_pct > 5:
                issues.append(f"Duplicate rows ({dup_pct:.1f}%)")
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        # Generate status message
        if score >= 90:
            status = "Excellent"
        elif score >= 75:
            status = "Good"
        elif score >= 60:
            status = "Fair"
        else:
            status = "Needs Attention"
        
        if issues:
            status += f" - Issues: {', '.join(issues)}"
        
        return score, status
    
    except Exception as e:
        return 0, f"Error calculating health score: {str(e)}"