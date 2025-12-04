"""
Data processing module for Business Intelligence Dashboard.
Handles data loading, validation, type detection, statistics, and filtering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import config


def load_dataset(file_path: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load dataset from CSV or Excel file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (DataFrame, status_message)
    """
    try:
        # Check file extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return None, f"Unsupported file type. Please upload {', '.join(config.SUPPORTED_FILE_TYPES)}"
        
        if df.empty:
            return None, "The uploaded file is empty."
        
        if len(df.columns) == 0:
            return None, "The uploaded file has no columns."
        
        return df, f"Dataset loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns"
    
    except UnicodeDecodeError:
        try:
            # Try with different encoding
            df = pd.read_csv(file_path, encoding='latin-1')
            return df, f"Dataset loaded successfully (latin-1 encoding)! Shape: {df.shape[0]} rows × {df.shape[1]} columns"
        except Exception as e:
            return None, f"Encoding error: {str(e)}"
    
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate dataset and check for common issues.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing validation results
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }
    
    if df is None or df.empty:
        validation['is_valid'] = False
        validation['errors'].append("Dataset is empty")
        return validation
    
    # Check for minimum requirements
    if len(df) < 1:
        validation['is_valid'] = False
        validation['errors'].append("Dataset must have at least 1 row")
    
    if len(df.columns) < 1:
        validation['is_valid'] = False
        validation['errors'].append("Dataset must have at least 1 column")
    
    # Check for duplicate column names
    if df.columns.duplicated().any():
        validation['warnings'].append("Dataset contains duplicate column names")
    
    # Check for missing values
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_pct > 50:
        validation['warnings'].append(f"Dataset has {missing_pct:.1f}% missing values")
    
    # Check for columns with all missing values
    all_missing = df.columns[df.isnull().all()].tolist()
    if all_missing:
        validation['warnings'].append(f"Columns with all missing values: {', '.join(all_missing)}")
    
    return validation


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get basic information about the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        'shape': df.shape,
        'rows': df.shape[0],
        'columns': df.shape[1],
        'column_names': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 ** 2),
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    }
    return info


def auto_detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Automatically detect column types based on data characteristics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping column names to detected types
    """
    column_types = {}
    
    for col in df.columns:
        # Skip columns with all missing values
        if df[col].isnull().all():
            column_types[col] = config.COLUMN_TYPES['TEXT']
            continue
        
        # Check if datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = config.COLUMN_TYPES['DATETIME']
        # Try to parse as datetime
        elif df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].dropna().head(100), errors='raise')
                column_types[col] = config.COLUMN_TYPES['DATETIME']
            except (ValueError, TypeError):
                # Check if numeric (stored as string)
                try:
                    pd.to_numeric(df[col].dropna().head(100), errors='raise')
                    column_types[col] = config.COLUMN_TYPES['NUMERIC']
                except (ValueError, TypeError):
                    # Check if categorical or text
                    n_unique = df[col].nunique()
                    n_total = len(df[col].dropna())
                    
                    if n_unique <= config.MAX_UNIQUE_FOR_CATEGORICAL and n_total > 0:
                        if n_unique / n_total < 0.5:  # Less than 50% unique values
                            column_types[col] = config.COLUMN_TYPES['CATEGORICAL']
                        else:
                            column_types[col] = config.COLUMN_TYPES['TEXT']
                    else:
                        column_types[col] = config.COLUMN_TYPES['TEXT']
        # Check if numeric
        elif pd.api.types.is_numeric_dtype(df[col]):
            n_unique = df[col].nunique()
            # If few unique values and integers, might be categorical
            if n_unique <= 10 and df[col].dtype in ['int64', 'int32']:
                column_types[col] = config.COLUMN_TYPES['CATEGORICAL']
            else:
                column_types[col] = config.COLUMN_TYPES['NUMERIC']
        else:
            column_types[col] = config.COLUMN_TYPES['TEXT']
    
    return column_types


def convert_column_type(df: pd.DataFrame, column: str, new_type: str) -> Tuple[pd.DataFrame, str]:
    """
    Convert a column to a specified type.
    
    Args:
        df: Input DataFrame
        column: Column name to convert
        new_type: Target type ('numeric', 'categorical', 'datetime', 'text')
        
    Returns:
        Tuple of (modified DataFrame, status message)
    """
    try:
        df_copy = df.copy()
        
        if new_type == config.COLUMN_TYPES['NUMERIC']:
            df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
            message = f"Column '{column}' converted to numeric"
        
        elif new_type == config.COLUMN_TYPES['CATEGORICAL']:
            df_copy[column] = df_copy[column].astype('category')
            message = f"Column '{column}' converted to categorical"
        
        elif new_type == config.COLUMN_TYPES['DATETIME']:
            df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
            message = f"Column '{column}' converted to datetime"
        
        elif new_type == config.COLUMN_TYPES['TEXT']:
            df_copy[column] = df_copy[column].astype(str)
            message = f"Column '{column}' converted to text"
        
        else:
            return df, f"Unknown type: {new_type}"
        
        return df_copy, message
    
    except Exception as e:
        return df, f"Error converting column '{column}': {str(e)}"



def apply_column_type_conversions(df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
    """
    Apply dtype conversions to DataFrame based on user-specified column types.
    
    This ensures that pandas dtypes match the user's column type classifications,
    preventing dtype mismatch errors when calculating statistics.
    
    Args:
        df: Input DataFrame  
        column_types: Dictionary mapping column names to their types
        
    Returns:
        DataFrame with converted dtypes
    """
    df_converted = df.copy()
    
    for column, col_type in column_types.items():
        if column not in df_converted.columns:
            continue
        
        try:
            if col_type == config.COLUMN_TYPES['NUMERIC']:
                # Convert to numeric dtype
                df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
            
            elif col_type == config.COLUMN_TYPES['CATEGORICAL']:
                # Convert to string (object dtype) for categorical columns
                # Using object dtype instead of category to avoid dtype mismatch errors
                df_converted[column] = df_converted[column].astype(str)
            
            elif col_type == config.COLUMN_TYPES['DATETIME']:
                # Convert to datetime
                df_converted[column] = pd.to_datetime(df_converted[column], errors='coerce')
            
            elif col_type == config.COLUMN_TYPES['TEXT']:
                # Convert to string  
                df_converted[column] = df_converted[column].astype(str)
        
        except Exception as e:
            print(f"Warning: Could not convert column '{column}' to type '{col_type}': {str(e)}")
            # Continue with other columns even if one fails
            continue
    
    return df_converted


def get_filterable_columns(df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Group columns by type for filter UI generation.
    
    Args:
        df: Input DataFrame
        column_types: Dictionary mapping column names to types
        
    Returns:
        Dictionary grouping columns by type
    """
    filterable = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'text': []
    }
    
    for col, col_type in column_types.items():
        if col in df.columns:
            filterable[col_type].append(col)
    
    return filterable


def calculate_numerical_stats(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Calculate comprehensive statistics for numerical columns.
    
    Args:
        df: Input DataFrame
        columns: List of numerical column names
        
    Returns:
        DataFrame with statistics for each column
    """
    if not columns:
        return pd.DataFrame()
    
    stats_dict = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        stats_dict[col] = {
            'Count': len(col_data),
            'Missing': df[col].isnull().sum(),
            'Missing %': f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%",
            'Mean': col_data.mean(),
            'Median': col_data.median(),
            'Std Dev': col_data.std(),
            'Min': col_data.min(),
            'Q1': col_data.quantile(0.25),
            'Q3': col_data.quantile(0.75),
            'Max': col_data.max(),
            'Range': col_data.max() - col_data.min()
        }
    
    if not stats_dict:
        return pd.DataFrame()
    
    stats_df = pd.DataFrame(stats_dict).T
    stats_df.index.name = 'Column'
    
    # Round only the truly numerical columns (skip 'Count', 'Missing', 'Missing %')
    numeric_cols_to_round = ['Mean', 'Median', 'Std Dev', 'Min', 'Q1', 'Q3', 'Max', 'Range']
    for col in numeric_cols_to_round:
        if col in stats_df.columns:
            # Ensure the column is numeric before rounding
            try:
                stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce').round(2)
            except Exception as e:
                print(f"Warning: Could not round column  '{col}': {str(e)}")
    
    return stats_df.reset_index()


def calculate_categorical_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate statistics for categorical columns.
    
    Args:
        df: Input DataFrame
        columns: List of categorical column names
        
    Returns:
        Dictionary with statistics for each column
    """
    stats_dict = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        value_counts = df[col].value_counts()
        top_5 = value_counts.head(5)
        
        stats_dict[col] = {
            'Unique Values': df[col].nunique(),
            'Missing': df[col].isnull().sum(),
            'Missing %': f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%",
            'Mode': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
            'Top 5 Values': top_5.to_dict()
        }
    
    return stats_dict


def calculate_correlation_matrix(df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
    """
    Calculate correlation matrix for numerical columns.
    
    Args:
        df: Input DataFrame
        numerical_cols: List of numerical column names
        
    Returns:
        Correlation matrix as DataFrame
    """
    if not numerical_cols or len(numerical_cols) < 2:
        return pd.DataFrame()
    
    try:
        # Select only the numerical columns that exist
        valid_cols = [col for col in numerical_cols if col in df.columns]
        
        if len(valid_cols) < 2:
            return pd.DataFrame()
        
        corr_matrix = df[valid_cols].corr()
        return corr_matrix
    
    except Exception as e:
        print(f"Error calculating correlation matrix: {str(e)}")
        return pd.DataFrame()


def generate_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive missing value report.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing value analysis
    """
    missing_data = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        missing_data.append({
            'Column': col,
            'Missing Count': missing_count,
            'Missing Percentage': f"{missing_pct:.2f}%",
            'Data Type': str(df[col].dtype)
        })
    
    missing_df = pd.DataFrame(missing_data)
    
    # Sort by missing count (descending)
    missing_df = missing_df.sort_values('Missing Count', ascending=False)
    
    return missing_df


def apply_numerical_filter(df: pd.DataFrame, column: str, min_val: float, max_val: float) -> pd.DataFrame:
    """
    Apply numerical range filter to DataFrame.
    
    Args:
        df: Input DataFrame
        column: Column name to filter
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Filtered DataFrame
    """
    try:
        return df[(df[column] >= min_val) & (df[column] <= max_val)]
    except Exception as e:
        print(f"Error applying numerical filter: {str(e)}")
        return df


def apply_categorical_filter(df: pd.DataFrame, column: str, selected_values: List[Any]) -> pd.DataFrame:
    """
    Apply categorical filter to DataFrame.
    
    Args:
        df: Input DataFrame
        column: Column name to filter
        selected_values: List of values to keep
        
    Returns:
        Filtered DataFrame
    """
    try:
        if not selected_values:
            return df
        return df[df[column].isin(selected_values)]
    except Exception as e:
        print(f"Error applying categorical filter: {str(e)}")
        return df


def apply_date_filter(df: pd.DataFrame, column: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Apply date range filter to DataFrame.
    
    Args:
        df: Input DataFrame
        column: Column name to filter
        start_date: Start date (string)
        end_date: End date (string)
        
    Returns:
        Filtered DataFrame
    """
    try:
        # Convert column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = pd.to_datetime(df[column], errors='coerce')
        
        # Convert filter dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        return df[(df[column] >= start) & (df[column] <= end)]
    except Exception as e:
        print(f"Error applying date filter: {str(e)}")
        return df


def apply_combined_filters(df: pd.DataFrame, filter_config: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
    """
    Apply multiple filters simultaneously.
    
    Args:
        df: Input DataFrame
        filter_config: Dictionary containing filter configurations
        
    Returns:
        Tuple of (filtered DataFrame, row count)
    """
    filtered_df = df.copy()
    
    try:
        # Apply numerical filters
        if 'numerical' in filter_config:
            for col, (min_val, max_val) in filter_config['numerical'].items():
                if col in filtered_df.columns and min_val is not None and max_val is not None:
                    filtered_df = apply_numerical_filter(filtered_df, col, min_val, max_val)
        
        # Apply categorical filters
        if 'categorical' in filter_config:
            for col, values in filter_config['categorical'].items():
                if col in filtered_df.columns and values:
                    filtered_df = apply_categorical_filter(filtered_df, col, values)
        
        # Apply date filters
        if 'datetime' in filter_config:
            for col, (start, end) in filter_config['datetime'].items():
                if col in filtered_df.columns and start and end:
                    filtered_df = apply_date_filter(filtered_df, col, start, end)
        
        return filtered_df, len(filtered_df)
    
    except Exception as e:
        print(f"Error applying combined filters: {str(e)}")
        return df, len(df)


def get_column_value_range(df: pd.DataFrame, column: str, col_type: str) -> Tuple[Any, Any]:
    """
    Get the value range for a column (min/max for numerical, unique values for categorical).
    
    Args:
        df: Input DataFrame
        column: Column name
        col_type: Column type
        
    Returns:
        Tuple of (min_value, max_value) or (None, unique_values)
    """
    try:
        if col_type == config.COLUMN_TYPES['NUMERIC']:
            return df[column].min(), df[column].max()
        elif col_type == config.COLUMN_TYPES['CATEGORICAL']:
            return None, df[column].dropna().unique().tolist()
        elif col_type == config.COLUMN_TYPES['DATETIME']:
            return df[column].min(), df[column].max()
        else:
            return None, None
    except Exception as e:
        print(f"Error getting column range: {str(e)}")
        return None, None