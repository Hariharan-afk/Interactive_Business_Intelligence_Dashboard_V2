"""
Insights generation module for Business Intelligence Dashboard.
Handles automated generation of data insights and patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
import config


def identify_top_bottom_performers(df: pd.DataFrame, metric_columns: List[str], 
                                   n: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Identify top N and bottom N values for each metric column.
    
    Args:
        df: Input DataFrame
        metric_columns: List of numerical columns to analyze
        n: Number of top/bottom performers to identify
        
    Returns:
        Dictionary with top and bottom performers DataFrames
    """
    results = {}
    
    for col in metric_columns:
        if col not in df.columns:
            continue
        
        # Get non-null data
        valid_data = df[df[col].notna()]
        
        if len(valid_data) == 0:
            continue
        
        # Get top performers
        top_n = valid_data.nlargest(n, col)
        
        # Get bottom performers
        bottom_n = valid_data.nsmallest(n, col)
        
        results[col] = {
            'top': top_n,
            'bottom': bottom_n,
            'top_mean': top_n[col].mean(),
            'bottom_mean': bottom_n[col].mean(),
            'overall_mean': valid_data[col].mean()
        }
    
    return results


def detect_outliers(df: pd.DataFrame, numerical_cols: List[str], 
                   method: str = 'iqr') -> Dict[str, Any]:
    """
    Detect outliers in numerical columns.
    
    Args:
        df: Input DataFrame
        numerical_cols: List of numerical columns
        method: 'iqr' or 'zscore'
        
    Returns:
        Dictionary with outlier information
    """
    outlier_info = {}
    
    for col in numerical_cols:
        if col not in df.columns:
            continue
        
        data = df[col].dropna()
        
        if len(data) < 4:  # Need minimum data points
            continue
        
        outliers = pd.Series(dtype=bool)
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - config.OUTLIER_IQR_MULTIPLIER * IQR
            upper_bound = Q3 + config.OUTLIER_IQR_MULTIPLIER * IQR
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = pd.Series(False, index=df.index)
            outliers.loc[data.index] = z_scores > config.OUTLIER_ZSCORE_THRESHOLD
        
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            outlier_info[col] = {
                'count': int(outlier_count),
                'percentage': (outlier_count / len(df)) * 100,
                'examples': df[outliers][col].head(5).tolist()
            }
    
    return outlier_info


def analyze_trends(df: pd.DataFrame, date_col: str, value_cols: List[str]) -> Dict[str, Any]:
    """
    Analyze trends in time series data.
    
    Args:
        df: Input DataFrame
        date_col: Date column name
        value_cols: List of value columns to analyze
        
    Returns:
        Dictionary with trend information
    """
    try:
        # Convert date column to datetime
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy = df_copy.sort_values(date_col)
        
        trends = {}
        
        for col in value_cols:
            if col not in df_copy.columns:
                continue
            
            # Remove missing values
            df_valid = df_copy[[date_col, col]].dropna()
            
            if len(df_valid) < 2:
                continue
            
            # Calculate overall trend (simple linear regression)
            x = np.arange(len(df_valid))
            y = df_valid[col].values
            
            if len(x) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                
                # Calculate growth rate
                first_value = df_valid[col].iloc[0]
                last_value = df_valid[col].iloc[-1]
                
                if first_value != 0:
                    growth_rate = ((last_value - first_value) / first_value) * 100
                else:
                    growth_rate = 0
                
                # Determine trend direction
                if slope > 0:
                    trend_direction = "Increasing"
                elif slope < 0:
                    trend_direction = "Decreasing"
                else:
                    trend_direction = "Stable"
                
                # Calculate volatility (coefficient of variation)
                volatility = (df_valid[col].std() / df_valid[col].mean()) * 100 if df_valid[col].mean() != 0 else 0
                
                trends[col] = {
                    'direction': trend_direction,
                    'slope': float(slope),
                    'growth_rate': float(growth_rate),
                    'volatility': float(volatility),
                    'first_value': float(first_value),
                    'last_value': float(last_value)
                }
        
        return trends
    
    except Exception as e:
        print(f"Error analyzing trends: {str(e)}")
        return {}


def generate_summary_insights(df: pd.DataFrame, column_types: Dict[str, str]) -> str:
    """
    Generate comprehensive text summary of insights.
    
    Args:
        df: Input DataFrame
        column_types: Dictionary mapping column names to types
        
    Returns:
        Formatted text summary
    """
    try:
        insights = []
        insights.append("## 📊 Automated Insights\n")
        
        # Dataset composition
        insights.append("### Dataset Overview")
        insights.append(f"- **Total Records**: {len(df):,}")
        insights.append(f"- **Total Features**: {len(df.columns)}")
        
        # Count column types
        type_counts = {}
        for col_type in column_types.values():
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
        
        insights.append(f"- **Numerical Columns**: {type_counts.get('numeric', 0)}")
        insights.append(f"- **Categorical Columns**: {type_counts.get('categorical', 0)}")
        insights.append(f"- **Date Columns**: {type_counts.get('datetime', 0)}")
        
        # Data quality
        insights.append("\n### Data Quality")
        total_missing = df.isnull().sum().sum()
        missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
        
        if missing_pct == 0:
            insights.append("- ✅ No missing values detected")
        elif missing_pct < 5:
            insights.append(f"- ⚠️ Low missing values: {missing_pct:.2f}%")
        elif missing_pct < 20:
            insights.append(f"- ⚠️ Moderate missing values: {missing_pct:.2f}%")
        else:
            insights.append(f"- ❌ High missing values: {missing_pct:.2f}% - Consider imputation")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            insights.append(f"- ⚠️ Found {duplicate_count} duplicate rows ({(duplicate_count/len(df)*100):.2f}%)")
        else:
            insights.append("- ✅ No duplicate rows")
        
        # Numerical insights
        numerical_cols = [col for col, ctype in column_types.items() if ctype == 'numeric']
        if numerical_cols:
            insights.append("\n### Numerical Features Analysis")
            for col in numerical_cols[:5]:  # Show top 5
                if col in df.columns:
                    data = df[col].dropna()
                    if len(data) > 0:
                        insights.append(f"- **{col}**: Mean={data.mean():.2f}, Median={data.median():.2f}, Std={data.std():.2f}")
        
        # Categorical insights
        categorical_cols = [col for col, ctype in column_types.items() if ctype == 'categorical']
        if categorical_cols:
            insights.append("\n### Categorical Features Analysis")
            for col in categorical_cols[:5]:  # Show top 5
                if col in df.columns:
                    unique_count = df[col].nunique()
                    mode_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
                    insights.append(f"- **{col}**: {unique_count} unique values, Most common: {mode_value}")
        
        return "\n".join(insights)
    
    except Exception as e:
        return f"Error generating summary insights: {str(e)}"


def calculate_category_concentration(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, Any]:
    """
    Calculate concentration metrics for categorical columns.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
        
    Returns:
        Dictionary with concentration metrics
    """
    concentration_info = {}
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
        
        value_counts = df[col].value_counts()
        
        if len(value_counts) == 0:
            continue
        
        # Calculate concentration ratio (top 1 value)
        top_1_ratio = value_counts.iloc[0] / len(df) * 100
        
        # Calculate concentration ratio (top 3 values)
        top_3_ratio = value_counts.head(3).sum() / len(df) * 100
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        proportions = value_counts / len(df)
        hhi = (proportions ** 2).sum() * 10000  # Scale to 0-10000
        
        concentration_info[col] = {
            'unique_values': len(value_counts),
            'top_1_ratio': float(top_1_ratio),
            'top_3_ratio': float(top_3_ratio),
            'hhi': float(hhi),
            'is_concentrated': top_1_ratio > 50  # More than 50% in one category
        }
    
    return concentration_info


def find_correlations(df: pd.DataFrame, numerical_cols: List[str], 
                     threshold: float = 0.7) -> List[Tuple[str, str, float]]:
    """
    Find strong correlations between numerical columns.
    
    Args:
        df: Input DataFrame
        numerical_cols: List of numerical column names
        threshold: Correlation threshold (absolute value)
        
    Returns:
        List of tuples (col1, col2, correlation)
    """
    try:
        if len(numerical_cols) < 2:
            return []
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Find strong correlations
        strong_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    strong_corr.append((col1, col2, float(corr_value)))
        
        # Sort by absolute correlation
        strong_corr.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return strong_corr
    
    except Exception as e:
        print(f"Error finding correlations: {str(e)}")
        return []


def detect_anomalies(df: pd.DataFrame, numerical_cols: List[str]) -> Dict[str, Any]:
    """
    Detect statistical anomalies in data.
    
    Args:
        df: Input DataFrame
        numerical_cols: List of numerical column names
        
    Returns:
        Dictionary with anomaly information
    """
    anomalies = {}
    
    for col in numerical_cols:
        if col not in df.columns:
            continue
        
        data = df[col].dropna()
        
        if len(data) < 10:  # Need sufficient data
            continue
        
        col_anomalies = []
        
        # Check for extreme values
        mean = data.mean()
        std = data.std()
        
        if std > 0:
            # Values more than 3 standard deviations from mean
            extreme_values = data[np.abs((data - mean) / std) > 3]
            if len(extreme_values) > 0:
                col_anomalies.append({
                    'type': 'Extreme Values',
                    'count': len(extreme_values),
                    'description': f'{len(extreme_values)} values more than 3 std from mean'
                })
        
        # Check for zero/negative values if they seem unusual
        if data.min() <= 0 and data.max() > 0:
            non_positive = data[data <= 0]
            if len(non_positive) < len(data) * 0.1:  # Less than 10%
                col_anomalies.append({
                    'type': 'Unusual Zero/Negative Values',
                    'count': len(non_positive),
                    'description': f'{len(non_positive)} zero or negative values in mostly positive data'
                })
        
        if col_anomalies:
            anomalies[col] = col_anomalies
    
    return anomalies


def generate_top_insights_text(df: pd.DataFrame, column_types: Dict[str, str], 
                               top_n: int = 5) -> str:
    """
    Generate text highlighting top insights for display.
    
    Args:
        df: Input DataFrame
        column_types: Dictionary of column types
        top_n: Number of insights to generate
        
    Returns:
        Formatted insight text
    """
    try:
        insights_text = []
        
        # Get numerical and categorical columns
        numerical_cols = [col for col, ctype in column_types.items() if ctype == 'numeric']
        categorical_cols = [col for col, ctype in column_types.items() if ctype == 'categorical']
        
        # Insight 1: Data completeness
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct < 5:
            insights_text.append(f"✅ **High Data Quality**: Only {missing_pct:.2f}% missing values")
        else:
            insights_text.append(f"⚠️ **Data Quality Alert**: {missing_pct:.2f}% missing values detected")
        
        # Insight 2: Outliers in numerical columns
        if numerical_cols:
            outliers = detect_outliers(df, numerical_cols[:3], method='iqr')
            if outliers:
                for col, info in list(outliers.items())[:2]:
                    insights_text.append(f"📊 **{col}**: {info['count']} outliers detected ({info['percentage']:.1f}%)")
        
        # Insight 3: Category concentration
        if categorical_cols:
            concentration = calculate_category_concentration(df, categorical_cols[:3])
            for col, info in list(concentration.items())[:2]:
                if info['is_concentrated']:
                    insights_text.append(f"📈 **{col}**: Highly concentrated - top value represents {info['top_1_ratio']:.1f}%")
        
        # Insight 4: Strong correlations
        if len(numerical_cols) >= 2:
            correlations = find_correlations(df, numerical_cols, threshold=0.7)
            for col1, col2, corr in correlations[:2]:
                corr_type = "positive" if corr > 0 else "negative"
                insights_text.append(f"🔗 **Strong {corr_type} correlation**: {col1} & {col2} (r={corr:.2f})")
        
        # Limit to top_n insights
        insights_text = insights_text[:top_n]
        
        if not insights_text:
            insights_text = ["📋 No significant patterns detected in the current dataset"]
        
        return "\n\n".join(insights_text)
    
    except Exception as e:
        return f"Error generating insights: {str(e)}"