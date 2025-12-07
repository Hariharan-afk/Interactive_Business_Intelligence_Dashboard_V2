"""
Configuration file for Business Intelligence Dashboard.
Contains constants and configuration settings.
"""

# File handling constants
MAX_FILE_SIZE_MB = 100
SUPPORTED_FILE_TYPES = ['.csv', '.xlsx', '.xls']
DEFAULT_PREVIEW_ROWS = 15
MAX_PREVIEW_ROWS = 1000

# Data processing constants
MAX_CATEGORIES_TO_DISPLAY = 20
MIN_UNIQUE_FOR_CATEGORICAL = 1
MAX_UNIQUE_FOR_CATEGORICAL = 50
OUTLIER_IQR_MULTIPLIER = 1.5
OUTLIER_ZSCORE_THRESHOLD = 3

# Visualization constants
CHART_FIGURE_SIZE = (10, 6)
HEATMAP_FIGURE_SIZE = (12, 10)
DPI = 100
DEFAULT_COLOR_PALETTE = 'Set2'

# Color schemes
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8'
}

# Chart colors
CHART_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# Statistical thresholds
CORRELATION_THRESHOLD = 0.7  # For identifying strong correlations
MISSING_VALUE_THRESHOLD = 0.5  # Flag columns with >50% missing values

# Export settings
EXPORT_CSV_ENCODING = 'utf-8'
EXPORT_PNG_DPI = 300
EXPORT_PNG_BBOX = 'tight'

# Column type mappings
COLUMN_TYPES = {
    'NUMERIC': 'numeric',
    'CATEGORICAL': 'categorical',
    'DATETIME': 'datetime',
    'TEXT': 'text'
}

# Aggregation methods
AGGREGATION_METHODS = ['sum', 'mean', 'count', 'median', 'min', 'max', 'std']

# Chart types
CHART_TYPES = {
    'TIME_SERIES': 'Time Series',
    'HISTOGRAM': 'Histogram',
    'BOX_PLOT': 'Box Plot',
    'BAR_CHART': 'Bar Chart',
    'PIE_CHART': 'Pie Chart',
    'SCATTER': 'Scatter Plot',
    'HEATMAP': 'Correlation Heatmap'
}