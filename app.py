"""
Business Intelligence Dashboard - Main Gradio Application
A comprehensive, interactive dashboard for business data analysis.
"""

import gradio as gr
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Import custom modules
from src.core import data_processor as dp
from src.visualization import charts as viz
from src.analytics import insights as ins
from src.utils import file_utils as utils
import config


# Global state to store data between function calls
class DashboardState:
    """Class to maintain dashboard state"""
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.column_types: Dict[str, str] = {}
        self.filtered_df: Optional[pd.DataFrame] = None
        self.active_filters: Dict[str, Any] = {
            'numerical': {},
            'categorical': {},
            'datetime': {}
        }


def upload_and_preview(file) -> Tuple[str, pd.DataFrame, pd.DataFrame, str, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Handle file upload and display preview.
    
    Returns:
        Tuple of (status_message, head_df, tail_df, summary_text, info_df)
    """
    try:
        if file is None:
            gr.Warning("Please upload a CSV or Excel file before clicking Load Dataset")
            return "⚠️ Please upload a CSV or Excel file before clicking Load Dataset", None, None, "", None, None, {}
        
        # Validate file
        is_valid, msg = utils.validate_file_upload(file.name)
        if not is_valid:
            gr.Warning(msg)
            return msg, None, None, "", None, None, {}
        
        # Load dataset
        df, status = dp.load_dataset(file.name)
        
        if df is None:
            gr.Warning(status)
            return status, None, None, "", None, None, {}
        
        # Validate dataset
        validation = dp.validate_dataset(df)
        if not validation['is_valid']:
            error_msg = f"⚠️ Dataset validation failed: {', '.join(validation['errors'])}"
            gr.Warning(error_msg)
            return error_msg, None, None, "", None, None, {}
        
        # Auto-detect column types
        column_types = dp.auto_detect_column_types(df)
        
        # Get dataset info
        info = dp.get_dataset_info(df)
        
        # Create info DataFrame for display
        info_data = {
            'Metric': ['Rows', 'Columns', 'Memory (MB)', 'Missing Values', 'Missing %'],
            'Value': [
                f"{info['rows']:,}",
                info['columns'],
                f"{info['memory_usage_mb']:.2f}",
                f"{info['total_missing']:,}",
                f"{info['missing_percentage']:.2f}%"
            ]
        }
        info_df = pd.DataFrame(info_data)
        
        # Get preview data
        head_df = utils.format_dataframe_for_display(df.head(config.DEFAULT_PREVIEW_ROWS))
        tail_df = utils.format_dataframe_for_display(df.tail(config.DEFAULT_PREVIEW_ROWS))
        
        # Create summary text
        summary = utils.create_summary_text(df, column_types)
        
        # Add warnings if any
        if validation['warnings']:
            summary += "\n\n### ⚠️ Warnings:\n"
            for warning in validation['warnings']:
                summary += f"- {warning}\n"
        
        return status, head_df, tail_df, summary, info_df, df, column_types
    
    except Exception as e:
        error_msg = f"⚠️ Error loading dataset. Please ensure the file is a valid CSV or Excel file."
        gr.Warning(error_msg)
        return error_msg, None, None, "", None, None, {}


def show_column_types(stored_df, stored_column_types) -> Tuple[pd.DataFrame, str]:
    """
    Display detected column types.
    
    Returns:
        Tuple of (column_types_df, status_message)
    """
    try:
        if stored_df is None or not stored_column_types:
            return None, "Please upload a file first"
        
        # Use stored data
        df = stored_df
        column_types = stored_column_types
        
        # Create DataFrame for display
        types_data = {
            'Column Name': list(column_types.keys()),
            'Detected Type': list(column_types.values()),
            'Sample Values': [str(df[col].dropna().head(3).tolist()) for col in column_types.keys()]
        }
        types_df = pd.DataFrame(types_data)
        
        return types_df, "Column types detected successfully. You can modify them below if needed."
    
    except Exception as e:
        return None, f"Error: {str(e)}"



def get_columns_from_file(file) -> gr.update:
    """
    Safely get columns from file for dropdown update.
    """
    if file is None:
        return gr.update(choices=[])
    
    df, _ = dp.load_dataset(file.name)
    if df is None:
        return gr.update(choices=[])
        
    return gr.update(choices=list(df.columns))


def update_column_type(stored_df, stored_column_types, column_name: str, new_type: str) -> Tuple[str, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Update a column's type.
    
    Args:
        stored_df: DataFrame from state
        stored_column_types: Column types dictionary from state
        column_name: Name of column to update
        new_type: New type for the column
    
    Returns:
        Tuple of (status_message, updated_types_df, updated_df, updated_column_types)
    """
    try:
        if stored_df is None or not stored_column_types:
            return "Please upload a file first", None, None, {}
        
        if not column_name or not new_type:
            return "Please select both column and type", None, None, {}
        
        # Use stored data
        df = stored_df.copy()
        column_types = stored_column_types.copy()
        
        # Convert column type
        df, msg = dp.convert_column_type(df, column_name, new_type)
        
        # Update column types dictionary
        column_types[column_name] = new_type
        
        # Get updated types (re-detect to show current state)
        updated_types = dp.auto_detect_column_types(df)
        
        # Create DataFrame for display
        types_data = {
            'Column Name': list(updated_types.keys()),
            'Detected Type': list(updated_types.values())
        }
        types_df = pd.DataFrame(types_data)
        
        return msg, types_df, df, column_types
    
    except Exception as e:
        return f"Error: {str(e)}", None, None, {}


def calculate_statistics(stored_df, stored_column_types) -> Tuple[pd.DataFrame, str, pd.DataFrame, str]:
    """
    Calculate and display comprehensive statistics.
    
    Returns:
        Tuple of (numerical_stats_df, categorical_stats_text, missing_values_df, correlation_status)
    """
    try:
        print(f"DEBUG: stored_df is None: {stored_df is None}")
        print(f"DEBUG: stored_column_types: {stored_column_types}")
        
        if stored_df is None or not stored_column_types:
            return None, "Please upload a file first and ensure column types are detected", None, ""
        
        # Use stored DataFrame and column types
        df = stored_df.copy()
        column_types = stored_column_types
        
        print(f"DEBUG: DataFrame shape: {df.shape}")
        print(f"DEBUG: Column types: {list(column_types.keys())}")
        
        # Apply dtype conversions based on user-specified column types
        df = dp.apply_column_type_conversions(df, column_types)
        print("DEBUG: Applied dtype conversions")
        
        # Get filterable columns
        filterable = dp.get_filterable_columns(df, column_types)
        
        # Calculate numerical statistics
        numerical_stats = dp.calculate_numerical_stats(df, filterable['numeric'])
        
        # Calculate categorical statistics
        categorical_stats = dp.calculate_categorical_stats(df, filterable['categorical'])
        
        # Format categorical stats as text
        cat_stats_text = "## Categorical Statistics\n\n"
        if categorical_stats:
            for col, stats in categorical_stats.items():
                cat_stats_text += f"### {col}\n"
                cat_stats_text += f"- **Unique Values**: {stats['Unique Values']}\n"
                cat_stats_text += f"- **Missing**: {stats['Missing']} ({stats['Missing %']})\n"
                cat_stats_text += f"- **Mode**: {stats['Mode']}\n"
                cat_stats_text += "- **Top 5 Values**:\n"
                for value, count in stats['Top 5 Values'].items():
                    cat_stats_text += f"  - {value}: {count}\n"
                cat_stats_text += "\n"
        else:
            cat_stats_text += "No categorical columns found.\n"
        
        # Generate missing value report
        missing_df = dp.generate_missing_value_report(df)
        
        # Correlation status
        corr_status = f"Correlation matrix available for {len(filterable['numeric'])} numerical columns"
        
        return numerical_stats, cat_stats_text, missing_df, corr_status
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in calculate_statistics:")
        print(error_details)
        return None, f"Error: {str(e)}", None, ""






# ==================== MULTI-FILTER FUNCTIONS ====================

def add_filter_to_list(active_filters, stored_df, stored_column_types, filter_column,
                       numeric_min, numeric_max, categorical_values, date_start, date_end):
    """
    Add a new filter to the active filters list.
    
    Returns:
        Tuple of (updated_active_filters, active_filters_display, filtered_preview, filter_count)
    """
    try:
        if not filter_column or stored_df is None:
            return active_filters, "No filters active", None, "0 filters active"
        
        col_type = stored_column_types.get(filter_column)
        
        # Create filter dict
        new_filter = {
            'column': filter_column,
            'type': col_type
        }
        
        # Add type-specific configuration
        if col_type == 'numeric':
            new_filter['min'] = numeric_min
            new_filter['max'] = numeric_max
            new_filter['display'] = f"{filter_column}: {numeric_min} ≤ x ≤ {numeric_max}"
        
        elif col_type == 'categorical':
            if not categorical_values:
                return active_filters, format_active_filters_display(active_filters), None, f"{len(active_filters)} filters active"
            new_filter['values'] = categorical_values
            values_str = ', '.join(categorical_values[:3]) + ('...' if len(categorical_values) > 3 else '')
            new_filter['display'] = f"{filter_column} in [{values_str}]"
        
        elif col_type == 'datetime':
            if not date_start or not date_end:
                return active_filters, format_active_filters_display(active_filters), None, f"{len(active_filters)} filters active"
            new_filter['start_date'] = date_start
            new_filter['end_date'] = date_end
            new_filter['display'] = f"{filter_column}: {date_start} to {date_end}"
        
        # Add to active filters
        active_filters = active_filters + [new_filter]
        
        # Apply all filters and get preview
        filtered_df = apply_all_active_filters(stored_df, active_filters)
        preview = filtered_df.head(20)
        
        # Format display
        display = format_active_filters_display(active_filters)
        count_msg = f"{len(active_filters)} filter(s) active | {len(filtered_df)} of {len(stored_df)} rows"
        
        return active_filters, display, preview, count_msg
    
    except Exception as e:
        print(f"Error adding filter: {str(e)}")
        return active_filters, format_active_filters_display(active_filters), None, f"{len(active_filters)} filters active"


def remove_filter_from_list(active_filters, stored_df, filter_index):
    """
    Remove a filter from the active filters list.
    
    Returns:
        Tuple of (updated_active_filters, active_filters_display, filtered_preview, filter_count)
    """
    try:
        if filter_index < 0 or filter_index >= len(active_filters):
            return active_filters, format_active_filters_display(active_filters), None, f"{len(active_filters)} filters active"
        
        # Remove filter
        active_filters = active_filters[:filter_index] + active_filters[filter_index+1:]
        
        # Apply remaining filters
        if active_filters and stored_df is not None:
            filtered_df = apply_all_active_filters(stored_df, active_filters)
            preview = filtered_df.head(20)
            count_msg = f"{len(active_filters)} filter(s) active | {len(filtered_df)} of {len(stored_df)} rows"
        else:
            preview = stored_df.head(20) if stored_df is not None else None
            count_msg = "No filters active" if not active_filters else f"{len(active_filters)} filters active"
        
        display = format_active_filters_display(active_filters)
        
        return active_filters, display, preview, count_msg
    
    except Exception as e:
        print(f"Error removing filter: {str(e)}")
        return active_filters, format_active_filters_display(active_filters), None, f"{len(active_filters)} filters active"


def apply_all_active_filters(df, active_filters):
    """
    Apply all active filters to the DataFrame.
    
    Returns:
        Filtered DataFrame
    """
    if not active_filters or df is None:
        return df
    
    filtered_df = df.copy()
    
    for filter_config in active_filters:
        column = filter_config['column']
        col_type = filter_config['type']
        
        try:
            if col_type == 'numeric':
                min_val = filter_config['min']
                max_val = filter_config['max']
                filtered_df = filtered_df[
                    (filtered_df[column] >= min_val) & (filtered_df[column] <= max_val)
                ]
            
            elif col_type == 'categorical':
                values = filter_config['values']
                filtered_df = filtered_df[filtered_df[column].astype(str).isin(values)]
            
            elif col_type == 'datetime':
                start_date = pd.to_datetime(filter_config['start_date'])
                end_date = pd.to_datetime(filter_config['end_date'])
                filtered_df[column] = pd.to_datetime(filtered_df[column], errors='coerce')
                filtered_df = filtered_df[
                    (filtered_df[column] >= start_date) & (filtered_df[column] <= end_date)
                ]
        
        except Exception as e:
            print(f"Error applying filter on column '{column}': {str(e)}")
            continue
    
    return filtered_df


def format_active_filters_display(active_filters):
    """
    Format active filters for display.
    
    Returns:
        Markdown formatted string
    """
    if not active_filters:
        return "**No filters active**\n\nClick 'Add Filter' to filter your data."
    
    display = f"**{len(active_filters)} Active Filter(s)**\n\n"
    
    for i, filter_config in enumerate(active_filters):
        display += f"{i+1}. ✓ {filter_config['display']}\n"
    
    display += "\n*Use the 'Remove Filter' buttons below to remove individual filters*"
    
    return display


def clear_all_active_filters(stored_df):
    """
    Clear all active filters.
    
    Returns:
        Tuple of (empty_filters_list, display_message, full_data_preview, status_message)
    """
    try:
        preview = stored_df.head(20) if stored_df is not None else None
        display = format_active_filters_display([])
        count_msg = f"No filters active | {len(stored_df)} total rows" if stored_df is not None else "No data loaded"
        
        return [], display, preview, count_msg
    
    except Exception as e:
        return [], "No filters active", None, "Error clearing filters"


def export_multi_filtered_data(stored_df, active_filters):
    """
    Export data with all active filters applied.
    
    Returns:
        File path to exported CSV
    """
    try:
        if stored_df is None:
            return None
        
        # Apply all active filters
        filtered_df = apply_all_active_filters(stored_df, active_filters)
        
        # Export to CSV
        output_path = "filtered_data.csv"
        filtered_df.to_csv(output_path, index=False)
        
        return output_path
    
    except Exception as e:
        print(f"Error exporting data: {str(e)}")
        return None


def setup_filters(stored_df, stored_column_types) -> Tuple[gr.update, str]:
    """
    Setup filter controls by populating column dropdown.
    
    Returns:
        Tuple of (column_dropdown_update, status_message)
    """
    try:
        if stored_df is None or not stored_column_types:
            return gr.update(choices=[]), "Please upload a file first"
        
        # Get all columns
        columns = list(stored_column_types.keys())
        
        status = f"Ready to filter! {len(columns)} columns available"
        
        return gr.update(choices=columns, value=None), status
    
    except Exception as e:
        return gr.update(choices=[]), f"Error: {str(e)}"


def update_filter_controls(stored_df, stored_column_types, selected_column):
    """
    Update filter controls based on selected column type.
    
    Returns:
        Tuple of updates for all filter controls
    """
    try:
        print(f"\n=== DEBUG update_filter_controls ===")
        print(f"Selected column: {selected_column}")
        print(f"DataFrame is None: {stored_df is None}")
        print(f"Column types dict: {stored_column_types}")
        
        if not selected_column or stored_df is None or not stored_column_types:
            print("DEBUG: Returning empty state (no selection or no data)")
            return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                   gr.update(value=0, visible=False), gr.update(value=100, visible=False), 
                   gr.update(choices=[], value=None, interactive=True), 
                   gr.update(visible=False), gr.update(visible=False))
        
        col_type = stored_column_types.get(selected_column)
        df = stored_df
        
        print(f"Column type for '{selected_column}': {col_type}")
        
        if col_type == "numeric":
            # Get min/max for numeric column
            min_val = float(df[selected_column].min())
            max_val = float(df[selected_column].max())
            print(f"DEBUG: Numeric column - min: {min_val}, max: {max_val}")
            return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                   gr.update(value=min_val, visible=True), gr.update(value=max_val, visible=True), 
                   gr.update(choices=[], value=None, interactive=True),
                   gr.update(visible=False), gr.update(visible=False))
        
        elif col_type == "categorical":
            # Get unique values for categorical column
            unique_vals = sorted(df[selected_column].dropna().unique().astype(str).tolist())
            print(f"DEBUG: Categorical column")
            print(f"  - Number of unique values: {len(unique_vals)}")
            print(f"  - First 5 values: {unique_vals[:5]}")
            print(f"  - Updating dropdown with choices={len(unique_vals)} items, value=None, interactive=True")
            
            result = (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                   gr.update(value=0, visible=False), gr.update(value=100, visible=False), 
                   gr.update(choices=unique_vals, value=None, interactive=True),
                   gr.update(visible=False), gr.update(visible=False))
            
            print(f"  - Categorical values update object: {result[5]}")
            return result
        
        elif col_type == "datetime":
            # Show date filter
            print("DEBUG: DateTime column")
            return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                   gr.update(value=0, visible=False), gr.update(value=100, visible=False), 
                   gr.update(choices=[], value=None, interactive=True),
                   gr.update(visible=True), gr.update(visible=True))
        
        else:  # text
            print(f"DEBUG: Text column (or unknown type: {col_type})")
            return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                   gr.update(value=0, visible=False), gr.update(value=100, visible=False), 
                   gr.update(choices=[], value=None, interactive=True),
                   gr.update(visible=False), gr.update(visible=False))
    
    except Exception as e:
        print(f"ERROR in update_filter_controls: {str(e)}")
        import traceback
        traceback.print_exc()
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
               gr.update(value=0, visible=False), gr.update(value=100, visible=False), 
               gr.update(choices=[], value=None, interactive=True),
               gr.update(visible=False), gr.update(visible=False))



def apply_filters(stored_df, stored_column_types, filter_column, 
                 numeric_min, numeric_max, categorical_values, date_start, date_end):
    """
    Apply filters to the DataFrame and return preview.
    
    Returns:
        Tuple of (status_message, filtered_dataframe_preview)
    """
    try:
        if stored_df is None:
            return "Please upload a file first", None
        
        if not filter_column:
            return "Please select a column to filter", None
        
        df = stored_df.copy()
        col_type = stored_column_types.get(filter_column)
        
        # Apply appropriate filter based on column type
        if col_type == "numeric":
            df = df[(df[filter_column] >= numeric_min) & (df[filter_column] <= numeric_max)]
            status = f"Filtered '{filter_column}': {numeric_min} to {numeric_max} → {len(df)} rows"
        
        elif col_type == "categorical":
            if categorical_values:
                df = df[df[filter_column].astype(str).isin(categorical_values)]
                status = f"Filtered '{filter_column}': {len(categorical_values)} values selected → {len(df)} rows"
            else:
                status = "No categorical values selected"
        
        elif col_type == "datetime":
            if date_start and date_end:
                df[filter_column] = pd.to_datetime(df[filter_column], errors="coerce")
                start = pd.to_datetime(date_start)
                end = pd.to_datetime(date_end)
                df = df[(df[filter_column] >= start) & (df[filter_column] <= end)]
                status = f"Filtered '{filter_column}': {date_start} to {date_end} → {len(df)} rows"
            else:
                status = "Please provide both start and end dates"
        
        else:
            status = f"Filtering not supported for '{col_type}' columns"
        
        # Return preview (first 20 rows)
        preview = df.head(20)
        return status, preview
    
    except Exception as e:
        return f"Error applying filter: {str(e)}", None


def reset_filters(stored_df):
    """
    Reset all filters and show full dataset.
    
    Returns:
        Tuple of (status_message, full_dataframe_preview, reset_column_dropdown)
    """
    try:
        if stored_df is None:
            return "No data loaded", None, gr.update(value=None)
        
        preview = stored_df.head(20)
        status = f"Filters reset. Showing all {len(stored_df)} rows"
        
        return status, preview, gr.update(value=None)
    
    except Exception as e:
        return f"Error: {str(e)}", None, gr.update(value=None)


def export_filtered_data(stored_df, stored_column_types, filter_column,
                        numeric_min, numeric_max, categorical_values, date_start, date_end):
    """
    Export filtered data to CSV.
    
    Returns:
        File path to exported CSV
    """
    try:
        if stored_df is None:
            return None
        
        # Apply the same filters
        df = stored_df.copy()
        
        if filter_column:
            col_type = stored_column_types.get(filter_column)
            
            if col_type == 'numeric':
                df = df[(df[filter_column] >= numeric_min) & (df[filter_column] <= numeric_max)]
            elif col_type == 'categorical' and categorical_values:
                df = df[df[filter_column].astype(str).isin(categorical_values)]
            elif col_type == 'datetime' and date_start and date_end:
                df[filter_column] = pd.to_datetime(df[filter_column], errors='coerce')
                start = pd.to_datetime(date_start)
                end = pd.to_datetime(date_end)
                df = df[(df[filter_column] >= start) & (df[filter_column] <= end)]
        
        # Export to CSV
        output_path = "filtered_data.csv"
        df.to_csv(output_path, index=False)
        
        return output_path
    
    except Exception as e:
        print(f"Error exporting data: {str(e)}")
        return None


def create_correlation_plot(stored_df, stored_column_types) -> plt.Figure:
    """
    Create correlation heatmap.
    
    Returns:
        Matplotlib figure
    """
    try:
        if stored_df is None or not stored_column_types:
            fig, ax = plt.subplots(figsize=config.CHART_FIGURE_SIZE)
            ax.text(0.5, 0.5, 'Please upload a file first', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Use stored data
        df = stored_df.copy()
        column_types = stored_column_types
        filterable = dp.get_filterable_columns(df, column_types)
        
        # Create correlation heatmap
        fig = viz.create_correlation_heatmap(df, filterable['numeric'])
        
        return fig
    
    except Exception as e:
        print(f"Error creating correlation plot: {str(e)}")
        return plt.figure()


def generate_overview_charts(stored_df, stored_column_types) -> Tuple[plt.Figure, plt.Figure, plt.Figure, plt.Figure]:
    """
    Generate overview visualizations.
    
    Returns:
        Tuple of (missing_plot, numerical_plot, categorical_plot, correlation_plot)
    """
    try:
        if stored_df is None or not stored_column_types:
            empty_fig = plt.figure()
            return empty_fig, empty_fig, empty_fig, empty_fig
        
        # Use stored data
        df = stored_df.copy()
        column_types = stored_column_types
        filterable = dp.get_filterable_columns(df, column_types)
        
        # Create charts
        missing_fig = viz.create_missing_value_chart(df)
        numerical_fig = viz.create_numerical_distributions(df, filterable['numeric'])
        categorical_fig = viz.create_categorical_distributions(df, filterable['categorical'])
        correlation_fig = viz.create_correlation_heatmap(df, filterable['numeric'])
        
        return missing_fig, numerical_fig, categorical_fig, correlation_fig
    
    except Exception as e:
        print(f"Error generating overview charts: {str(e)}")
        empty_fig = plt.figure()
        return empty_fig, empty_fig, empty_fig, empty_fig


def get_column_choices(stored_df, stored_column_types, col_type: str = 'all') -> gr.update:
    """
    Get column choices for dropdowns.
    
    Args:
        file: Uploaded file
        col_type: Type of columns to return ('numeric', 'categorical', 'datetime', 'all')
    
    Returns:
        Gradio dropdown update
    """
    try:
        if stored_df is None or not stored_column_types:
            return gr.update(choices=[])
        
        df = stored_df
        column_types = stored_column_types
        filterable = dp.get_filterable_columns(df, column_types)
        
        if col_type == 'numeric':
            return gr.update(choices=filterable['numeric'])
        elif col_type == 'categorical':
            return gr.update(choices=filterable['categorical'])
        elif col_type == 'datetime':
            return gr.update(choices=filterable['datetime'])
        else:
            return gr.update(choices=list(df.columns))
    
    except Exception as e:
        return gr.update(choices=[])


def create_custom_chart(stored_df, stored_column_types, chart_type: str, x_col: str, y_col: str = None, 
                       color_col: str = None, agg_method: str = 'mean') -> go.Figure:
    """
    Create custom visualization based on user selections.
    
    Returns:
        Plotly figure
    """
    try:
        if stored_df is None or not x_col:
            fig = go.Figure()
            fig.add_annotation(text="Please upload a file and select columns", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Use stored data
        df = stored_df.copy()
        column_types = stored_column_types
        
        # Create appropriate chart based on type
        if chart_type == "Time Series":
            if x_col and y_col:
                return viz.create_time_series_plot(df, x_col, y_col, agg_method)
        
        elif chart_type == "Distribution":
            if x_col:
                return viz.create_distribution_plot(df, x_col, 'histogram')
        
        elif chart_type == "Box Plot":
            if x_col:
                return viz.create_distribution_plot(df, x_col, 'box')
        
        elif chart_type == "Bar Chart":
            if x_col:
                return viz.create_category_analysis(df, x_col, y_col, 'bar', agg_method)
        
        elif chart_type == "Pie Chart":
            if x_col:
                return viz.create_category_analysis(df, x_col, y_col, 'pie', agg_method)
        
        elif chart_type == "Scatter Plot":
            if x_col and y_col:
                return viz.create_scatter_plot(df, x_col, y_col, color_col)
        
        # Default empty figure
        fig = go.Figure()
        fig.add_annotation(text="Please select appropriate columns for this chart type", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig
    
    except Exception as e:
        print(f"Error creating custom chart: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig


def generate_insights_report(stored_df, stored_column_types) -> Tuple[str, str, pd.DataFrame]:
    """
    Generate comprehensive insights report.
    
    Returns:
        Tuple of (summary_text, top_insights_text, performers_df)
    """
    try:
        if stored_df is None or not stored_column_types:
            return "Please upload a file first", "", None
        
        # Use stored data
        df = stored_df.copy()
        column_types = stored_column_types
        filterable = dp.get_filterable_columns(df, column_types)
        
        # Generate summary insights
        summary = ins.generate_summary_insights(df, column_types)
        
        # Generate top insights
        top_insights = ins.generate_top_insights_text(df, column_types, top_n=5)
        
        # Get top/bottom performers for numerical columns
        performers_data = []
        if filterable['numeric']:
            top_performers = ins.identify_top_bottom_performers(df, filterable['numeric'][:3], n=5)
            
            for col, data in top_performers.items():
                # Add top performers
                for idx, row in data['top'].iterrows():
                    performers_data.append({
                        'Metric': col,
                        'Type': 'Top',
                        'Value': f"{row[col]:.2f}" if pd.notna(row[col]) else "N/A",
                        'Rank': len([p for p in performers_data if p['Metric'] == col and p['Type'] == 'Top']) + 1
                    })
                
                # Add bottom performers
                for idx, row in data['bottom'].iterrows():
                    performers_data.append({
                        'Metric': col,
                        'Type': 'Bottom',
                        'Value': f"{row[col]:.2f}" if pd.notna(row[col]) else "N/A",
                        'Rank': len([p for p in performers_data if p['Metric'] == col and p['Type'] == 'Bottom']) + 1
                    })
        
        performers_df = pd.DataFrame(performers_data) if performers_data else None
        
        return summary, top_insights, performers_df
    
    except Exception as e:
        return f"Error: {str(e)}", "", None


def create_dashboard():
    """
    Create and configure the main Gradio dashboard interface.
    
    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title="Business Intelligence Dashboard") as demo:
        
        # Header
        gr.Markdown("""
        # 📊 Business Intelligence Dashboard
        ### Interactive Data Analysis & Visualization Platform
        
        Upload your dataset to explore data, generate insights, and create visualizations.
        """)
        
        # Shared file state
        uploaded_file = gr.State(None)
        stored_df = gr.State(None)  # State to store DataFrame
        stored_column_types = gr.State({})  # State to store column types
        active_filters = gr.State([])  # State to store active filters list
        
        # ============= TAB 1: DATA UPLOAD =============
        with gr.Tab("📁 Data Upload"):
            gr.Markdown("### Upload Your Dataset")
            gr.Markdown("Supported formats: CSV, Excel (.xlsx, .xls)")
            
            # File Upload Section (Static)
            gr.Markdown("#### 📤 File Upload")
            file_upload = gr.File(
                label="Upload File",
                file_types=['.csv', '.xlsx', '.xls'],
                type="filepath"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    pass
                with gr.Column(scale=1):
                    upload_btn = gr.Button("Load Dataset", variant="primary", size="lg")
                with gr.Column(scale=1):
                    pass
            
            upload_status = gr.Textbox(label="Status", interactive=False, scale=2)
            
            # Dataset Information and Summary side-by-side
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 📋 Dataset Information")
                    dataset_info = gr.Dataframe(
                        label="Dataset Information",
                        interactive=False,
                        max_height=300,
                        wrap=True,
                        column_widths=["50%", "50%"]
                    )
                with gr.Column(scale=1):
                    gr.Markdown("#### 📝 Dataset Summary")
                    summary_text = gr.Markdown()
            
            # Data Preview (Static)
            gr.Markdown("#### 📊 Data Preview")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**First "+str(config.DEFAULT_PREVIEW_ROWS)+" Rows**")
                    head_preview = gr.Dataframe(label="Head", interactive=False, max_height=500)
                with gr.Column():
                    gr.Markdown("**Last "+str(config.DEFAULT_PREVIEW_ROWS)+" Rows**")
                    tail_preview = gr.Dataframe(label="Tail", interactive=False, max_height=500)
            
            # Connect upload functionality
            upload_btn.click(
                fn=upload_and_preview,
                inputs=[file_upload],
                outputs=[upload_status, head_preview, tail_preview, summary_text, dataset_info, stored_df, stored_column_types]
                ).then(
                    fn=lambda x: x,
                    inputs=[file_upload],
                    outputs=[uploaded_file]
                )
        
        # ============= TAB 2: COLUMN TYPE MANAGEMENT =============
        with gr.Tab("🔧 Column Type Management"):
            gr.Markdown("## Manage Column Data Types")
            gr.Markdown("**Auto-detected column types are displayed below. You can modify them if needed.**")
            
            # gr.Markdown("#### Auto-Detected Column Types")
            column_types_display = gr.Dataframe(
                label="Auto-Detected Column Types",
                interactive=False,
                max_height=400,
                wrap=True
            )
            type_status = gr.Textbox(label="Status", interactive=False, value="Upload a dataset to see column types")
            
            gr.Markdown("### Update Column Type")
            with gr.Row():
                with gr.Column(scale=2):
                    column_selector = gr.Dropdown(
                        label="Select Column to Modify",
                        choices=[],
                        interactive=True
                    )
                with gr.Column(scale=2):
                    type_selector = gr.Dropdown(
                        label="Select Type",
                        choices=['numeric', 'categorical', 'datetime', 'text'],
                        interactive=True
                    )
                
            update_type_btn = gr.Button("Update Type", variant="secondary", size="lg")
            
            # Connect Column Type Management functionality
            update_type_btn.click(
                fn=update_column_type,
                inputs=[stored_df, stored_column_types, column_selector, type_selector],
                outputs=[type_status, column_types_display, stored_df, stored_column_types]
            )
            
            # Auto-populate column types when dataset is uploaded
            upload_btn.click(
                fn=show_column_types,
                inputs=[stored_df, stored_column_types],
                outputs=[column_types_display, type_status]
            ).then(
                fn=get_columns_from_file,
                inputs=[file_upload],
                outputs=[column_selector]
            )
        
        # ============= TAB 3: STATISTICS =============
        with gr.Tab("📈 Statistics"):
            gr.Markdown("### Comprehensive Statistical Analysis")
            
            calc_stats_btn = gr.Button("Calculate Statistics", variant="primary", size="lg")
            
            with gr.Accordion("📊 Numerical Statistics", open=True):
                numerical_stats_display = gr.Dataframe(
                    label="Numerical Column Statistics",
                    interactive=False,
                    max_height=400
                )
            
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("📝 Categorical Statistics", open=True):
                        categorical_stats_display = gr.Markdown()
                
                with gr.Column():
                    with gr.Accordion("❓ Missing Values Report", open=True):
                        missing_values_display = gr.Dataframe(
                            label="Missing Values by Column",
                            interactive=False,
                            max_height=400
                        )
            
            with gr.Accordion("🔗 Correlation Analysis", open=True):
                correlation_status = gr.Textbox(label="Status", interactive=False)
                correlation_plot = gr.Plot(label="Correlation Heatmap")
            
            # Connect statistics functionality
            calc_stats_btn.click(
                fn=calculate_statistics,
        inputs=[stored_df, stored_column_types],
                outputs=[numerical_stats_display, categorical_stats_display, 
                        missing_values_display, correlation_status]
            ).then(
                fn=create_correlation_plot,
                inputs=[stored_df, stored_column_types],
                outputs=[correlation_plot]
            )
        
        # ============= TAB 4: FILTER & EXPLORE =============
        with gr.Tab("🔍 Filter & Explore"):
            gr.Markdown("### Filter Data and Export")
            gr.Markdown("Add multiple filters to explore specific subsets of your data.")
            
            with gr.Row():
                # Left Column: Filter Controls
                with gr.Column(scale=1):
                    # Add New Filter Section
                    with gr.Accordion("➕ Add New Filter", open=True):
                        # Column selection
                        filter_column = gr.Dropdown(
                            label="1. Select Column",
                            choices=[],
                            interactive=True
                        )
                        
                        # Dynamic filter controls based on type
                        # Numeric
                        with gr.Group(visible=False) as numeric_filter_group:
                            gr.Markdown("**2. Set Range**")
                            with gr.Row():
                                numeric_min = gr.Number(label="Min", value=0)
                                numeric_max = gr.Number(label="Max", value=100)
                        
                        # Categorical
                        with gr.Group(visible=False) as categorical_filter_group:
                            gr.Markdown("**2. Select Values**")
                            categorical_values = gr.Dropdown(
                                label="Values",
                                choices=[],
                                multiselect=True,
                                interactive=True
                            )
                        
                        # Date
                        with gr.Group(visible=False) as date_filter_group:
                            gr.Markdown("**2. Set Date Range**")
                            date_start = gr.Textbox(label="Start Date (YYYY-MM-DD)", placeholder="2020-01-01")
                            date_end = gr.Textbox(label="End Date (YYYY-MM-DD)", placeholder="2020-12-31")
                        
                        add_filter_btn = gr.Button("3. + Add Filter", variant="primary")
                    
                    # Active Filters Section
                    with gr.Accordion("✓ Active Filters", open=True):
                        active_filters_display = gr.Markdown("**No filters active**")
                        
                        with gr.Row():
                            filter_to_remove = gr.Number(label="Filter # to Remove", value=1, precision=0, minimum=1)
                            remove_filter_btn = gr.Button("Remove Filter", size="sm", variant="stop")
                        
                        clear_all_btn = gr.Button("Clear All Filters", variant="secondary")
                
                # Right Column: Results
                with gr.Column(scale=2):
                    filter_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                    filtered_preview = gr.Dataframe(
                        label="Filtered Data Preview (Top 20 rows)",
                        interactive=False
                    )
            
            export_btn = gr.Button("📥 Export Filtered Data (CSV)", variant="primary")
            export_file = gr.File(label="Download", interactive=False)
            
            # Event Handlers
            
            # 1. Update controls when column changes
            filter_column.change(
                fn=update_filter_controls,
                inputs=[stored_df, stored_column_types, filter_column],
                outputs=[numeric_filter_group, categorical_filter_group, date_filter_group,
                        numeric_min, numeric_max, categorical_values, date_start, date_end]
            ).then(
                # Force re-render of categorical dropdown with a second update
                fn=lambda df, types, col: (
                    gr.update(
                        choices=sorted(df[col].dropna().unique().astype(str).tolist()) 
                        if col and types.get(col) == 'categorical' and df is not None 
                        else [],
                        value=None
                    )
                ),
                inputs=[stored_df, stored_column_types, filter_column],
                outputs=[categorical_values]
            )
            
            # 2. Add Filter
            add_filter_btn.click(
                fn=add_filter_to_list,
                inputs=[active_filters, stored_df, stored_column_types, filter_column,
                       numeric_min, numeric_max, categorical_values, date_start, date_end],
                outputs=[active_filters, active_filters_display, filtered_preview, filter_status]
            )
            
            # 3. Remove Filter
            remove_filter_btn.click(
                fn=lambda filters, df, idx: remove_filter_from_list(filters, df, int(idx)-1),
                inputs=[active_filters, stored_df, filter_to_remove],
                outputs=[active_filters, active_filters_display, filtered_preview, filter_status]
            )
            
            # 4. Clear All
            clear_all_btn.click(
                fn=clear_all_active_filters,
                inputs=[stored_df],
                outputs=[active_filters, active_filters_display, filtered_preview, filter_status]
            )
            
            # 5. Export
            export_btn.click(
                fn=export_multi_filtered_data,
                inputs=[stored_df, active_filters],
                outputs=[export_file]
            )
            
            # Auto-populate filter column dropdown when dataset is uploaded
            upload_btn.click(
                fn=setup_filters,
                inputs=[stored_df, stored_column_types],
                outputs=[filter_column, filter_status]
            )
        
        # ============= TAB 5: VISUALIZATIONS - OVERVIEW =============
        with gr.Tab("📊 Visualizations - Overview"):
            gr.Markdown("### Automated Data Visualizations")
            gr.Markdown("Generate comprehensive overview charts of your dataset")
            
            generate_charts_btn = gr.Button("Generate Overview Charts", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Missing Values")
                    missing_plot = gr.Plot()
                with gr.Column():
                    gr.Markdown("#### Numerical Distributions")
                    numerical_plot = gr.Plot()
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Categorical Distributions")
                    categorical_plot = gr.Plot()
                with gr.Column():
                    gr.Markdown("#### Correlation Heatmap")
                    correlation_plot_overview = gr.Plot()
            
            # Connect overview charts
            generate_charts_btn.click(
                fn=generate_overview_charts,
                inputs=[stored_df, stored_column_types],
                outputs=[missing_plot, numerical_plot, categorical_plot, correlation_plot_overview]
            )
        
        # ============= TAB 6: VISUALIZATIONS - CUSTOM =============
        with gr.Tab("🎨 Visualizations - Custom"):
            gr.Markdown("### Create Custom Visualizations")
            gr.Markdown("Build your own charts by selecting columns and chart types")
            
            with gr.Row():
                chart_type = gr.Radio(
                    label="Chart Type",
                    choices=["Time Series", "Distribution", "Box Plot", "Bar Chart", "Pie Chart", "Scatter Plot"],
                    value="Bar Chart"
                )
            
            with gr.Row():
                x_column = gr.Dropdown(label="X-Axis / Category Column", choices=[])
                y_column = gr.Dropdown(label="Y-Axis / Value Column (optional)", choices=[])
                color_column = gr.Dropdown(label="Color By (optional)", choices=[])
            
            with gr.Row():
                agg_method = gr.Radio(
                    label="Aggregation Method",
                    choices=['sum', 'mean', 'count', 'median'],
                    value='mean'
                )
            
            create_chart_btn = gr.Button("Create Chart", variant="primary", size="lg")
            custom_chart = gr.Plot(label="Custom Chart")
            

            
            # Update column dropdowns when data is uploaded (via stored_df change)
            upload_btn.click(
                fn=lambda df, types: get_column_choices(df, types, 'all'),
                inputs=[stored_df, stored_column_types],
                outputs=[x_column]
            ).then(
                fn=lambda df, types: get_column_choices(df, types, 'all'),
                inputs=[stored_df, stored_column_types],
                outputs=[y_column]
            ).then(
                fn=lambda df, types: get_column_choices(df, types, 'all'),
                inputs=[stored_df, stored_column_types],
                outputs=[color_column]
            )
            
            # Connect custom chart creation
            create_chart_btn.click(
                fn=create_custom_chart,
                inputs=[stored_df, stored_column_types, chart_type, x_column, y_column, color_column, agg_method],
                outputs=[custom_chart]
            )
        
        # ============= TAB 7: INSIGHTS =============
        with gr.Tab("💡 Insights"):
            gr.Markdown("### Automated Insights & Pattern Detection")
            gr.Markdown("Discover key patterns, outliers, and trends in your data")
            
            generate_insights_btn = gr.Button("Generate Insights", variant="primary", size="lg")
            
            gr.Markdown("#### Key Insights")
            top_insights_display = gr.Markdown()
            
            gr.Markdown("#### Detailed Analysis")
            summary_insights_display = gr.Markdown()
            
            gr.Markdown("#### Top/Bottom Performers")
            performers_display = gr.Dataframe(label="Performance Metrics", interactive=False)
            
            # Connect insights generation
            generate_insights_btn.click(
                fn=generate_insights_report,
                inputs=[stored_df, stored_column_types],
                outputs=[summary_insights_display, top_insights_display, performers_display]
            )
        
        # Footer
        gr.Markdown("""
        ---
        **Business Intelligence Dashboard** | Built with Gradio, pandas, matplotlib, seaborn, and plotly
        """)
    
    return demo


# Main execution
if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(
        server_name="127.0.0.1",  # Use 0.0.0.0 for HF Spaces compatibility
        server_port=7860,  # Default Gradio port for HF Spaces
        share=False,  # Not needed on HF Spaces (auto-handled)
        show_error=True,
        ssr_mode=False,  # Disable SSR to prevent event loop issues
        theme=gr.themes.Soft()
    )