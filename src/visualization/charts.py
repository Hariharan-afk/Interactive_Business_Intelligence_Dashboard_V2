"""
Visualization module for Business Intelligence Dashboard.
Handles creation of all charts and graphs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Tuple
import config


# Set default style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette(config.DEFAULT_COLOR_PALETTE)


def create_missing_value_chart(df: pd.DataFrame) -> plt.Figure:
    """
    Create bar chart showing missing values by column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Matplotlib figure
    """
    try:
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
        
        if len(missing_data) == 0:
            # No missing values
            fig, ax = plt.subplots(figsize=config.CHART_FIGURE_SIZE)
            ax.text(0.5, 0.5, 'No Missing Values Found!', 
                   ha='center', va='center', fontsize=16, color='green')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        fig, ax = plt.subplots(figsize=config.CHART_FIGURE_SIZE)
        missing_data.plot(kind='barh', ax=ax, color=config.COLORS['warning'])
        ax.set_xlabel('Number of Missing Values', fontsize=12)
        ax.set_ylabel('Column Name', fontsize=12)
        ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error creating missing value chart: {str(e)}")
        return plt.figure()


def create_numerical_distributions(df: pd.DataFrame, numerical_cols: List[str], 
                                   max_cols: int = 6) -> plt.Figure:
    """
    Create histograms for numerical columns.
    
    Args:
        df: Input DataFrame
        numerical_cols: List of numerical column names
        max_cols: Maximum number of columns to plot
        
    Returns:
        Matplotlib figure
    """
    try:
        if not numerical_cols:
            fig, ax = plt.subplots(figsize=config.CHART_FIGURE_SIZE)
            ax.text(0.5, 0.5, 'No Numerical Columns Found', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Limit to max_cols
        cols_to_plot = numerical_cols[:max_cols]
        n_cols = len(cols_to_plot)
        
        # Calculate grid dimensions
        n_rows = (n_cols + 1) // 2
        n_plot_cols = min(n_cols, 2)
        
        fig, axes = plt.subplots(n_rows, n_plot_cols, 
                                figsize=(12, 4 * n_rows))
        
        if n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, col in enumerate(cols_to_plot):
            data = df[col].dropna()
            
            if len(data) > 0:
                axes[idx].hist(data, bins=30, color=config.CHART_COLORS[idx % len(config.CHART_COLORS)],
                             edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error creating numerical distributions: {str(e)}")
        return plt.figure()


def create_categorical_distributions(df: pd.DataFrame, categorical_cols: List[str],
                                    max_cols: int = 6, top_n: int = 10) -> plt.Figure:
    """
    Create bar charts for categorical columns.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
        max_cols: Maximum number of columns to plot
        top_n: Show only top N categories per column
        
    Returns:
        Matplotlib figure
    """
    try:
        if not categorical_cols:
            fig, ax = plt.subplots(figsize=config.CHART_FIGURE_SIZE)
            ax.text(0.5, 0.5, 'No Categorical Columns Found', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Limit to max_cols
        cols_to_plot = categorical_cols[:max_cols]
        n_cols = len(cols_to_plot)
        
        # Calculate grid dimensions
        n_rows = (n_cols + 1) // 2
        n_plot_cols = min(n_cols, 2)
        
        fig, axes = plt.subplots(n_rows, n_plot_cols, 
                                figsize=(12, 4 * n_rows))
        
        if n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, col in enumerate(cols_to_plot):
            value_counts = df[col].value_counts().head(top_n)
            
            if len(value_counts) > 0:
                axes[idx].barh(range(len(value_counts)), value_counts.values,
                             color=config.CHART_COLORS[idx % len(config.CHART_COLORS)])
                axes[idx].set_yticks(range(len(value_counts)))
                axes[idx].set_yticklabels(value_counts.index)
                axes[idx].set_title(f'Top {min(top_n, len(value_counts))} in {col}', 
                                  fontweight='bold')
                axes[idx].set_xlabel('Count')
                axes[idx].grid(axis='x', alpha=0.3)
                
                # Add value labels on bars
                for i, v in enumerate(value_counts.values):
                    axes[idx].text(v, i, f' {v}', va='center')
        
        # Hide unused subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error creating categorical distributions: {str(e)}")
        return plt.figure()


def create_correlation_heatmap(df: pd.DataFrame, numerical_cols: List[str]) -> plt.Figure:
    """
    Create correlation heatmap for numerical columns.
    
    Args:
        df: Input DataFrame
        numerical_cols: List of numerical column names
        
    Returns:
        Matplotlib figure
    """
    try:
        if not numerical_cols or len(numerical_cols) < 2:
            fig, ax = plt.subplots(figsize=config.CHART_FIGURE_SIZE)
            ax.text(0.5, 0.5, 'Need at least 2 numerical columns for correlation', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=config.HEATMAP_FIGURE_SIZE)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax, vmin=-1, vmax=1)
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error creating correlation heatmap: {str(e)}")
        return plt.figure()


def create_time_series_plot(df: pd.DataFrame, date_col: str, value_col: str, 
                           agg_method: str = 'mean') -> go.Figure:
    """
    Create time series line plot.
    
    Args:
        df: Input DataFrame
        date_col: Date column name
        value_col: Value column name
        agg_method: Aggregation method ('sum', 'mean', 'count', 'median')
        
    Returns:
        Plotly figure
    """
    try:
        # Prepare data
        df_copy = df[[date_col, value_col]].copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy = df_copy.dropna()
        
        # Group by date and aggregate
        if agg_method == 'sum':
            df_agg = df_copy.groupby(date_col)[value_col].sum().reset_index()
        elif agg_method == 'mean':
            df_agg = df_copy.groupby(date_col)[value_col].mean().reset_index()
        elif agg_method == 'median':
            df_agg = df_copy.groupby(date_col)[value_col].median().reset_index()
        elif agg_method == 'count':
            df_agg = df_copy.groupby(date_col)[value_col].count().reset_index()
        else:
            df_agg = df_copy.groupby(date_col)[value_col].mean().reset_index()
        
        # Create plot
        fig = px.line(df_agg, x=date_col, y=value_col,
                     title=f'{value_col} over Time ({agg_method.capitalize()})',
                     labels={date_col: 'Date', value_col: value_col})
        
        fig.update_traces(line_color=config.COLORS['primary'], line_width=2)
        fig.update_layout(
            hovermode='x unified',
            plot_bgcolor='white',
            font=dict(size=12)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    except Exception as e:
        print(f"Error creating time series plot: {str(e)}")
        return go.Figure()


def create_distribution_plot(df: pd.DataFrame, column: str, 
                            plot_type: str = 'histogram') -> go.Figure:
    """
    Create distribution plot (histogram or box plot).
    
    Args:
        df: Input DataFrame
        column: Column name
        plot_type: 'histogram' or 'box'
        
    Returns:
        Plotly figure
    """
    try:
        data = df[column].dropna()
        
        if len(data) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        if plot_type == 'histogram':
            fig = px.histogram(df, x=column, nbins=30,
                             title=f'Distribution of {column}',
                             labels={column: column})
            fig.update_traces(marker_color=config.COLORS['primary'])
        else:  # box plot
            fig = px.box(df, y=column, title=f'Box Plot of {column}',
                        labels={column: column})
            fig.update_traces(marker_color=config.COLORS['primary'])
        
        fig.update_layout(
            plot_bgcolor='white',
            font=dict(size=12)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    except Exception as e:
        print(f"Error creating distribution plot: {str(e)}")
        return go.Figure()


def create_category_analysis(df: pd.DataFrame, category_col: str, 
                            value_col: Optional[str] = None,
                            chart_type: str = 'bar', 
                            agg_method: str = 'count',
                            top_n: int = 15) -> go.Figure:
    """
    Create category analysis chart (bar or pie).
    
    Args:
        df: Input DataFrame
        category_col: Category column name
        value_col: Value column (optional, for aggregation)
        chart_type: 'bar' or 'pie'
        agg_method: Aggregation method if value_col provided
        top_n: Show top N categories
        
    Returns:
        Plotly figure
    """
    try:
        if value_col and value_col in df.columns:
            # Aggregate by category
            if agg_method == 'sum':
                df_agg = df.groupby(category_col)[value_col].sum().reset_index()
            elif agg_method == 'mean':
                df_agg = df.groupby(category_col)[value_col].mean().reset_index()
            elif agg_method == 'median':
                df_agg = df.groupby(category_col)[value_col].median().reset_index()
            elif agg_method == 'count':
                df_agg = df.groupby(category_col)[value_col].count().reset_index()
            else:
                df_agg = df.groupby(category_col)[value_col].sum().reset_index()
            
            df_agg = df_agg.sort_values(value_col, ascending=False).head(top_n)
            
            if chart_type == 'bar':
                fig = px.bar(df_agg, x=category_col, y=value_col,
                           title=f'{value_col} by {category_col} ({agg_method.capitalize()})',
                           labels={category_col: category_col, value_col: value_col})
            else:  # pie
                fig = px.pie(df_agg, names=category_col, values=value_col,
                           title=f'{value_col} by {category_col} ({agg_method.capitalize()})')
        else:
            # Just count occurrences
            value_counts = df[category_col].value_counts().head(top_n).reset_index()
            value_counts.columns = [category_col, 'count']
            
            if chart_type == 'bar':
                fig = px.bar(value_counts, x=category_col, y='count',
                           title=f'Count by {category_col}',
                           labels={category_col: category_col, 'count': 'Count'})
            else:  # pie
                fig = px.pie(value_counts, names=category_col, values='count',
                           title=f'Distribution of {category_col}')
        
        fig.update_layout(
            plot_bgcolor='white',
            font=dict(size=12)
        )
        
        if chart_type == 'bar':
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    except Exception as e:
        print(f"Error creating category analysis: {str(e)}")
        return go.Figure()


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                       color_col: Optional[str] = None,
                       add_trendline: bool = False) -> go.Figure:
    """
    Create scatter plot.
    
    Args:
        df: Input DataFrame
        x_col: X-axis column name
        y_col: Y-axis column name
        color_col: Optional column for color encoding
        add_trendline: Whether to add trend line
        
    Returns:
        Plotly figure
    """
    try:
        df_plot = df[[x_col, y_col]].dropna()
        
        if color_col and color_col in df.columns:
            df_plot[color_col] = df[color_col]
            df_plot = df_plot.dropna(subset=[color_col])
        
        if len(df_plot) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        trendline = 'ols' if add_trendline else None
        
        if color_col and color_col in df_plot.columns:
            fig = px.scatter(df_plot, x=x_col, y=y_col, color=color_col,
                           title=f'{y_col} vs {x_col}',
                           labels={x_col: x_col, y_col: y_col},
                           trendline=trendline)
        else:
            fig = px.scatter(df_plot, x=x_col, y=y_col,
                           title=f'{y_col} vs {x_col}',
                           labels={x_col: x_col, y_col: y_col},
                           trendline=trendline)
            fig.update_traces(marker=dict(color=config.COLORS['primary']))
        
        fig.update_layout(
            plot_bgcolor='white',
            font=dict(size=12)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    except Exception as e:
        print(f"Error creating scatter plot: {str(e)}")
        return go.Figure()


def create_grouped_bar_chart(df: pd.DataFrame, category_col: str, 
                             group_col: str, value_col: str,
                             agg_method: str = 'sum') -> go.Figure:
    """
    Create grouped bar chart.
    
    Args:
        df: Input DataFrame
        category_col: Main category column
        group_col: Grouping column
        value_col: Value column
        agg_method: Aggregation method
        
    Returns:
        Plotly figure
    """
    try:
        # Aggregate data
        if agg_method == 'sum':
            df_agg = df.groupby([category_col, group_col])[value_col].sum().reset_index()
        elif agg_method == 'mean':
            df_agg = df.groupby([category_col, group_col])[value_col].mean().reset_index()
        elif agg_method == 'median':
            df_agg = df.groupby([category_col, group_col])[value_col].median().reset_index()
        elif agg_method == 'count':
            df_agg = df.groupby([category_col, group_col])[value_col].count().reset_index()
        else:
            df_agg = df.groupby([category_col, group_col])[value_col].sum().reset_index()
        
        # Create grouped bar chart
        fig = px.bar(df_agg, x=category_col, y=value_col, color=group_col,
                    title=f'{value_col} by {category_col} and {group_col} ({agg_method.capitalize()})',
                    labels={category_col: category_col, value_col: value_col, 
                           group_col: group_col},
                    barmode='group')
        
        fig.update_layout(
            plot_bgcolor='white',
            font=dict(size=12)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    except Exception as e:
        print(f"Error creating grouped bar chart: {str(e)}")
        return go.Figure()