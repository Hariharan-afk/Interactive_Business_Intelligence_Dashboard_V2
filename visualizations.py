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


def create_missing_value_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create bar chart showing missing values by column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Plotly figure
    """
    try:
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
        
        if len(missing_data) == 0:
            # No missing values
            fig = go.Figure()
            fig.add_annotation(
                text='No Missing Values Found!',
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20, color='green')
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=400
            )
            return fig
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=missing_data.values,
            y=missing_data.index,
            orientation='h',
            marker=dict(color=config.COLORS['warning']),
            text=missing_data.values,
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Missing: %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text='Missing Values by Column', font=dict(size=16, family='Arial Black')),
            xaxis=dict(title='Number of Missing Values', gridcolor='lightgray'),
            yaxis=dict(title='Column Name'),
            height=max(400, len(missing_data) * 30),
            hovermode='closest',
            plot_bgcolor='white'
        )
        
        return fig
    
    except Exception as e:
        print(f"Error creating missing value chart: {str(e)}")
        return go.Figure()


def create_numerical_distributions(df: pd.DataFrame, numerical_cols: List[str], 
                                   max_cols: int = 6) -> go.Figure:
    """
    Create histograms for numerical columns.
    
    Args:
        df: Input DataFrame
        numerical_cols: List of numerical column names
        max_cols: Maximum number of columns to plot
        
    Returns:
        Plotly figure
    """
    try:
        if not numerical_cols:
            fig = go.Figure()
            fig.add_annotation(
                text='No Numerical Columns Found',
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=400
            )
            return fig
        
        # Limit to max_cols
        cols_to_plot = numerical_cols[:max_cols]
        n_cols = len(cols_to_plot)
        
        # Calculate grid dimensions
        n_plot_cols = min(n_cols, 2)
        n_rows = (n_cols + 1) // 2
        
        # Import make_subplots
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows, 
            cols=n_plot_cols,
            subplot_titles=[f'Distribution of {col}' for col in cols_to_plot],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        colors = config.CHART_COLORS
        
        for idx, col in enumerate(cols_to_plot):
            row = (idx // n_plot_cols) + 1
            col_pos = (idx % n_plot_cols) + 1
            
            data = df[col].dropna()
            
            if len(data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        name=col,
                        marker=dict(
                            color=colors[idx % len(colors)],
                            line=dict(color='black', width=1)
                        ),
                        opacity=0.7,
                        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
                        showlegend=False
                    ),
                    row=row, col=col_pos
                )
                
                fig.update_xaxes(title_text=col, row=row, col=col_pos, gridcolor='lightgray')
                fig.update_yaxes(title_text='Frequency', row=row, col=col_pos, gridcolor='lightgray')
        
        fig.update_layout(
            title=dict(text='Numerical Column Distributions', font=dict(size=18, family='Arial Black')),
            height=400 * n_rows,
            showlegend=False,
            hovermode='closest',
            plot_bgcolor='white'
        )
        
        return fig
    
    except Exception as e:
        print(f"Error creating numerical distributions: {str(e)}")
        return go.Figure()


def create_categorical_distributions(df: pd.DataFrame, categorical_cols: List[str],
                                    max_cols: int = 6, top_n: int = 10) -> go.Figure:
    """
    Create bar charts for categorical columns.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
        max_cols: Maximum number of columns to plot
        top_n: Show only top N categories per column
        
    Returns:
        Plotly figure
    """
    try:
        if not categorical_cols:
            fig = go.Figure()
            fig.add_annotation(
                text='No Categorical Columns Found',
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=400
            )
            return fig
        
        # Limit to max_cols
        cols_to_plot = categorical_cols[:max_cols]
        n_cols = len(cols_to_plot)
        
        # Calculate grid dimensions
        n_plot_cols = min(n_cols, 2)
        n_rows = (n_cols + 1) // 2
        
        # Import make_subplots
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows, 
            cols=n_plot_cols,
            subplot_titles=[f'Top {top_n} in {col}' for col in cols_to_plot],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        colors = config.CHART_COLORS
        
        for idx, col in enumerate(cols_to_plot):
            row = (idx // n_plot_cols) + 1
            col_pos = (idx % n_plot_cols) + 1
            
            value_counts = df[col].value_counts().head(top_n)
            
            if len(value_counts) > 0:
                fig.add_trace(
                    go.Bar(
                        x=value_counts.values,
                        y=value_counts.index,
                        orientation='h',
                        name=col,
                        marker=dict(color=colors[idx % len(colors)]),
                        text=value_counts.values,
                        textposition='auto',
                        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>',
                        showlegend=False
                    ),
                    row=row, col=col_pos
                )
                
                fig.update_xaxes(title_text='Count', row=row, col=col_pos, gridcolor='lightgray')
                fig.update_yaxes(row=row, col=col_pos)
        
        fig.update_layout(
            title=dict(text='Categorical Column Distributions', font=dict(size=18, family='Arial Black')),
            height=400 * n_rows,
            showlegend=False,
            hovermode='closest',
            plot_bgcolor='white'
        )
        
        return fig
    
    except Exception as e:
        print(f"Error creating categorical distributions: {str(e)}")
        return go.Figure()


def create_correlation_heatmap(df: pd.DataFrame, numerical_cols: List[str]) -> go.Figure:
    """
    Create correlation heatmap for numerical columns.
    
    Args:
        df: Input DataFrame
        numerical_cols: List of numerical column names
        
    Returns:
        Plotly figure
    """
    try:
        if not numerical_cols or len(numerical_cols) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text='Need at least 2 numerical columns for correlation',
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=400
            )
            return fig
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Create annotations for heatmap
        annotations = []
        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f'{corr_matrix.iloc[i, j]:.2f}',
                        showarrow=False,
                        font=dict(color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black', size=10)
                    )
                )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',  # Red-Blue reversed (similar to coolwarm)
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Correlation'),
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        # Add annotations
        fig.update_layout(
            title=dict(text='Correlation Heatmap', font=dict(size=18, family='Arial Black')),
            annotations=annotations,
            xaxis=dict(side='bottom', tickangle=-45),
            yaxis=dict(autorange='reversed'),
            height=max(500, len(numerical_cols) * 50),
            width=max(600, len(numerical_cols) * 50),
            plot_bgcolor='white'
        )
        
        return fig
    
    except Exception as e:
        print(f"Error creating correlation heatmap: {str(e)}")
        return go.Figure()


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