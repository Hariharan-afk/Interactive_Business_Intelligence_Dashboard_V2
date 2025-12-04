# Business Intelligence Dashboard

A comprehensive, interactive web-based dashboard for business data analysis and visualization. Built with Gradio, pandas, matplotlib, seaborn, and plotly.

## 🎯 Features

### Data Upload & Management
- **Multi-format Support**: Upload CSV and Excel files (.xlsx, .xls)
- **Automatic Data Validation**: Checks for common data quality issues
- **Column Type Detection**: Automatically identifies numeric, categorical, and datetime columns
- **Manual Type Override**: Customize column types through an intuitive interface
- **Data Preview**: View first and last rows of your dataset

### Statistical Analysis
- **Numerical Statistics**: Mean, median, std dev, quartiles, min/max for all numerical columns
- **Categorical Statistics**: Unique values, mode, top categories with counts
- **Missing Value Analysis**: Comprehensive report on missing data
- **Correlation Analysis**: Interactive correlation heatmap for numerical features

### Interactive Filtering
- **Dynamic Filters**: Filter by numerical ranges, categorical values, and date ranges
- **Real-time Updates**: See filtered row counts instantly
- **Data Export**: Export filtered data as CSV

### Visualizations

#### Overview Charts (Auto-generated)
- Missing value analysis chart
- Numerical distributions (histograms)
- Categorical distributions (bar charts)
- Correlation heatmap

#### Custom Charts (User-defined)
- **Time Series**: Trend analysis over time with multiple aggregation methods
- **Distribution**: Histograms and box plots
- **Category Analysis**: Bar charts and pie charts
- **Scatter Plots**: Relationship visualization with optional color encoding
- **Grouped Charts**: Compare metrics across categories

### Automated Insights
- **Data Quality Scores**: Overall health assessment
- **Top/Bottom Performers**: Identify highest and lowest values
- **Outlier Detection**: Statistical anomaly identification
- **Trend Analysis**: Growth rates and pattern detection
- **Correlation Findings**: Strong relationships between variables

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download the project**
   ```bash
   cd bi_dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
python app.py
```

The dashboard will launch and be accessible at:
- Local: `http://localhost:7860`
- Network: `http://0.0.0.0:7860`

## 📊 Sample Datasets

Two sample datasets are provided in the `data/` folder:

### 1. Sales Data (`sales_data.csv`)
- **Size**: 1,500 rows × 7 columns
- **Date Range**: Last 2 years of sales transactions
- **Columns**:
  - `date`: Transaction date
  - `product_category`: Product category (8 categories)
  - `region`: Sales region (5 regions)
  - `sales_amount`: Revenue in dollars
  - `quantity`: Units sold
  - `customer_id`: Customer identifier
  - `discount_applied`: Discount percentage (0-30%)
- **Features**: Contains missing values, outliers, and seasonal trends

### 2. Customer Analytics (`customer_data.csv`)
- **Size**: 1,200 rows × 9 columns
- **Date Range**: 3 years of customer data
- **Columns**:
  - `customer_id`: Unique customer identifier
  - `signup_date`: Account creation date
  - `age`: Customer age
  - `country`: Customer location (15 countries)
  - `subscription_type`: Free, Basic, or Premium
  - `monthly_spend`: Average monthly spending
  - `lifetime_value`: Total customer value
  - `churn`: Churn status (Yes/No)
  - `last_active_date`: Most recent activity
- **Features**: Realistic distributions with missing values and business metrics

## 📖 Usage Guide

### Step 1: Upload Data
1. Navigate to the **"📁 Data Upload"** tab
2. Click "Upload File" and select your CSV or Excel file
3. Click "Load Dataset" to process the file
4. Review the dataset summary and preview

### Step 2: Manage Column Types (Optional)
1. In the "Column Type Management" accordion
2. Click "Show Column Types" to view detected types
3. Select a column and choose a new type if needed
4. Click "Update Type" to apply changes

### Step 3: Explore Statistics
1. Navigate to the **"📈 Statistics"** tab
2. Click "Calculate Statistics"
3. Review:
   - Numerical statistics table
   - Categorical value distributions
   - Missing values report
   - Correlation heatmap

### Step 4: Filter Data
1. Navigate to the **"🔍 Filter & Explore"** tab
2. Click "Setup Filters" to see available filter options
3. Configure filters for numerical, categorical, or date columns
4. Click "Apply Filters" to see results
5. Export filtered data using the "Export Filtered Data" button

### Step 5: Generate Visualizations

#### Overview Charts
1. Navigate to **"📊 Visualizations - Overview"** tab
2. Click "Generate Overview Charts"
3. View automatically generated charts:
   - Missing values chart
   - Numerical distributions
   - Categorical distributions
   - Correlation heatmap

#### Custom Charts
1. Navigate to **"🎨 Visualizations - Custom"** tab
2. Select a chart type (Time Series, Bar Chart, etc.)
3. Choose columns for X-axis and Y-axis
4. Optionally select a color-by column
5. Choose aggregation method if applicable
6. Click "Create Chart"

### Step 6: Generate Insights
1. Navigate to the **"💡 Insights"** tab
2. Click "Generate Insights"
3. Review:
   - Key insights summary
   - Detailed statistical analysis
   - Top/bottom performers
   - Data quality metrics

## 🏗️ Project Structure

```
bi_dashboard/
│
├── app.py                  # Main Gradio application
├── data_processor.py       # Data loading, validation, filtering
├── visualizations.py       # Chart creation functions
├── insights.py            # Automated insight generation
├── utils.py               # Helper functions (exports, formatting)
├── config.py              # Configuration constants
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── data/                 # Sample datasets
    ├── sales_data.csv
    └── customer_data.csv
```

## 🔧 Configuration

Edit `config.py` to customize:
- Maximum file size
- Number of preview rows
- Chart colors and themes
- Statistical thresholds
- Export settings

## 💡 Tips & Best Practices

### Data Preparation
- Ensure column names are descriptive
- Remove or handle special characters in column names
- Verify date formats are consistent

### Performance
- For large datasets (>100,000 rows), consider sampling
- Close unused visualizations to free memory
- Export filtered data before creating complex charts

### Visualization Selection
- **Time Series**: Use for data with date/time columns
- **Distribution**: Understand spread of numerical values
- **Bar/Pie Charts**: Compare categories
- **Scatter Plots**: Find relationships between two variables

### Insights Interpretation
- Pay attention to data quality scores
- Investigate outliers that might be errors
- Consider business context when reviewing correlations

## 🐛 Troubleshooting

### Common Issues

**Issue**: File upload fails
- **Solution**: Check file format (CSV or Excel), ensure file isn't corrupted, verify file size < 100MB

**Issue**: Charts not displaying
- **Solution**: Ensure data has been loaded, check that selected columns exist, verify column types are correct

**Issue**: Missing values causing errors
- **Solution**: Review missing value report, consider filtering out rows with missing critical values

**Issue**: Slow performance
- **Solution**: Reduce dataset size, close unused visualizations, limit number of categories in charts

## 📝 Notes

- The dashboard stores data in memory during the session
- Uploaded files are temporarily stored and can be deleted after use
- Export files are saved to the `/tmp` directory
- Filters are applied in real-time without modifying the original dataset

## 🤝 Contributing

To extend the dashboard:
1. Add new visualization types in `visualizations.py`
2. Implement additional insights in `insights.py`
3. Create custom filters in `data_processor.py`
4. Modify UI in `app.py`

## 📄 License

This project is created for educational purposes as part of a Business Intelligence Dashboard assignment.

## 🙏 Acknowledgments

Built using:
- **Gradio**: For the web interface
- **pandas**: For data manipulation
- **matplotlib & seaborn**: For static visualizations
- **plotly**: For interactive charts
- **numpy & scipy**: For statistical computations

---

**Version**: 1.0.0  
**Last Updated**: December 2025  
**Author**: Graduate Student - AI/ML Program