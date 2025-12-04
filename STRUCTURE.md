# BI Dashboard - Project Structure

## Directory Organization

```
bi_dashboard/
├── app.py                     # Main application entry point
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── src/                       # Source code (modularized)
│   ├── core/                  # Core business logic
│   │   └── data_processor.py  # Data loading & type detection
│   │
│   ├── analytics/             # Analytics & insights
│   │   └── insights.py        # AI-powered insights generation
│   │
│   ├── visualization/         # Charts and plots
│   │   └── charts.py          # Visualization functions
│   │
│   ├── ui/                    # UI components (for future modularization)
│   │   └── tabs/              # Individual tab modules
│   │
│   └── utils/                 # Utility functions
│       └── file_utils.py      # File operations & formatting
│
├── data/                      # Data directory
│   └── samples/               # Sample datasets
│
└── tests/                     # Unit tests (for future)
```

## Module Descriptions

### `src/core/`
Core data processing functionality:
- `data_processor.py`: Dataset loading, type detection, filtering, statistics calculation

### `src/analytics/`
Analytics and insights: - `insights.py`: AI-powered insights generation

### `src/visualization/`
Visualization components:
- `charts.py`: Chart creation (matplotlib, plotly, seaborn)

### `src/utils/`
Utility functions:
- `file_utils.py`: File I/O, data formatting, CSV export

### `src/ui/tabs/`
Individual tab components (for future modularization):
- Ready for splitting app.py into modular tab files

## Import Pattern

All imports now follow the new modular structure:

```python
# In app.py
from src.core import data_processor as dp
from src.visualization import charts as viz
from src.analytics import insights
from src.utils import file_utils as utils
```

## Benefits of New Structure

✅ **Better Organization** - Code grouped by functionality  
✅ **Easier Maintenance** - Clear module boundaries  
✅ **Scalable** - Easy to add new modules  
✅ **Professional** - Industry-standard project layout  
✅ **Future-Ready** - Prepared for further modularization

## Next Steps for Further Modularization

The current `app.py` (1000+ lines) can be further split:
1. Extract each tab to `src/ui/tabs/[tab_name]_tab.py`
2. Move callback functions to `src/ui/callbacks.py`
3. Extract reusable components to `src/ui/components.py`

This structure is now ready for incremental refactoring as needed.
