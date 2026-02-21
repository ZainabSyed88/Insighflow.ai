# AI Data Insights Analyst

A sophisticated multi-agent system that automates data cleaning, verification, analysis, and visualization using LangGraph and LLMs.

## Overview

This project uses a graph-based multi-agent architecture to analyze CSV/Excel files through a pipeline of specialized agents:

1. **Data Cleaning Agent** - Removes outliers, handles missing values, validates data types
2. **Verification Agent** - Validates cleaning quality and data integrity
3. **Insights Agent** - Detects patterns, correlations, and anomalies
4. **Visualization Agent** - Generates interactive charts and visualizations

## Technology Stack

- **Orchestration**: LangGraph (graph-based multi-agent workflow)
- **LLM**: GPT OSS 120B via NVIDIA API (langchain-nvidia-ai-endpoints)
- **UI**: Streamlit
- **Database**: SQLite (with LangGraph checkpoint persistence)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **File Handling**: openpyxl (Excel), built-in CSV

## Project Structure

```
src/
├── agents/
│   ├── base_agent.py          # Abstract Agent class
│   ├── cleaning_agent.py      # Data cleaning
│   ├── verification_agent.py  # Quality verification
│   ├── insights_agent.py      # Pattern analysis
│   └── visualization_agent.py # Chart generation
├── graph/
│   ├── state.py               # LangGraph state schema
│   └── workflow.py            # Graph definition
├── database/
│   ├── models.py             # SQLAlchemy models
│   └── db_init.py            # Schema initialization
├── utils/
│   ├── logger.py             # Logging setup
│   ├── validators.py         # Data validation
│   └── file_handlers.py      # File I/O
└── config.py                 # Configuration

ui/
├── app.py                    # Main Streamlit app
├── pages/                    # Page components
└── components.py             # Reusable UI components
```

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA API Key (for LLM access)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-Data-Insights-Analyst
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your NVIDIA API key and settings
```

5. Initialize database:
```bash
python -m src.database.db_init
```

## Usage

### Running the Application

```bash
streamlit run main.py
```

The application will launch at `http://localhost:8501`

### Workflow

1. **Upload**: Select a CSV or Excel file to analyze
2. **Cleaning**: Review cleaning parameters and approve cleaned data
3. **Verification**: System verifies data quality (up to 3 attempts)
4. **Analysis**: Automatic pattern detection and insights generation
5. **Visualization**: Interactive charts and visualizations

## Configuration

All settings can be configured via environment variables in `.env`:

- `NVIDIA_API_KEY` - API key for NVIDIA LLM access
- `LLM_MODEL` - Model identifier (default: meta/llama-2-70b-chat)
- `MAX_VERIFICATION_ATTEMPTS` - Max retry attempts for verification (default: 3)
- `OUTLIER_IQR_MULTIPLIER` - IQR multiplier for outlier detection (default: 1.5)
- `MISSING_VALUE_THRESHOLD` - Threshold for removing columns with missing values (default: 0.5)

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Agents

1. Create agent class in `src/agents/` inheriting from `Agent`
2. Implement `execute()` method
3. Add node to workflow graph in `src/graph/workflow.py`
4. Add conditional edges as needed

## Features

- **Automated Data Pipeline**: Multi-stage processing with checkpoints
- **User Approval Gates**: Hybrid workflow with manual approval steps
- **Retry Logic**: Automatic retry with feedback for failed verification
- **State Persistence**: Resume workflows at any stage
- **Verbose Logging**: Complete audit trail of all operations
- **Interactive Visualizations**: Plotly-based charts and dashboards

## Limitations

- Max file size: 50MB (configurable)
- Max verification attempts: 3 (configurable)
- Designed for CSV/Excel files only

## Future Enhancements

- [ ] Support for JSON and database sources
- [ ] Advanced imputation strategies (KNN, MICE)
- [ ] Custom agent definitions
- [ ] Real-time collaboration features
- [ ] Export reports to PDF/HTML
- [ ] Integration with BI tools

## License

[Your License Here]

## Support

For issues or questions, please create an issue on GitHub.
