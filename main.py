"""Main entry point for the AI Data Insights Analyst application."""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import setup_logging
from src.database.db_init import init_db

# Setup logging first
logger = setup_logging()

# Initialize database
init_db()

# Now import and run Streamlit app
if __name__ == "__main__":
    logger.info("Starting AI Data Insights Analyst")
    
    # Import UI app after logging setup
    from ui.app import main as ui_main
    
    logger.info("Streamlit app initialized, launching UI...")
    
    # Run the Streamlit app
    ui_main()
