import os
from dotenv import load_dotenv
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deepanalytica.log')
    ]
)
logger = logging.getLogger('deepanalytica')

# Load environment variables
load_dotenv()

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# App configuration
APP_NAME = "DeepInsight"
APP_VERSION = "1.0.0"

# Cache configuration
CACHE_TTL = 3600  # 1 hour

# File upload configuration
ALLOWED_EXTENSIONS = {
    'csv', 'xlsx', 'xls', 'parquet', 'json', 
    'txt', 'tsv', 'pkl', 'h5', 'hdf5'
}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

# UI Configuration
THEME = {
    "primaryColor": "#4F46E5",
    "backgroundColor": "#111827",
    "secondaryBackgroundColor": "#1F2937",
    "textColor": "#F9FAFB",
    "font": "sans-serif"
}

def validate_config():
    """Validate configuration settings"""
    warnings = []
    errors = []
    
    # Check API key configuration
    if not DEEPSEEK_API_KEY:
        warnings.append("DeepSeek API key not configured. AI features will require manual configuration.")
    
    # Log all warnings and errors
    for warning in warnings:
        logger.warning(warning)
    for error in errors:
        logger.error(error)
        
    return len(errors) == 0

# Validate configuration on import
config_valid = validate_config()

# Log configuration status
if config_valid:
    logger.info("Configuration validation successful")
else:
    logger.warning("Configuration validation failed. Some features may be unavailable.")