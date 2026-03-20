import logging
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Constants for log configuration
LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log"
MAX_LOG_SIZE = 5*1024*1024 # 5MB
BACKUP_COUNT = 3

# Construct log file path
root_path = Path(__file__).parent.parent.parent
log_dir_path = root_path / 'logs'
log_dir_path.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir_path / LOG_FILE

def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = RotatingFileHandler(filename=log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)

        formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

        handler.setFormatter(formatter)

        logger.addHandler(handler)
    
    return logger

