import os
from datetime import datetime

from src import custom_metrics
from src import helpers
from src import logger
from src import plots

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
CURRENT_TIME = str(datetime.now())
