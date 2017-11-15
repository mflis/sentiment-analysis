import os
from datetime import datetime

from src.mechanics.postprocessing import plots, logger, custom_metrics
from src.mechanics.preprocessing import helpers

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
CURRENT_TIME = str(datetime.now())
