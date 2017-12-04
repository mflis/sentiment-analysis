import os

import arrow

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
CURRENT_TIME = arrow.now().format('MM-DD HH:mm')
