import os

from sacred import Ingredient

from definitions import ROOT_DIR, CURRENT_TIME
from mechanics.postprocessing.FilterTensorBoard import FilterTensorBoard

loggers_ingredient = Ingredient('loggers')


@loggers_ingredient.config
def conf():
    log_dir = 'results/logs'


@loggers_ingredient.capture
def loggers(log_dir, tag):
    train_log_dir = os.path.join(ROOT_DIR, '{}/{}-{}-train'.format(log_dir, tag, CURRENT_TIME))
    val_log_dir = os.path.join(ROOT_DIR, '{}/{}-{}-validation'.format(log_dir, tag, CURRENT_TIME))
    train_tboard_logger = FilterTensorBoard(log_dir=train_log_dir,
                                            write_graph=True,
                                            write_images=False, log_regex=r'^(?!val).*')
    val_tboard_logger = FilterTensorBoard(log_dir=val_log_dir, write_graph=False,
                                          write_images=False, log_regex=r"^val")
    return [train_tboard_logger, val_tboard_logger]
