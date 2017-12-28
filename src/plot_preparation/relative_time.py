import os
from pathlib import Path

import pandas as pd

current_plot_dir = "tf-idf_batch_size"
target_plot_dir = "{}_processed".format(current_plot_dir)
ROOT_PLOT_DIR = os.path.abspath(
    os.path.join(Path.home(),
                 "Desktop/SemestrVII/INZYNIERKA/sentiment-analysis-thesis/praca-inzynierska/plot_data"))

os.makedirs(os.path.join(ROOT_PLOT_DIR, target_plot_dir), exist_ok=True)
plot_files = os.listdir(os.path.join(ROOT_PLOT_DIR, current_plot_dir))
for file in plot_files:
    csv = pd.read_csv(os.path.join(ROOT_PLOT_DIR, current_plot_dir, file))
    first_epoch_time = csv['Wall time'].iloc[0]
    wall_time_scaled = (csv['Wall time'] - first_epoch_time) / 3600
    csv['relative'] = wall_time_scaled
    csv.to_csv(os.path.join(ROOT_PLOT_DIR, target_plot_dir, file))
