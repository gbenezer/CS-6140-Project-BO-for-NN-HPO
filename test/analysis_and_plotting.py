import os
import pandas as pd
from pathlib import Path
from scipy.stats import bayes_mvs
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import sys
from ax.service.utils.instantiation import InstantiationBase

sys.path.insert(0, "C:/Users/Gil/Documents/Repositories/Python/CS_6140/Project")

# defining basal directories
experiment_directory = Path(os.getcwd()) / "logs" / "csv_logs" / "experiment_logs"
official_experiment_directory = experiment_directory / "official"
plot_directory = Path(os.getcwd()) / "plots" / "html_plots"

# getting benchmark data for both sets of data, as well as summary statistics
MNIST_benchmark = pd.read_csv(
    filepath_or_buffer=(experiment_directory / "classification_benchmark_results.csv")
)
MNIST_benchmark = MNIST_benchmark.assign(trial_index=-1, acquisition="Benchmark", replicate=1)
MNIST_accuracy_mean = MNIST_benchmark["test_accuracy"].mean()
MNIST_accuracy_std =  MNIST_benchmark["test_accuracy"].std()
MNIST_upper_error = MNIST_accuracy_mean + 2 * MNIST_accuracy_std
MNIST_lower_error = MNIST_accuracy_mean - 2 * MNIST_accuracy_std

Super_benchmark = pd.read_csv(
    filepath_or_buffer=(experiment_directory / "regression_benchmark_results.csv")
)
Super_benchmark = Super_benchmark.assign(trial_index=-1, acquisition="Benchmark", replicate=1)

# extracting the rest of the data
experimental_filename_list = os.listdir(path=official_experiment_directory)
MNIST_data = pd.DataFrame()
Super_data = pd.DataFrame()

for experimental_filename in experimental_filename_list:
    
    current_experiment_data = pd.read_csv(filepath_or_buffer=(official_experiment_directory / experimental_filename))
    
    acquisition_function, dataset, number_networks, replicate, _ = experimental_filename.split(sep="_")
    
    current_experiment_data = current_experiment_data.assign(acquisition=acquisition_function, replicate=replicate)
    
    if dataset == "MNIST":
        MNIST_data = pd.concat([MNIST_data, current_experiment_data]).copy()
        
    elif dataset == "Super":
        Super_data = pd.concat([Super_data, current_experiment_data]).copy()

print(MNIST_data)
fig = px.scatter(data_frame=MNIST_data, x="trial_index", y="test_accuracy", color="acquisition", symbol="acquisition")
# fig.add_hline(y=MNIST_benchmark_mean)
# fig.add_hline(y=MNIST_benchmark_limits[0], line_dash="dash")
# fig.add_hline(y=MNIST_benchmark_limits[1], line_dash="dash")
# fig.show()

# fig = px.scatter(data_frame=Super_data, x="trial_index", y="test_nrmse_range", color="acquisition", symbol="acquisition")
# fig.add_hline(y=Super_benchmark_mean)
# fig.add_hline(y=Super_benchmark_limits[0], line_dash="dash")
# fig.add_hline(y=Super_benchmark_limits[1], line_dash="dash")
# fig.show()