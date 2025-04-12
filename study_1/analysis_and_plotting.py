import os
import pandas as pd
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.io as pio
import sys
from ax.service.utils.instantiation import InstantiationBase
import matplotlib

matplotlib.rcParams.update(
    {
        "legend.fontsize": 16,
        "font.size": 24,
        "font.family": "serif",
    }
)
FIGURE_PATH = Path(
    "C:/Users/Gil/Documents/Repositories/Python/CS_6140/Project/plots/static_plots"
)
INTERACTIVE_PATH = Path(
    "C:/Users/Gil/Documents/Repositories/Python/CS_6140/Project/plots/html_plots"
)
FIG_SIZE = (12, 9)
DPI = 80
BENCHMARK_COLOR = "black"
RANDOM_COLOR = "#332288"
LOGNEI_COLOR = "#cc6677"
JES_COLOR = "#117733"
MARKER_ALPHA = 0.35
LINE_ALPHA = 0.75
LINE_WIDTH = 3
ACTIVATION_FUNCTION_MAPPING = {"swish": 0.0, "sigmoid": 0.5, "relu": 1.0, "leaky_relu": 1.5}

# defining basal directories
experiment_directory = Path(os.getcwd()) / "logs" / "csv_logs" / "experiment_logs"
official_experiment_directory = experiment_directory / "official"
plot_directory = Path(os.getcwd()) / "plots" / "html_plots"

# getting benchmark data for both sets of data, as well as summary statistics
MNIST_benchmark = pd.read_csv(
    filepath_or_buffer=(experiment_directory / "classification_benchmark_results.csv")
)

MNIST_benchmark = MNIST_benchmark.assign(
    trial_index=-1, acquisition="Benchmark", replicate=1
)

MNIST_accuracy_mean = MNIST_benchmark["test_accuracy"].mean()
MNIST_accuracy_std = MNIST_benchmark["test_accuracy"].std()
MNIST_upper_error = MNIST_accuracy_mean + 2 * MNIST_accuracy_std
MNIST_lower_error = MNIST_accuracy_mean - 2 * MNIST_accuracy_std

Super_benchmark = pd.read_csv(
    filepath_or_buffer=(experiment_directory / "regression_benchmark_results.csv")
)

Super_benchmark = Super_benchmark.assign(
    trial_index=-1, acquisition="Benchmark", replicate=1
)

Super_nrmse_range_mean = Super_benchmark["test_nrmse_range"].mean()
Super_nrmse_range_std = Super_benchmark["test_nrmse_range"].std()
Super_upper_error = Super_nrmse_range_mean + 2 * Super_nrmse_range_std
Super_lower_error = Super_nrmse_range_mean - 2 * Super_nrmse_range_std

# extracting the rest of the data
experimental_filename_list = os.listdir(path=official_experiment_directory)
MNIST_data = pd.DataFrame()
Super_data = pd.DataFrame()

for experimental_filename in experimental_filename_list:

    current_experiment_data = pd.read_csv(
        filepath_or_buffer=(official_experiment_directory / experimental_filename)
    )

    acquisition_function, dataset, number_networks, replicate, _ = (
        experimental_filename.split(sep="_")
    )
    if acquisition_function != "Random":
        acquisition_function = acquisition_function[1:]

    current_experiment_data = current_experiment_data.assign(
        acquisition=acquisition_function, replicate=replicate
    )

    if dataset == "MNIST":
        MNIST_data = pd.concat([MNIST_data, current_experiment_data]).copy()

    elif dataset == "Super":
        Super_data = pd.concat([Super_data, current_experiment_data]).copy()

# mapping activation functions to numeric values
# then extracting the data of runs that are better than the typical null model
MNIST_data["activation"] = MNIST_data["activation"].replace(
    to_replace=ACTIVATION_FUNCTION_MAPPING
)
# average accuracy for a random classifier in the case of 10 classes is 1 in 10
# so look at networks that did at least 1% better than random
MNIST_threshold_accuracy = 0.11 # (max(MNIST_data["test_accuracy"]) - 0.1) / 2
MNIST_above_threshold = MNIST_data[MNIST_data["test_accuracy"] > MNIST_threshold_accuracy]

Super_data["activation"] = Super_data["activation"].replace(
    to_replace=ACTIVATION_FUNCTION_MAPPING
)
# getting typical performance of null regressor based on empirical threshold
Super_null_nrmse = Super_data["test_nrmse_range"][Super_data["test_nrmse_range"] >= 0.44].mean()
Super_threshold_nrmse = 0.3 # (Super_null_nrmse - min(Super_data["test_nrmse_range"])) / 2
Super_below_threshold = Super_data[Super_data["test_nrmse_range"] <= Super_threshold_nrmse]

# creating interactive parallel coordinate plots
fig = go.Figure(
    data=go.Parcoords(
        line=dict(
            color=MNIST_above_threshold["test_accuracy"],
            colorscale="plasma",
            showscale=True,
            cmin=0,
            cmax=1.0,
        ),
        dimensions=list(
            [
                dict(
                    range=[0.0, 1.0],
                    label="Input Dropout Probability",
                    values=MNIST_above_threshold["input_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Hidden Dropout Probability",
                    values=MNIST_above_threshold["hidden_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Output Dropout Probability",
                    values=MNIST_above_threshold["output_dropout_probability"],
                ),
                dict(
                    range=[0, 1000],
                    label="Hidden Layer 1 Nodes",
                    values=MNIST_above_threshold["hidden_layer_nodes_1"],
                ),
                dict(
                    range=[0, 1000],
                    label="Hidden Layer 2 Nodes",
                    values=MNIST_above_threshold["hidden_layer_nodes_2"],
                ),
                dict(
                    range=[0, 1000],
                    label="Hidden Layer 3 Nodes",
                    values=MNIST_above_threshold["hidden_layer_nodes_3"],
                ),
                dict(
                    range=[-1, 2],
                    tickvals=[0, 0.5, 1, 1.5],
                    ticktext=["swish", "sigmoid", "relu", "leaky_relu"],
                    label="Activation Function",
                    values=MNIST_above_threshold["activation"],
                ),
                dict(
                    range=[-9, 1],
                    label="Log(Learning Rate)",
                    values=MNIST_above_threshold["learning_rate"].apply(np.log10),
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Momentum Constant",
                    values=MNIST_above_threshold["beta1"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Adaptive Learning Rate Constant",
                    values=MNIST_above_threshold["beta2"],
                ),
                dict(
                    range=[0.0, 1.0], label="Weight Decay", values=MNIST_above_threshold["w_decay"]
                ),
            ]
        ),
    )
)
fig.show()

# Splitting and processing MNIST data by acquisition function
MNIST_random = MNIST_data[MNIST_data["acquisition"] == "Random"]
MNIST_lognei = MNIST_data[MNIST_data["acquisition"] == "LogNEI"]
MNIST_jes = MNIST_data[MNIST_data["acquisition"] == "JES"]

# getting cumulative median accuracies
MNIST_random_median = (
    MNIST_random[["trial_index", "test_accuracy"]]
    .groupby(["trial_index"], as_index=False)
    .median()
)
MNIST_lognei_median = (
    MNIST_lognei[["trial_index", "test_accuracy"]]
    .groupby(["trial_index"], as_index=False)
    .median()
)
MNIST_jes_median = (
    MNIST_jes[["trial_index", "test_accuracy"]]
    .groupby(["trial_index"], as_index=False)
    .median()
)

MNIST_random_median_cummax = MNIST_random_median.cummax()
MNIST_lognei_median_cummax = MNIST_lognei_median.cummax()
MNIST_jes_median_cummax = MNIST_jes_median.cummax()


# getting cumulative regrets relative to hand-tuned network
MNIST_random_median_cummax["immediate_regret"] = abs(
    MNIST_random_median_cummax["test_accuracy"] - MNIST_accuracy_mean
)
MNIST_random_median_cummax["cumulative_regret"] = MNIST_random_median_cummax[
    "immediate_regret"
].cumsum()
MNIST_random_median_cummax["log_cumulative_regret"] = MNIST_random_median_cummax[
    "cumulative_regret"
].apply(lambda x: np.log10(abs(x)))


MNIST_lognei_median_cummax["immediate_regret"] = abs(
    MNIST_lognei_median_cummax["test_accuracy"] - MNIST_accuracy_mean
)
MNIST_lognei_median_cummax["cumulative_regret"] = MNIST_lognei_median_cummax[
    "immediate_regret"
].cumsum()
MNIST_lognei_median_cummax["log_cumulative_regret"] = MNIST_lognei_median_cummax[
    "cumulative_regret"
].apply(lambda x: np.log10(abs(x)))

MNIST_jes_median_cummax["immediate_regret"] = abs(
    MNIST_jes_median_cummax["test_accuracy"] - MNIST_accuracy_mean
)
MNIST_jes_median_cummax["cumulative_regret"] = MNIST_jes_median_cummax[
    "immediate_regret"
].cumsum()
MNIST_jes_median_cummax["log_cumulative_regret"] = MNIST_jes_median_cummax[
    "cumulative_regret"
].apply(lambda x: np.log10(abs(x)))

# test accuracy plot as a function of hyperparameter evaluation
fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
ax.scatter(
    x=MNIST_random["trial_index"],
    y=MNIST_random["test_accuracy"],
    color=RANDOM_COLOR,
    alpha=MARKER_ALPHA,
)
ax.plot(
    MNIST_random_median_cummax["trial_index"],
    MNIST_random_median_cummax["test_accuracy"],
    "-",
    color=RANDOM_COLOR,
    label="Random",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.scatter(
    x=MNIST_lognei["trial_index"],
    y=MNIST_lognei["test_accuracy"],
    color=LOGNEI_COLOR,
    alpha=MARKER_ALPHA,
)
ax.plot(
    MNIST_lognei_median_cummax["trial_index"],
    MNIST_lognei_median_cummax["test_accuracy"],
    "-",
    color=LOGNEI_COLOR,
    label="LogNEI",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.scatter(
    x=MNIST_jes["trial_index"],
    y=MNIST_jes["test_accuracy"],
    color=JES_COLOR,
    alpha=MARKER_ALPHA,
)
ax.plot(
    MNIST_jes_median_cummax["trial_index"],
    MNIST_jes_median_cummax["test_accuracy"],
    "-",
    color=JES_COLOR,
    label="JES",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.hlines(
    y=MNIST_accuracy_mean,
    xmin=0,
    xmax=50,
    colors=BENCHMARK_COLOR,
    linestyles="solid",
    label="Benchmark",
)
ax.hlines(
    y=[MNIST_lower_error, MNIST_upper_error],
    xmin=0,
    xmax=50,
    colors=BENCHMARK_COLOR,
    linestyles="dashed",
)
ax.set_ylim([0, 1.1])
ax.set_xlabel("Hyperparameter Evaluation")
ax.set_ylabel("Test Accuracy (Fraction)")
plt.legend()
# plt.savefig(str((FIGURE_PATH / "MNIST_Test_Accuracy_Versus_Evaluation.png")))
plt.show()

# cumulative regret as a function of hyperparameter evaluation

fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
ax.plot(
    MNIST_random_median_cummax["trial_index"],
    MNIST_random_median_cummax["cumulative_regret"],
    "-",
    color=RANDOM_COLOR,
    label="Random",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.plot(
    MNIST_lognei_median_cummax["trial_index"],
    MNIST_lognei_median_cummax["cumulative_regret"],
    "-",
    color=LOGNEI_COLOR,
    label="LogNEI",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.plot(
    MNIST_jes_median_cummax["trial_index"],
    MNIST_jes_median_cummax["cumulative_regret"],
    "-",
    color=JES_COLOR,
    label="JES",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.set_xlabel("Hyperparameter Evaluation")
ax.set_ylabel("Cumulative Regret")
plt.legend()
# plt.savefig(str((FIGURE_PATH / "MNIST_Cumulative_Regret_Versus_Evaluation.png")))
plt.show()

# Splitting and processing Superconductivity data by acquisition function
Super_random = Super_data[Super_data["acquisition"] == "Random"]
Super_lognei = Super_data[Super_data["acquisition"] == "LogNEI"]
Super_jes = Super_data[Super_data["acquisition"] == "JES"]

# getting cumulative minimum test nrmse (range normalized)
Super_random_median = (
    Super_random[["trial_index", "test_nrmse_range"]]
    .groupby(["trial_index"], as_index=False)
    .median()
)
Super_lognei_median = (
    Super_lognei[["trial_index", "test_nrmse_range"]]
    .groupby(["trial_index"], as_index=False)
    .median()
)
Super_jes_median = (
    Super_jes[["trial_index", "test_nrmse_range"]]
    .groupby(["trial_index"], as_index=False)
    .median()
)

Super_random_median["cumulative"] = Super_random_median["test_nrmse_range"].cummin()
Super_lognei_median["cumulative"] = Super_lognei_median["test_nrmse_range"].cummin()
Super_jes_median["cumulative"] = Super_jes_median["test_nrmse_range"].cummin()

# getting cumulative regrets relative to hand-tuned network
Super_random_median["immediate_regret"] = abs(
    Super_random_median["cumulative"] - Super_nrmse_range_mean
)
Super_random_median["cumulative_regret"] = Super_random_median[
    "immediate_regret"
].cumsum()
Super_random_median["log_cumulative_regret"] = Super_random_median[
    "cumulative_regret"
].apply(lambda x: np.log10(abs(x)))

Super_lognei_median["immediate_regret"] = abs(
    Super_lognei_median["cumulative"] - Super_nrmse_range_mean
)
Super_lognei_median["cumulative_regret"] = Super_lognei_median[
    "immediate_regret"
].cumsum()
Super_lognei_median["log_cumulative_regret"] = Super_lognei_median[
    "cumulative_regret"
].apply(lambda x: np.log10(abs(x)))

Super_jes_median["immediate_regret"] = abs(
    Super_jes_median["cumulative"] - Super_nrmse_range_mean
)
Super_jes_median["cumulative_regret"] = Super_jes_median["immediate_regret"].cumsum()
Super_jes_median["log_cumulative_regret"] = Super_jes_median["cumulative_regret"].apply(
    lambda x: np.log10(abs(x))
)

# creating interactive parallel coordinate plots
fig = go.Figure(
    data=go.Parcoords(
        line=dict(
            color=Super_below_threshold["test_nrmse_range"],
            colorscale=list(reversed(px.colors.sequential.Plasma)),
            showscale=True,
            cmin=min(Super_below_threshold["test_nrmse_range"]),
            cmax=max(Super_below_threshold["test_nrmse_range"]),
        ),
        dimensions=list(
            [
                dict(
                    range=[0.0, 1.0],
                    label="Input Dropout Probability",
                    values=Super_below_threshold["input_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Hidden Dropout Probability",
                    values=Super_below_threshold["hidden_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Output Dropout Probability",
                    values=Super_below_threshold["output_dropout_probability"],
                ),
                dict(
                    range=[0, 300],
                    label="Hidden Layer 1 Nodes",
                    values=Super_below_threshold["hidden_layer_nodes_1"],
                ),
                dict(
                    range=[0, 300],
                    label="Hidden Layer 2 Nodes",
                    values=Super_below_threshold["hidden_layer_nodes_2"],
                ),
                dict(
                    range=[0, 300],
                    label="Hidden Layer 3 Nodes",
                    values=Super_below_threshold["hidden_layer_nodes_3"],
                ),
                dict(
                    range=[-1, 2],
                    tickvals=[0, 0.5, 1, 1.5],
                    ticktext=["swish", "sigmoid", "relu", "leaky_relu"],
                    label="Activation Function",
                    values=Super_below_threshold["activation"],
                ),
                dict(
                    range=[-9, 1],
                    label="Log(Learning Rate)",
                    values=Super_below_threshold["learning_rate"].apply(np.log10),
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Momentum Constant",
                    values=Super_below_threshold["beta1"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Adaptive Learning Rate Constant",
                    values=Super_below_threshold["beta2"],
                ),
                dict(
                    range=[0.0, 1.0], label="Weight Decay", values=Super_below_threshold["w_decay"]
                ),
            ]
        ),
    )
)
fig.show()


# test NRMSE as a function of number of hyperparameter evaluations
fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
ax.scatter(
    x=Super_random["trial_index"],
    y=Super_random["test_nrmse_range"],
    color=RANDOM_COLOR,
    alpha=MARKER_ALPHA,
)
ax.plot(
    Super_random_median["trial_index"],
    Super_random_median["cumulative"],
    "-",
    color=RANDOM_COLOR,
    label="Random",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.scatter(
    x=Super_lognei["trial_index"],
    y=Super_lognei["test_nrmse_range"],
    color=LOGNEI_COLOR,
    alpha=MARKER_ALPHA,
)
ax.plot(
    Super_lognei_median["trial_index"],
    Super_lognei_median["cumulative"],
    "-",
    color=LOGNEI_COLOR,
    label="LogNEI",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.scatter(
    x=Super_jes["trial_index"],
    y=Super_jes["test_nrmse_range"],
    color=JES_COLOR,
    alpha=MARKER_ALPHA,
)
ax.plot(
    Super_jes_median["trial_index"],
    Super_jes_median["cumulative"],
    "-",
    color=JES_COLOR,
    label="JES",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.hlines(
    y=Super_nrmse_range_mean,
    xmin=0,
    xmax=50,
    colors=BENCHMARK_COLOR,
    linestyles="solid",
    label="Benchmark",
)
ax.hlines(
    y=[Super_upper_error, Super_lower_error],
    xmin=0,
    xmax=50,
    colors=BENCHMARK_COLOR,
    linestyles="dashed",
)
ax.set_ylim([0, 0.5])
ax.set_xlabel("Hyperparameter Evaluation")
ax.set_ylabel("Test NRMSE (Range Normalized)")
plt.legend()
# plt.savefig(str((FIGURE_PATH / "Super_NRMSE_Versus_Evaluation.png")))
plt.show()

# cumulative regret as a function of hyperparameter evaluation

fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
ax.plot(
    Super_random_median["trial_index"],
    Super_random_median["cumulative_regret"],
    "-",
    color=RANDOM_COLOR,
    label="Random",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.plot(
    Super_lognei_median["trial_index"],
    Super_lognei_median["cumulative_regret"],
    "-",
    color=LOGNEI_COLOR,
    label="LogNEI",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.plot(
    Super_jes_median["trial_index"],
    Super_jes_median["cumulative_regret"],
    "-",
    color=JES_COLOR,
    label="JES",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.set_xlabel("Hyperparameter Evaluation")
ax.set_ylabel("Cumulative Regret")
plt.legend()
# plt.savefig(str((FIGURE_PATH / "Super_Cumulative_Regret_Versus_Evaluation.png")))
plt.show()
