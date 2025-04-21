import os
import pandas as pd
from pathlib import Path
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import sys
from ax.service.utils.instantiation import InstantiationBase
import matplotlib

pd.set_option("future.no_silent_downcasting", True)
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
ACTIVATION_FUNCTION_MAPPING = {
    "swish": 0.0,
    "sigmoid": 0.5,
    "relu": 1.0,
    "leaky_relu": 1.5,
}

# defining basal directories
experiment_directory = Path(os.getcwd()) / "logs" / "csv_logs" / "experiment_logs"
official_experiment_directory = experiment_directory / "official"
plot_directory = Path(os.getcwd()) / "plots" / "html_plots"
static_plot_directory = Path(os.getcwd()) / "plots" / "static_plots"

# getting benchmark data for both sets of data, as well as summary statistics
MNIST_benchmark = pd.read_csv(
    filepath_or_buffer=(experiment_directory / "classification_benchmark_results.csv")
)

MNIST_benchmark = MNIST_benchmark.assign(
    trial_index=-1, acquisition="Benchmark", replicate=1
)

MNIST_accuracy_mean = MNIST_benchmark["test_accuracy"].mean()
# print(MNIST_accuracy_mean)
MNIST_accuracy_std = MNIST_benchmark["test_accuracy"].std()
# print(MNIST_accuracy_std)
MNIST_upper_error = MNIST_accuracy_mean + 2 * MNIST_accuracy_std
MNIST_lower_error = MNIST_accuracy_mean - 2 * MNIST_accuracy_std

Super_benchmark = pd.read_csv(
    filepath_or_buffer=(experiment_directory / "regression_benchmark_results.csv")
)

Super_benchmark = Super_benchmark.assign(
    trial_index=-1, acquisition="Benchmark", replicate=1
)

Super_nrmse_range_mean = Super_benchmark["test_nrmse_range"].mean()
# print(Super_nrmse_range_mean)
Super_nrmse_range_std = Super_benchmark["test_nrmse_range"].std()
# print(Super_nrmse_range_std)
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
MNIST_data["activation"] = (
    MNIST_data["activation"]
    .replace(to_replace=ACTIVATION_FUNCTION_MAPPING)
    .astype(float)
)
# average accuracy for a random classifier in the case of 10 classes is 1 in 10
# so look at networks that did at least 1% better than random
MNIST_threshold_accuracy = MNIST_data[
    "test_accuracy"
].median()  # (max(MNIST_data["test_accuracy"]) - 0.1) / 2
MNIST_above_threshold = MNIST_data[
    MNIST_data["test_accuracy"] > MNIST_threshold_accuracy
]
# print(len(MNIST_data))
# print(MNIST_data["test_accuracy"].median())
# print(len(MNIST_above_threshold))
# print(len(MNIST_above_threshold[MNIST_above_threshold["activation"] == 0.5]))

Super_data["activation"] = (
    Super_data["activation"]
    .replace(to_replace=ACTIVATION_FUNCTION_MAPPING)
    .astype(float)
)
# getting typical performance of null regressor based on empirical threshold
Super_null_nrmse = Super_data["test_nrmse_range"][
    Super_data["test_nrmse_range"] >= 0.44
].median()
# print(Super_null_nrmse)
Super_threshold_nrmse = Super_data[
    "test_nrmse_range"
].median()  # (Super_null_nrmse - min(Super_data["test_nrmse_range"])) / 2
Super_below_threshold = Super_data[
    Super_data["test_nrmse_range"] < Super_threshold_nrmse
]
# print(len(Super_data))
# print(Super_data["test_nrmse_range"].median())
# print(
#     len(
#         Super_data[
#             (0.45 < Super_data["test_nrmse_range"])
#             & (Super_data["test_nrmse_range"] < 0.5)
#         ]
#     )
# )
# print(len(Super_below_threshold))

distribution_figure = px.violin(
    data_frame=MNIST_data, y="test_accuracy", points="all"
)
distribution_figure.show()
distribution_figure = px.violin(
    data_frame=Super_data, y="test_nrmse_range", points="all"
)
distribution_figure.show()
distribution_figure = px.violin(
    data_frame=MNIST_data, x="acquisition", y="test_accuracy", points="all"
)
distribution_figure.show()
distribution_figure = px.violin(
    data_frame=Super_data, x="acquisition", y="test_nrmse_range", points="all"
)
distribution_figure.show()

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
                    range=[0.0, 1.0],
                    label="Weight Decay",
                    values=MNIST_above_threshold["w_decay"],
                ),
            ]
        ),
    )
)
# fig.write_image(
#     file=(static_plot_directory / "MNIST_parcoords_above_threshold.pdf"),
#     format="pdf",
#     width=1600,
#     height=900,
# )
fig.show()

# Splitting and processing MNIST data by acquisition function
MNIST_random = MNIST_data[MNIST_data["acquisition"] == "Random"].copy()
MNIST_lognei = MNIST_data[MNIST_data["acquisition"] == "LogNEI"].copy()
MNIST_jes = MNIST_data[MNIST_data["acquisition"] == "JES"].copy()


print("Median number parameters, MNIST:", MNIST_data[MNIST_data["test_accuracy"] >= MNIST_accuracy_mean]["number_parameters"].median())
print("Median number parameters relative to hand-tuned, MNIST:", MNIST_data[MNIST_data["test_accuracy"] >= MNIST_accuracy_mean]["number_parameters"].median() / 447500.0)

print("Median number parameters, LogNEI, MNIST:", MNIST_lognei[MNIST_lognei["test_accuracy"] >= MNIST_accuracy_mean]["number_parameters"].median())
print("Median number parameters relative to hand-tuned, LogNEI, MNIST:", MNIST_lognei[MNIST_lognei["test_accuracy"] >= MNIST_accuracy_mean]["number_parameters"].median() / 447500.0)

print("Median checkpoint size, bytes, MNIST:", MNIST_data[MNIST_data["test_accuracy"] >= MNIST_accuracy_mean]["checkpoint_size"].median())
print("Median checkpoint size relative to hand-tuned, MNIST:", MNIST_data[MNIST_data["test_accuracy"] >= MNIST_accuracy_mean]["checkpoint_size"].median() / 5389811.0)

print("Median checkpoint size, bytes, LogNEI, MNIST:", MNIST_lognei[MNIST_lognei["test_accuracy"] >= MNIST_accuracy_mean]["checkpoint_size"].median())
print("Median checkpoint size relative to hand-tuned, LogNEI, MNIST:", MNIST_lognei[MNIST_lognei["test_accuracy"] >= MNIST_accuracy_mean]["checkpoint_size"].median() / 5389811.0)

# Calculating cumulative regret for each replicate
MNIST_random["immediate_regret"] = abs(
    MNIST_random["test_accuracy"] - MNIST_accuracy_mean
)
MNIST_random["cumulative_regret"] = (
    MNIST_random[["replicate", "immediate_regret"]]
    .groupby(["replicate"], as_index=False)
    .cumsum()
)
MNIST_lognei["immediate_regret"] = abs(
    MNIST_lognei["test_accuracy"] - MNIST_accuracy_mean
)
MNIST_lognei["cumulative_regret"] = (
    MNIST_lognei[["replicate", "immediate_regret"]]
    .groupby(["replicate"], as_index=False)
    .cumsum()
)
MNIST_jes["immediate_regret"] = abs(MNIST_jes["test_accuracy"] - MNIST_accuracy_mean)
MNIST_jes["cumulative_regret"] = (
    MNIST_jes[["replicate", "immediate_regret"]]
    .groupby(["replicate"], as_index=False)
    .cumsum()
)

# getting cumulative median accuracies and regrets
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
MNIST_random_cumulative_regret_median = (
    MNIST_random[["trial_index", "cumulative_regret"]]
    .groupby(["trial_index"], as_index=False)
    .median()
)
MNIST_lognei_cumulative_regret_median = (
    MNIST_lognei[["trial_index", "cumulative_regret"]]
    .groupby(["trial_index"], as_index=False)
    .median()
)
MNIST_jes_cumulative_regret_median = (
    MNIST_jes[["trial_index", "cumulative_regret"]]
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
    y=100 * MNIST_random["test_accuracy"],
    color=RANDOM_COLOR,
    alpha=MARKER_ALPHA,
)
ax.plot(
    MNIST_random_median_cummax["trial_index"],
    100 * MNIST_random_median_cummax["test_accuracy"],
    "-",
    color=RANDOM_COLOR,
    label="Random",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.scatter(
    x=MNIST_lognei["trial_index"],
    y=100 * MNIST_lognei["test_accuracy"],
    color=LOGNEI_COLOR,
    alpha=MARKER_ALPHA,
)
ax.plot(
    MNIST_lognei_median_cummax["trial_index"],
    100 * MNIST_lognei_median_cummax["test_accuracy"],
    "-",
    color=LOGNEI_COLOR,
    label="LogNEI",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.scatter(
    x=MNIST_jes["trial_index"],
    y=100 * MNIST_jes["test_accuracy"],
    color=JES_COLOR,
    alpha=MARKER_ALPHA,
)
ax.plot(
    MNIST_jes_median_cummax["trial_index"],
    100 * MNIST_jes_median_cummax["test_accuracy"],
    "-",
    color=JES_COLOR,
    label="JES",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.hlines(
    y=100 * MNIST_accuracy_mean,
    xmin=0,
    xmax=50,
    colors=BENCHMARK_COLOR,
    linestyles="solid",
    label="Benchmark",
)
ax.hlines(
    y=[100 * MNIST_lower_error, 100 * MNIST_upper_error],
    xmin=0,
    xmax=50,
    colors=BENCHMARK_COLOR,
    linestyles="dashed",
)
ax.set_ylim([0, 110])
ax.set_xlabel("Hyperparameter Evaluation")
ax.set_ylabel("Test Accuracy (%)")
plt.legend()
# plt.savefig(str((FIGURE_PATH / "MNIST_Test_Accuracy_Versus_Evaluation.pdf")))
plt.show()

# cumulative regret as a function of hyperparameter evaluation

fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
ax.scatter(
    x=MNIST_random["trial_index"],
    y=MNIST_random["cumulative_regret"],
    color=RANDOM_COLOR,
    alpha=MARKER_ALPHA,
)
ax.plot(
    MNIST_random_cumulative_regret_median["trial_index"],
    MNIST_random_cumulative_regret_median["cumulative_regret"],
    "-",
    color=RANDOM_COLOR,
    label="Random",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.scatter(
    x=MNIST_lognei["trial_index"],
    y=MNIST_lognei["cumulative_regret"],
    color=LOGNEI_COLOR,
    alpha=MARKER_ALPHA,
)
ax.plot(
    MNIST_lognei_cumulative_regret_median["trial_index"],
    MNIST_lognei_cumulative_regret_median["cumulative_regret"],
    "-",
    color=LOGNEI_COLOR,
    label="LogNEI",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.scatter(
    x=MNIST_jes["trial_index"],
    y=MNIST_jes["cumulative_regret"],
    color=JES_COLOR,
    alpha=MARKER_ALPHA,
)
ax.plot(
    MNIST_jes_cumulative_regret_median["trial_index"],
    MNIST_jes_cumulative_regret_median["cumulative_regret"],
    "-",
    color=JES_COLOR,
    label="JES",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.set_xlabel("Hyperparameter Evaluation")
ax.set_ylabel("Cumulative Regret")
plt.legend()
# plt.savefig(str((FIGURE_PATH / "MNIST_Cumulative_Regret_Versus_Evaluation.pdf")))
plt.show()

# creating interactive parallel coordinate plot for random search
fig = go.Figure(
    data=go.Parcoords(
        line=dict(
            color=MNIST_random["test_accuracy"],
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
                    values=MNIST_random["input_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Hidden Dropout Probability",
                    values=MNIST_random["hidden_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Output Dropout Probability",
                    values=MNIST_random["output_dropout_probability"],
                ),
                dict(
                    range=[0, 1000],
                    label="Hidden Layer 1 Nodes",
                    values=MNIST_random["hidden_layer_nodes_1"],
                ),
                dict(
                    range=[0, 1000],
                    label="Hidden Layer 2 Nodes",
                    values=MNIST_random["hidden_layer_nodes_2"],
                ),
                dict(
                    range=[0, 1000],
                    label="Hidden Layer 3 Nodes",
                    values=MNIST_random["hidden_layer_nodes_3"],
                ),
                dict(
                    range=[-1, 2],
                    tickvals=[0, 0.5, 1, 1.5],
                    ticktext=["swish", "sigmoid", "relu", "leaky_relu"],
                    label="Activation Function",
                    values=MNIST_random["activation"],
                ),
                dict(
                    range=[-9, 1],
                    label="Log(Learning Rate)",
                    values=MNIST_random["learning_rate"].apply(np.log10),
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Momentum Constant",
                    values=MNIST_random["beta1"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Adaptive Learning Rate Constant",
                    values=MNIST_random["beta2"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Weight Decay",
                    values=MNIST_random["w_decay"],
                ),
            ]
        ),
    )
)
# fig.write_image(file=(static_plot_directory / "MNIST_parcoords_Random.pdf"), format="pdf", width=1600, height=900)
# fig.show()

# creating interactive parallel coordinate plot for lognei MNIST
fig = go.Figure(
    data=go.Parcoords(
        line=dict(
            color=MNIST_lognei["test_accuracy"],
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
                    values=MNIST_lognei["input_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Hidden Dropout Probability",
                    values=MNIST_lognei["hidden_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Output Dropout Probability",
                    values=MNIST_lognei["output_dropout_probability"],
                ),
                dict(
                    range=[0, 1000],
                    label="Hidden Layer 1 Nodes",
                    values=MNIST_lognei["hidden_layer_nodes_1"],
                ),
                dict(
                    range=[0, 1000],
                    label="Hidden Layer 2 Nodes",
                    values=MNIST_lognei["hidden_layer_nodes_2"],
                ),
                dict(
                    range=[0, 1000],
                    label="Hidden Layer 3 Nodes",
                    values=MNIST_lognei["hidden_layer_nodes_3"],
                ),
                dict(
                    range=[-1, 2],
                    tickvals=[0, 0.5, 1, 1.5],
                    ticktext=["swish", "sigmoid", "relu", "leaky_relu"],
                    label="Activation Function",
                    values=MNIST_lognei["activation"],
                ),
                dict(
                    range=[-9, 1],
                    label="Log(Learning Rate)",
                    values=MNIST_lognei["learning_rate"].apply(np.log10),
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Momentum Constant",
                    values=MNIST_lognei["beta1"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Adaptive Learning Rate Constant",
                    values=MNIST_lognei["beta2"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Weight Decay",
                    values=MNIST_lognei["w_decay"],
                ),
            ]
        ),
    )
)
# fig.write_image(file=(static_plot_directory / "MNIST_parcoords_LogNEI.pdf"), format="pdf", width=1600, height=900)
# fig.show()

# creating interactive parallel coordinate plot for JES MNIST
fig = go.Figure(
    data=go.Parcoords(
        line=dict(
            color=MNIST_jes["test_accuracy"],
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
                    values=MNIST_jes["input_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Hidden Dropout Probability",
                    values=MNIST_jes["hidden_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Output Dropout Probability",
                    values=MNIST_jes["output_dropout_probability"],
                ),
                dict(
                    range=[0, 1000],
                    label="Hidden Layer 1 Nodes",
                    values=MNIST_jes["hidden_layer_nodes_1"],
                ),
                dict(
                    range=[0, 1000],
                    label="Hidden Layer 2 Nodes",
                    values=MNIST_jes["hidden_layer_nodes_2"],
                ),
                dict(
                    range=[0, 1000],
                    label="Hidden Layer 3 Nodes",
                    values=MNIST_jes["hidden_layer_nodes_3"],
                ),
                dict(
                    range=[-1, 2],
                    tickvals=[0, 0.5, 1, 1.5],
                    ticktext=["swish", "sigmoid", "relu", "leaky_relu"],
                    label="Activation Function",
                    values=MNIST_jes["activation"],
                ),
                dict(
                    range=[-9, 1],
                    label="Log(Learning Rate)",
                    values=MNIST_jes["learning_rate"].apply(np.log10),
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Momentum Constant",
                    values=MNIST_jes["beta1"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Adaptive Learning Rate Constant",
                    values=MNIST_jes["beta2"],
                ),
                dict(
                    range=[0.0, 1.0], label="Weight Decay", values=MNIST_jes["w_decay"]
                ),
            ]
        ),
    )
)
# fig.write_image(file=(static_plot_directory / "MNIST_parcoords_JES.pdf"), format="pdf", width=1600, height=900)
# fig.show()

# Splitting and processing Superconductivity data by acquisition function
Super_random = Super_data[Super_data["acquisition"] == "Random"].copy()
Super_lognei = Super_data[Super_data["acquisition"] == "LogNEI"].copy()
Super_jes = Super_data[Super_data["acquisition"] == "JES"].copy()

print("Median number parameters, Super:", Super_data[Super_data["test_nrmse_range"] >= Super_nrmse_range_mean]["number_parameters"].median())
print("Median number parameters relative to hand-tuned, Super:", Super_data[Super_data["test_nrmse_range"] >= Super_nrmse_range_mean]["number_parameters"].median() / 41580.0)

print("Median number parameters, LogNEI, Super:", Super_lognei[Super_lognei["test_nrmse_range"] >= Super_nrmse_range_mean]["number_parameters"].median())
print("Median number parameters relative to hand-tuned, LogNEI, Super:", Super_lognei[Super_lognei["test_nrmse_range"] >= Super_nrmse_range_mean]["number_parameters"].median() / 41580.0)

print("Median checkpoint size, bytes, Super:", Super_data[Super_data["test_nrmse_range"] >= Super_nrmse_range_mean]["checkpoint_size"].median())
print("Median checkpoint size relative to hand-tuned, Super:", Super_data[Super_data["test_nrmse_range"] >= Super_nrmse_range_mean]["checkpoint_size"].median() / 514995.0)

print("Median checkpoint size, bytes, LogNEI, Super:", Super_lognei[Super_lognei["test_nrmse_range"] >= Super_nrmse_range_mean]["checkpoint_size"].median())
print("Median checkpoint size relative to hand-tuned, LogNEI, Super:", Super_lognei[Super_lognei["test_nrmse_range"] >= Super_nrmse_range_mean]["checkpoint_size"].median() / 514995.0)

# calculating cumulative regret for every replicate
Super_random["immediate_regret"] = abs(
    Super_random["test_nrmse_range"] - Super_nrmse_range_mean
)
Super_random["cumulative_regret"] = (
    Super_random[["replicate", "immediate_regret"]]
    .groupby(["replicate"], as_index=False)
    .cumsum()
)
Super_lognei["immediate_regret"] = abs(
    Super_lognei["test_nrmse_range"] - Super_nrmse_range_mean
)
Super_lognei["cumulative_regret"] = (
    Super_lognei[["replicate", "immediate_regret"]]
    .groupby(["replicate"], as_index=False)
    .cumsum()
)
Super_jes["immediate_regret"] = abs(
    Super_jes["test_nrmse_range"] - Super_nrmse_range_mean
)
Super_jes["cumulative_regret"] = (
    Super_jes[["replicate", "immediate_regret"]]
    .groupby(["replicate"], as_index=False)
    .cumsum()
)

# getting cumulative minimum test nrmse (range normalized) and cumulative regrets
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
Super_random_cumulative_regret_median = (
    Super_random[["trial_index", "cumulative_regret"]]
    .groupby(["trial_index"], as_index=False)
    .median()
)
Super_lognei_cumulative_regret_median = (
    Super_lognei[["trial_index", "cumulative_regret"]]
    .groupby(["trial_index"], as_index=False)
    .median()
)
Super_jes_cumulative_regret_median = (
    Super_jes[["trial_index", "cumulative_regret"]]
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
                    range=[0.0, 1.0],
                    label="Weight Decay",
                    values=Super_below_threshold["w_decay"],
                ),
            ]
        ),
    )
)
# fig.write_image(
#     file=(static_plot_directory / "Super_parcoords_below_threshold.pdf"),
#     format="pdf",
#     width=1600,
#     height=900,
# )
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
# plt.savefig(str((FIGURE_PATH / "Super_NRMSE_Versus_Evaluation.pdf")))
plt.show()

# cumulative regret as a function of hyperparameter evaluation

fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
ax.scatter(
    x=Super_random["trial_index"],
    y=Super_random["cumulative_regret"],
    color=RANDOM_COLOR,
    alpha=MARKER_ALPHA,
)
ax.plot(
    Super_random_cumulative_regret_median["trial_index"],
    Super_random_cumulative_regret_median["cumulative_regret"],
    "-",
    color=RANDOM_COLOR,
    label="Random",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.scatter(
    x=Super_lognei["trial_index"],
    y=Super_lognei["cumulative_regret"],
    color=LOGNEI_COLOR,
    alpha=MARKER_ALPHA,
)
ax.plot(
    Super_lognei_cumulative_regret_median["trial_index"],
    Super_lognei_cumulative_regret_median["cumulative_regret"],
    "-",
    color=LOGNEI_COLOR,
    label="LogNEI",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.scatter(
    x=Super_jes["trial_index"],
    y=Super_jes["cumulative_regret"],
    color=JES_COLOR,
    alpha=MARKER_ALPHA,
)
ax.plot(
    Super_jes_cumulative_regret_median["trial_index"],
    Super_jes_cumulative_regret_median["cumulative_regret"],
    "-",
    color=JES_COLOR,
    label="JES",
    linewidth=LINE_WIDTH,
    alpha=LINE_ALPHA,
)
ax.set_xlabel("Hyperparameter Evaluation")
ax.set_ylabel("Cumulative Regret")
plt.legend()
# plt.savefig(str((FIGURE_PATH / "Super_Cumulative_Regret_Versus_Evaluation.pdf")))
plt.show()

# creating interactive parallel coordinate plots for random Super
fig = go.Figure(
    data=go.Parcoords(
        line=dict(
            color=Super_random["test_nrmse_range"],
            colorscale=list(reversed(px.colors.sequential.Plasma)),
            showscale=True,
            cmin=0,
            cmax=0.5,
        ),
        dimensions=list(
            [
                dict(
                    range=[0.0, 1.0],
                    label="Input Dropout Probability",
                    values=Super_random["input_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Hidden Dropout Probability",
                    values=Super_random["hidden_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Output Dropout Probability",
                    values=Super_random["output_dropout_probability"],
                ),
                dict(
                    range=[0, 300],
                    label="Hidden Layer 1 Nodes",
                    values=Super_random["hidden_layer_nodes_1"],
                ),
                dict(
                    range=[0, 300],
                    label="Hidden Layer 2 Nodes",
                    values=Super_random["hidden_layer_nodes_2"],
                ),
                dict(
                    range=[0, 300],
                    label="Hidden Layer 3 Nodes",
                    values=Super_random["hidden_layer_nodes_3"],
                ),
                dict(
                    range=[-1, 2],
                    tickvals=[0, 0.5, 1, 1.5],
                    ticktext=["swish", "sigmoid", "relu", "leaky_relu"],
                    label="Activation Function",
                    values=Super_random["activation"],
                ),
                dict(
                    range=[-9, 1],
                    label="Log(Learning Rate)",
                    values=Super_random["learning_rate"].apply(np.log10),
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Momentum Constant",
                    values=Super_random["beta1"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Adaptive Learning Rate Constant",
                    values=Super_random["beta2"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Weight Decay",
                    values=Super_random["w_decay"],
                ),
            ]
        ),
    )
)
# fig.write_image(file=(static_plot_directory / "Super_parcoords_Random.pdf"), format="pdf", width=1600, height=900)
# fig.show()

# creating interactive parallel coordinate plots for lognei Super
fig = go.Figure(
    data=go.Parcoords(
        line=dict(
            color=Super_lognei["test_nrmse_range"],
            colorscale=list(reversed(px.colors.sequential.Plasma)),
            showscale=True,
            cmin=0,
            cmax=0.5,
        ),
        dimensions=list(
            [
                dict(
                    range=[0.0, 1.0],
                    label="Input Dropout Probability",
                    values=Super_lognei["input_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Hidden Dropout Probability",
                    values=Super_lognei["hidden_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Output Dropout Probability",
                    values=Super_lognei["output_dropout_probability"],
                ),
                dict(
                    range=[0, 300],
                    label="Hidden Layer 1 Nodes",
                    values=Super_lognei["hidden_layer_nodes_1"],
                ),
                dict(
                    range=[0, 300],
                    label="Hidden Layer 2 Nodes",
                    values=Super_lognei["hidden_layer_nodes_2"],
                ),
                dict(
                    range=[0, 300],
                    label="Hidden Layer 3 Nodes",
                    values=Super_lognei["hidden_layer_nodes_3"],
                ),
                dict(
                    range=[-1, 2],
                    tickvals=[0, 0.5, 1, 1.5],
                    ticktext=["swish", "sigmoid", "relu", "leaky_relu"],
                    label="Activation Function",
                    values=Super_lognei["activation"],
                ),
                dict(
                    range=[-9, 1],
                    label="Log(Learning Rate)",
                    values=Super_lognei["learning_rate"].apply(np.log10),
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Momentum Constant",
                    values=Super_lognei["beta1"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Adaptive Learning Rate Constant",
                    values=Super_lognei["beta2"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Weight Decay",
                    values=Super_lognei["w_decay"],
                ),
            ]
        ),
    )
)
# fig.write_image(file=(static_plot_directory / "Super_parcoords_LogNEI.pdf"), format="pdf", width=1600, height=900)
# fig.show()

# creating interactive parallel coordinate plots for jes Super
fig = go.Figure(
    data=go.Parcoords(
        line=dict(
            color=Super_jes["test_nrmse_range"],
            colorscale=list(reversed(px.colors.sequential.Plasma)),
            showscale=True,
            cmin=0,
            cmax=0.5,
        ),
        dimensions=list(
            [
                dict(
                    range=[0.0, 1.0],
                    label="Input Dropout Probability",
                    values=Super_jes["input_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Hidden Dropout Probability",
                    values=Super_jes["hidden_dropout_probability"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Output Dropout Probability",
                    values=Super_jes["output_dropout_probability"],
                ),
                dict(
                    range=[0, 300],
                    label="Hidden Layer 1 Nodes",
                    values=Super_jes["hidden_layer_nodes_1"],
                ),
                dict(
                    range=[0, 300],
                    label="Hidden Layer 2 Nodes",
                    values=Super_jes["hidden_layer_nodes_2"],
                ),
                dict(
                    range=[0, 300],
                    label="Hidden Layer 3 Nodes",
                    values=Super_jes["hidden_layer_nodes_3"],
                ),
                dict(
                    range=[-1, 2],
                    tickvals=[0, 0.5, 1, 1.5],
                    ticktext=["swish", "sigmoid", "relu", "leaky_relu"],
                    label="Activation Function",
                    values=Super_jes["activation"],
                ),
                dict(
                    range=[-9, 1],
                    label="Log(Learning Rate)",
                    values=Super_jes["learning_rate"].apply(np.log10),
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Momentum Constant",
                    values=Super_jes["beta1"],
                ),
                dict(
                    range=[0.0, 1.0],
                    label="Adaptive Learning Rate Constant",
                    values=Super_jes["beta2"],
                ),
                dict(
                    range=[0.0, 1.0], label="Weight Decay", values=Super_jes["w_decay"]
                ),
            ]
        ),
    )
)
# fig.write_image(file=(static_plot_directory / "Super_parcoords_JES.pdf"), format="pdf", width=1600, height=900)
# fig.show()
