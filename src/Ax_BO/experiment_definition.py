import logging
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.instantiation import InstantiationBase

# # hopefully gets rid of printed output
# logging.getLogger("ax.service").setLevel(logging.WARNING)
# logging.getLogger("ax.modelbridge").setLevel(logging.WARNING)
# logging.getLogger("ax.modelbridge.dispatch_utils").setLevel(logging.WARNING)
# logging.getLogger("ax.service.utils.instantiation").setLevel(logging.WARNING)
# logging.getLogger("ax.service.ax_client").setLevel(logging.WARNING)

# Status quo parameter settings that get really good results
# (benchmark configurations)
MNIST_status_quo_parameters = {
    "input_dropout_probability": 0.1,
    "hidden_dropout_probability": 0.5,
    "output_dropout_probability": 0.1,
    "hidden_layer_nodes_1": 500,
    "hidden_layer_nodes_2": 100,
    "hidden_layer_nodes_3": 50,
    "activation": "relu",
    "learning_rate": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "w_decay": 0,
}
Superconductivity_status_quo_parameters = {
    "input_dropout_probability": 0.1,
    "hidden_dropout_probability": 0.5,
    "output_dropout_probability": 0.1,
    "hidden_layer_nodes_1": 243,
    "hidden_layer_nodes_2": 81,
    "hidden_layer_nodes_3": 27,
    "activation": "leaky_relu",
    "learning_rate": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "w_decay": 0,
}

# Experimental parameter bounds
MNIST_parameters = [
    {
        "name": "input_dropout_probability",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "hidden_dropout_probability",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "output_dropout_probability",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "hidden_layer_nodes_1",
        "type": "range",
        "bounds": [20, 1000],
        "value_type": "int",
        "log_scale": False,
    },
    {
        "name": "hidden_layer_nodes_2",
        "type": "range",
        "bounds": [20, 1000],
        "value_type": "int",
        "log_scale": False,
    },
    {
        "name": "hidden_layer_nodes_3",
        "type": "range",
        "bounds": [20, 1000],
        "value_type": "int",
        "log_scale": False,
    },
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [1e-8, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "beta1",
        "type": "range",
        "bounds": [0.1, (1 - (1e-8))],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "beta2",
        "type": "range",
        "bounds": [0.1, (1 - (1e-8))],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "w_decay",
        "type": "range",
        "bounds": [0.0, (1 - (1e-8))],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "activation",
        "type": "choice",
        "values": ["swish", "sigmoid", "relu", "leaky_relu"],
        "value_type": "str",
        "is_ordered": False,
        "sort_values": False,
    },
]

Superconductivity_parameters = [
    {
        "name": "input_dropout_probability",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "hidden_dropout_probability",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "output_dropout_probability",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "hidden_layer_nodes_1",
        "type": "range",
        "bounds": [3, 300],
        "value_type": "int",
        "log_scale": False,
    },
    {
        "name": "hidden_layer_nodes_2",
        "type": "range",
        "bounds": [3, 300],
        "value_type": "int",
        "log_scale": False,
    },
    {
        "name": "hidden_layer_nodes_3",
        "type": "range",
        "bounds": [3, 300],
        "value_type": "int",
        "log_scale": False,
    },
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [1e-8, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "beta1",
        "type": "range",
        "bounds": [0.1, (1 - (1e-8))],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "beta2",
        "type": "range",
        "bounds": [0.1, (1 - (1e-8))],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "w_decay",
        "type": "range",
        "bounds": [0.0, (1 - (1e-8))],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "activation",
        "type": "choice",
        "values": ["swish", "sigmoid", "relu", "leaky_relu"],
        "value_type": "str",
        "is_ordered": False,
        "sort_values": False,
    },
]

# definition and bounds for budget variables just in case
# budget_variables = [
#     {
#         "name": "max_epochs",
#         "type": "range",
#         "bounds": [1, 100],
#         "value_type": "int",
#         "log_scale": False,
#     },
#     {
#         "name": "fraction_train_data",
#         "type": "range",
#         "bounds": [0.1, 1.0],
#         "value_type": "float",
#         "log_scale": False,
#     },
# ]

# Single objective and multiobjective definitions
MNIST_single_objective = {"test_accuracy": ObjectiveProperties(minimize=False)}

Superconductivity_single_objective = {
    "test_nrmse_range": ObjectiveProperties(minimize=True)
}

# NOTE: removed thresholds after unsuccessful MNIST hypervolume experiment
# thresholds were too high
MNIST_multiobjective = {
    "test_accuracy": ObjectiveProperties(minimize=False, threshold=0.75),
    "number_parameters": ObjectiveProperties(minimize=True),
}

Superconductivity_multiobjective = {
    "test_nrmse_range": ObjectiveProperties(minimize=True, threshold=0.3),
    "number_parameters": ObjectiveProperties(minimize=True),
}

# Parameter constraints (relative to each other; bounds are defined in parameters)
p_constraints = [
    "hidden_layer_nodes_2 <= hidden_layer_nodes_1",
    "hidden_layer_nodes_3 <= hidden_layer_nodes_2",
]

# Outcome constraints
o_constraints = None

# Metrics that are not necessarily objectives but are output by the evaluate hyperparameters function
general_tracking_metrics = [
    "mean_validation_loss",
    "cumulative_validation_loss",
    "mean_test_loss",
    "cumulative_test_loss",
    "testing_time",
    "validation_time",
    "training_time",
    "checkpoint_size",
]

single_objective_added_metrics = ["number_parameters"]
classification_metrics = ["validation_accuracy"]
regression_metrics = [
    "validation_mse",
    "validation_nrmse_mean",
    "validation_nrmse_range",
    "validation_nrmse_std",
    "test_mse",
    "test_nrmse_mean",
    "test_nrmse_std",
]

# Combinations for actual use in experiments
classification_tracking_metrics_multi = (
    general_tracking_metrics + classification_metrics
)
regression_tracking_metrics_multi = general_tracking_metrics + regression_metrics


classification_tracking_metrics_single = (
    general_tracking_metrics + classification_metrics + single_objective_added_metrics
)
regression_tracking_metrics_single = (
    general_tracking_metrics + regression_metrics + single_objective_added_metrics
)