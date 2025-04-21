from pathlib import Path
import os
import torch
from typing import Literal
import pandas as pd
import numpy as np
import ax.plot.pareto_frontier as pareto_frontier
import ax.plot.pareto_utils as pareto_utils
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.service.ax_client import AxClient
from ax.exceptions.core import OptimizationShouldStop
from ax.plot.contour import interact_contour
from ax.plot.diagnostic import interact_cross_validation
from ax.plot.scatter import interact_fitted, plot_objective_vs_constraints
from ax.modelbridge.cross_validation import cross_validate
from ax.storage.botorch_modular_registry import (
    ACQUISITION_FUNCTION_REGISTRY,
    register_acquisition_function,
)
import plotly.io as pio
from botorch.acquisition import AcquisitionFunction
from ax.storage.json_store.save import save_experiment
from src.network.evaluate_network import evaluate_hyperparameters
from src.Ax_BO.threshold_global_stopping import ThresholdGlobalStoppingStrategy


# TODO: add function docstring
def conduct_experiment(
    task: Literal["regression", "classification"],
    parameter_space: dict,
    objective: dict,
    param_constraints: list | None,
    out_constraints: list | None,
    tracking_metrics: list | None,
    acquisition_func_class: AcquisitionFunction,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    input_shape: tuple,
    number_input_features: int,
    number_output_features: int,
    loss: torch.nn.modules.loss._Loss,
    max_trials: int,
    num_reps_per_trial: int,
    max_epochs: int,
    experiment_name: str,
    fully_random: bool,
    interactive_plots: bool,
    global_early_stop: bool,
    seed: int | None = None,
):
    """_summary_

    Args:
        task (Literal[&quot;regression&quot;, &quot;classification&quot;]): _description_
        parameter_space (dict): _description_
        objective (dict): _description_
        param_constraints (list | None): _description_
        out_constraints (list | None): _description_
        tracking_metrics (list | None): _description_
        acquisition_func_class (AcquisitionFunction): _description_
        train_loader (torch.utils.data.DataLoader): _description_
        valid_loader (torch.utils.data.DataLoader): _description_
        test_loader (torch.utils.data.DataLoader): _description_
        input_shape (tuple): _description_
        number_input_features (int): _description_
        number_output_features (int): _description_
        loss (torch.nn.modules.loss._Loss): _description_
        max_trials (int): _description_
        num_reps_per_trial (int): _description_
        max_epochs (int): _description_
        experiment_name (str): _description_
        fully_random (bool): _description_
        interactive_plots (bool): _description_
        global_early_stop (bool): _description_
        seed (int | None, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # making the experiment CSV directory
    experiment_df_dir = Path(os.getcwd()) / "logs" / "csv_logs" / "experiment_logs"
    JSON_dir = Path(os.getcwd()) / "logs" / "JSON_logs"
    finished_JSON_dir = JSON_dir / "experiments"
    experiment_JSON_dir = JSON_dir / "experiments_snapshots" / experiment_name
    client_JSON_dir = JSON_dir / "AxClient_snapshots" / experiment_name

    if not os.path.exists(experiment_df_dir):
        experiment_df_dir.mkdir(parents=True)
    if not os.path.exists(finished_JSON_dir):
        finished_JSON_dir.mkdir(parents=True)
    if not os.path.exists(experiment_JSON_dir):
        experiment_JSON_dir.mkdir(parents=True)
    if not os.path.exists(client_JSON_dir):
        client_JSON_dir.mkdir(parents=True)

    # the minimum number of initialization trials should be twice the parameter space dimension
    # (guideline heuristic in BoTorch is max(5, 2 * len(parameter_space))),
    # and all the parameter spaces
    # I am looking at are higher dimensional by a factor of 2
    min_number_initialization_trials = 2 * len(parameter_space)

    # if the trial is supposed to be fully random,
    # force the AxClient to only use fully quasi-random search
    if fully_random:
        gen_strat = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.UNIFORM,  # as random as possible
                    num_trials=-1,  # unlimited number of trials can be generated
                )
            ],
            name="uniform_random",
        )
    else:
        gen_strat = GenerationStrategy(
            steps=[
                GenerationStep(
                    # quasi-random space filling generation to effectively initialize posterior
                    model=Models.SOBOL,
                    # minimum and maximum are the same for this strategy
                    num_trials=min_number_initialization_trials,
                    min_trials_observed=min_number_initialization_trials,
                ),
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=-1,  # unlimited
                    model_kwargs={  # Kwargs to pass to `BoTorchModel.__init__`
                        "botorch_acqf_class": acquisition_func_class,
                    },
                ),
            ],
            name="custom_acquisition_function_and_model",
        )

    print("generation strategy name:", gen_strat.name)

    # NOTE: This will break for multi-objective problems
    # so set global_early_stop to False for multi-objective problems
    if global_early_stop and task == "classification":
        # 95% test accuracy threshold
        gss = ThresholdGlobalStoppingStrategy(min_trials=5, threshold=0.97)
    elif global_early_stop and task == "regression":
        # 0.2 range-normalized root mean square deviation
        gss = ThresholdGlobalStoppingStrategy(min_trials=5, threshold=0.1)
    else:
        gss = None

    print("global early stopping strategy:", gss)

    # add the acquisition function to the acquisition function registry if necessary for saving to JSON format
    if acquisition_func_class not in ACQUISITION_FUNCTION_REGISTRY.keys():
        register_acquisition_function(acqf_class=acquisition_func_class)

    # create AxClient object, experiment, and add tracking metrics
    ax_client = AxClient(
        generation_strategy=gen_strat, random_seed=seed, global_stopping_strategy=gss
    )

    ax_client.create_experiment(
        name=experiment_name,
        parameters=parameter_space,
        objectives=objective,
        parameter_constraints=param_constraints,
        outcome_constraints=out_constraints,
    )

    ax_client.add_tracking_metrics(tracking_metrics)

    print("AxClient Object post-initialization:", ax_client)

    # creating the name and path of an unofficial experiment csv
    unofficial_experiment_csv = experiment_name + "_unofficial.csv"
    unofficial_experiment_csv_path = experiment_df_dir / unofficial_experiment_csv

    print("unofficial experiment csv path", unofficial_experiment_csv_path)

    ax_client.save_to_json_file(
        filepath=str(client_JSON_dir / "ax_client_snapshot_0.json")
    )
    save_experiment(
        experiment=ax_client.experiment,
        filepath=str(experiment_JSON_dir / "experiment_snapshot_0.json"),
    )

    # run the optimization loop
    for i in range(max_trials):

        # this will fully stop the experiment due to global stopping strategy
        try:
            parameterization, trial_index = ax_client.get_next_trial()
        except OptimizationShouldStop as exc:
            print(exc.message)
            break

        # print("next trial parameterization:", parameterization)

        trial_log_dir_name = experiment_name + "/" + f"trial_{trial_index}"

        # print("trial_log_dir_name:", trial_log_dir_name)

        trial_data = evaluate_hyperparameters(
            task=task,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            input_shape=input_shape,
            number_input_features=number_input_features,
            number_output_features=number_output_features,
            loss=loss,
            log_dir_name=trial_log_dir_name,
            num_rep=num_reps_per_trial,
            max_epochs=max_epochs,
            parameterization=parameterization,
        )

        # print("trial_data:", trial_data)

        # test whether or not there are absurd test loss values
        # (indexing needed when tuple is returned)

        # NOTE: empirically, mean test loss appears to be below 10 for MNIST and
        # below 100 for Superconductivity
        message_1 = "mean test loss too high"
        if task == "classification":
            if num_reps_per_trial > 1:
                if trial_data["mean_test_loss"][0] > 10:
                    ax_client.abandon_trial(trial_index=trial_index, reason=message_1)
                    continue

            elif trial_data["mean_test_loss"] > 10:
                ax_client.abandon_trial(trial_index=trial_index, reason=message_1)
                continue
        elif task == "regression":
            if num_reps_per_trial > 1:
                if trial_data["mean_test_loss"][0] > 100:
                    ax_client.abandon_trial(trial_index=trial_index, reason=message_1)
                    continue

            elif trial_data["mean_test_loss"] > 100:
                ax_client.abandon_trial(trial_index=trial_index, reason=message_1)
                continue

        # Either create or append the data from the trial
        # to the unofficial experiment csv so that the data can be saved in the event of
        # an incomplete experiment

        unofficial_row_data = {}
        for k, v in trial_data.items():
            if not isinstance(v, tuple):
                unofficial_row_data[k] = [v]
            else:
                unofficial_row_data[k] = [v[0]]

        dataframe_parameterization = {k: [v] for k, v in parameterization.items()}
        unofficial_row_data.update(dataframe_parameterization)
        unofficial_row_data["trial_index"] = [trial_index]
        row_df = pd.DataFrame(unofficial_row_data)

        # if the file does not yet exist, make it
        if not os.path.exists(unofficial_experiment_csv_path):
            row_df.to_csv(
                path_or_buf=unofficial_experiment_csv_path, mode="w", header=True
            )
        else:
            row_df.to_csv(
                path_or_buf=unofficial_experiment_csv_path, mode="a", header=False
            )

        # another attempt at testing for NaN entries
        nan_present = False
        message_2 = "NaN entries in metrics"
        for metric_value in trial_data.values():
            if not isinstance(metric_value, tuple):
                if np.isnan(metric_value):
                    ax_client.abandon_trial(trial_index=trial_index, reason=message_2)
                    nan_present = True
                    break
            else:
                if np.isnan(metric_value[0]):
                    ax_client.abandon_trial(trial_index=trial_index, reason=message_2)
                    nan_present = True
                    break

        if not nan_present:
            ax_client.complete_trial(
                trial_index=trial_index,
                raw_data=trial_data,
            )

        ax_client.save_to_json_file(
            filepath=str(client_JSON_dir / f"ax_client_snapshot_{i+1}.json")
        )
        save_experiment(
            experiment=ax_client.experiment,
            filepath=str(experiment_JSON_dir / f"experiment_snapshot_{i+1}.json"),
        )

    # getting results of experiment
    experiment_df = ax_client.experiment.to_df()
    client_object = ax_client
    experiment_object = ax_client.experiment

    # saving the official results of the experiment to a CSV file
    # and JSON files
    experiment_csv_name = experiment_name + "_official.csv"
    experiment_df.to_csv(path_or_buf=experiment_df_dir / experiment_csv_name)
    client_object.save_to_json_file(
        filepath=str(finished_JSON_dir / f"{experiment_name}_ax_client.json")
    )
    save_experiment(
        experiment=experiment_object,
        filepath=str(finished_JSON_dir / f"{experiment_name}_experiment_file.json"),
    )

    if interactive_plots:

        # making the plot directory
        plot_directory = Path(os.getcwd()) / "plots/html_plots" / experiment_name
        if not os.path.exists(plot_directory):
            plot_directory.mkdir(parents=True)

        # saving interactive plots (only works if there are non-RandomBridgeModel trials completed)
        model = ax_client.generation_strategy.model

        for key in objective.keys():

            plotly_metric_contour = interact_contour(model=model, metric_name=key).data

            pio.write_html(
                fig=plotly_metric_contour,
                file=plot_directory
                / f"{experiment_name}_{key}_interactive_contour.html",
            )

            plotly_tradeoff = plot_objective_vs_constraints(model, key, rel=False).data

            pio.write_html(
                fig=plotly_tradeoff,
                file=plot_directory
                / f"{experiment_name}_tradeoff_{key}_interactive_plot.html",
            )

        cv_results = cross_validate(model)
        plotly_cv = interact_cross_validation(cv_results).data

        plotly_tile = interact_fitted(model, rel=False).data

        pio.write_html(
            fig=plotly_cv, file=plot_directory / f"{experiment_name}_cv_plot.html"
        )
        pio.write_html(
            fig=plotly_tile, file=plot_directory / f"{experiment_name}_tile_plot.html"
        )

        if experiment_object.is_moo_problem:

            # if this is a multiobjective optimization problem, then plot Pareto frontiers
            hypervolume_trace_plot_figure = (
                pareto_frontier.scatter_plot_with_hypervolume_trace_plotly(
                    experiment=experiment_object
                )
            )
            observed_pareto_frontiers = pareto_utils.get_observed_pareto_frontiers(
                experiment=experiment_object
            )
            pareto_frontier_plot_figure = pareto_frontier.interact_pareto_frontier(
                frontier_list=observed_pareto_frontiers
            ).data

            pio.write_html(
                fig=pareto_frontier_plot_figure,
                file=plot_directory / f"{experiment_name}_pareto_frontier.html",
            )
            pio.write_html(
                fig=hypervolume_trace_plot_figure,
                file=plot_directory / f"{experiment_name}_hypervolume_trace.html",
            )

    return experiment_df, client_object, experiment_object
