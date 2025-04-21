# File to determine how function evaluation (neural network hyperparameter performance evaluation)
# will work

# Plans as of 09Mar2025 are to implement method for generation of test accuracy, inference time,
# and inference memory performance as three separate objective functions
# Inference time and memory utilization are subject to more noise, so that may be a consideration

# TODO: function type annotation

from pathlib import Path
import time
import os
import pandas as pd
import numpy as np
import lightning as L
import torch
from typing import Literal
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks import ModelSummary

# adding all the modules and submodules to the path
import sys

sys.path.insert(0, "C:/Users/Gil/Documents/Repositories/Python/CS_6140/Project")

from src.network.create_network_lightning import create_ff_model

def evaluate_hyperparameters(
    task: Literal["regression", "classification"],
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    input_shape: tuple,
    number_input_features: int,
    number_output_features: int,
    loss: torch.nn.modules.loss._Loss,
    log_dir_name: str,
    num_rep: int,
    max_epochs: int,
    parameterization: dict,
):
    """_summary_

    Args:
        task (Literal["regression", "classification"]): _description_
        train_loader (torch.utils.data.DataLoader): _description_
        valid_loader (torch.utils.data.DataLoader): _description_
        test_loader (torch.utils.data.DataLoader): _description_
        input_shape (tuple): _description_
        number_input_features (int): _description_
        number_output_features (int): _description_
        loss (torch.nn.modules.loss._Loss): _description_
        log_dir_name (str): _description_
        num_rep (int): _description_
        max_epochs (int): _description_
        parameterization (dict): _description_

    Returns:
        _type_: _description_
    """
    # define some lists to keep track of outputs
    training_times = []
    validation_times = []
    testing_times = []
    validation_outputs = []
    testing_outputs = []
    checkpoint_sizes = []

    for rep in range(num_rep):

        # generate the LightningModule
        # TODO: allow for the use of the create_ff_model_varied_layers function

        test_module = create_ff_model(
            task=task,
            input_shape=input_shape,
            number_input_features=number_input_features,
            number_output_features=number_output_features,
            loss=loss,
            **parameterization,
        )
        print(test_module.model)

        # create the loggers and profiler
        tensorboard_logger = pl_loggers.TensorBoardLogger(
            save_dir="logs/tb_logs/",
            name=log_dir_name,
            version=f"replicate_{rep}",
            log_graph=True,
        )
        csv_logger = pl_loggers.CSVLogger(
            save_dir="logs/csv_logs/trial_logs",
            name=log_dir_name,
            version=f"replicate_{rep}",
        )

        tensorboard_dir_path = Path(tensorboard_logger.log_dir)
        checkpoint_dir_path = tensorboard_dir_path / "checkpoints"
        simple_profiler = SimpleProfiler(filename="profiling_output")

        # create Trainer
        trainer = L.Trainer(
            max_epochs=max_epochs,
            enable_progress_bar=False,
            logger=[tensorboard_logger, csv_logger],
            profiler=simple_profiler,
            callbacks=[ModelSummary(max_depth=5)]
        )

        # train the module and calculate training time
        fit_start = time.time()

        trainer.fit(
            model=test_module,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )

        fit_end = time.time()
        training_time = (fit_end - fit_start) / 60.0
        training_times.append(training_time)

        # validate the model and capture validation outputs
        validate_start = time.time()
        validation_output = trainer.validate(
            model=test_module, dataloaders=valid_loader
        )
        validate_end = time.time()
        validation_time = (validate_end - validate_start) / 60.0
        validation_times.append(validation_time)
        validation_outputs.append(validation_output[0])

        # test the model and capture test outputs
        test_start = time.time()
        testing_output = trainer.test(model=test_module, dataloaders=test_loader)
        test_end = time.time()
        testing_time = (test_end - test_start) / 60.0
        testing_times.append(testing_time)
        testing_outputs.append(testing_output[0])

        size = 0
        for file in os.scandir(checkpoint_dir_path):
            size += os.path.getsize(file)

        checkpoint_size = size
        checkpoint_sizes.append(checkpoint_size)

    # getting number of parameters
    number_parameters = test_module.num_params

    # number of parameters is an integer and has no standard error of mean (std / sqrt(n))
    # needs to be specified anyway
    
    
    if num_rep > 1:
        # calculating and adding metrics and their SEMs
        output_dict = {"number_parameters": (int(number_parameters), 0.0)}
        output_dict["training_time"] = (
            np.mean(training_times),
            np.std(training_times) / np.sqrt(float(len(training_times))),
        )
        output_dict["validation_time"] = (
            np.mean(validation_times),
            np.std(validation_times) / np.sqrt(float(len(validation_times))),
        )
        output_dict["testing_time"] = (
            np.mean(testing_times),
            np.std(testing_times) / np.sqrt(float(len(testing_times))),
        )
        output_dict["checkpoint_size"] = (
            np.mean(checkpoint_sizes),
            np.std(checkpoint_sizes) / np.sqrt(float(len(checkpoint_sizes))),
        )
    else:
        # calculating and adding metrics without SEMs
        output_dict = {"number_parameters": int(number_parameters)}
        output_dict["training_time"] = np.mean(training_times)
        output_dict["validation_time"] = np.mean(validation_times)
        output_dict["testing_time"] = np.mean(testing_times)
        output_dict["checkpoint_size"] = np.mean(checkpoint_sizes)

    # creating single dictionaries from lists of dictionaries through a
    # DataFrame intermediate
    validation_dict = pd.DataFrame(validation_outputs).to_dict(orient="list")
    testing_dict = pd.DataFrame(testing_outputs).to_dict(orient="list")
    
    # deleting the number of parameters from each of the dictionaries as it is already in the output dictionary
    del validation_dict["number_parameters"]
    del testing_dict["number_parameters"]
    
    if num_rep > 1:
        # summarizing each of the variables in a format according to the expected experiment output
        validation_summary_dict = {k: (np.mean(v), (np.std(v) / np.sqrt(float(len(v))))) for k, v in validation_dict.items()}
        testing_summary_dict = {k: (np.mean(v), (np.std(v) / np.sqrt(float(len(v))))) for k, v in testing_dict.items()}
    else:
        # summarizing each of the variables in a format according to the expected experiment output
        validation_summary_dict = {k: np.mean(v) for k, v in validation_dict.items()}
        testing_summary_dict = {k: np.mean(v) for k, v in testing_dict.items()}
    
    # combine all the dictionaries into the output dictionary
    output_dict.update(validation_summary_dict)
    output_dict.update(testing_summary_dict)

    return output_dict