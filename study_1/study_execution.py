# code to replicate study procedure;
# exact replication may not be possible given most experiments
# did not use a particular random seed
# and outputs may not be placed in exactly the same directories
# due to different organization post-execution

# WARNING: THIS TAKES DAYS TO RUN ON A GAMING LAPTOP
# RUN AT OWN RISK

# standard imports
import torch.nn as nn
from multiprocessing import freeze_support

# acquisition functions
from botorch.acquisition.logei import (
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.joint_entropy_search import qJointEntropySearch

# adding all the modules and submodules to the path
import sys

sys.path.insert(0, "C:/Users/Gil/Documents/Repositories/Python/CS_6140/Project")

# importing the correct package defined functions
from src.network.load_data import get_MNIST_data, get_Superconductivity_data
from src.Ax_BO.conduct_experiment import conduct_experiment
import src.Ax_BO.experiment_definition as exp_def

SEED = 0

if __name__ == "__main__":
    # needed for multithreading support on Windows
    freeze_support()

    # Data Loading

    (
        trainset_MNIST,
        validset_MNIST,
        testset_MNIST,
        trainloader_MNIST,
        validloader_MNIST,
        testloader_MNIST,
    ) = get_MNIST_data(
        valid_fraction=0.2,
        random_seed=SEED,
        n_workers=15,
        batch_n=64,
        download_data=False,
    )

    (
        fulldataset_Super,
        trainset_Super,
        validset_Super,
        testset_Super,
        trainloader_Super,
        validloader_Super,
        testloader_Super,
    ) = get_Superconductivity_data(
        valid_fraction=0.2,
        test_fraction=0.2,
        random_seed=SEED,
        n_workers=15,
        batch_n=20,
        local=False,
    )

    for i in range(5):

        # random classification, single objective
        MNIST_experiment_df, MNIST_client_object, MNIST_experiment_object = (
            conduct_experiment(
                task="classification",
                parameter_space=exp_def.MNIST_parameters,
                objective=exp_def.MNIST_single_objective,
                param_constraints=exp_def.p_constraints,
                out_constraints=exp_def.o_constraints,
                tracking_metrics=exp_def.classification_tracking_metrics_single,
                acquisition_func_class=qJointEntropySearch,
                train_loader=trainloader_MNIST,
                valid_loader=validloader_MNIST,
                test_loader=testloader_MNIST,
                input_shape=(1, 28, 28),
                number_input_features=784,
                number_output_features=10,
                loss=nn.CrossEntropyLoss(),
                max_trials=50,
                num_reps_per_trial=1,
                max_epochs=5,
                experiment_name=f"Random_MNIST_50_{i+1}",
                global_early_stop=False,
                fully_random=True,
                interactive_plots=False,
                seed=SEED,
            )
        )

        # random regression, single objective
        Super_experiment_df, Super_client_object, Super_experiment_object = (
            conduct_experiment(
                task="regression",
                parameter_space=exp_def.Superconductivity_parameters,
                objective=exp_def.Superconductivity_single_objective,
                param_constraints=exp_def.p_constraints,
                out_constraints=exp_def.o_constraints,
                tracking_metrics=exp_def.regression_tracking_metrics_single,
                acquisition_func_class=qJointEntropySearch,
                train_loader=trainloader_Super,
                valid_loader=validloader_Super,
                test_loader=testloader_Super,
                input_shape=(1, 1, 81),
                number_input_features=81,
                number_output_features=1,
                loss=nn.HuberLoss(),
                max_trials=50,
                num_reps_per_trial=1,
                max_epochs=5,
                experiment_name=f"Random_Super_50_{i+1}",
                global_early_stop=False,
                fully_random=True,
                interactive_plots=False,
                seed=SEED,
            )
        )

        # non-random classification, single objective, MC-sampler based Logarithm of Noisy Expected Improvement
        MNIST_experiment_df, MNIST_client_object, MNIST_experiment_object = (
            conduct_experiment(
                task="classification",
                parameter_space=exp_def.MNIST_parameters,
                objective=exp_def.MNIST_single_objective,
                param_constraints=exp_def.p_constraints,
                out_constraints=exp_def.o_constraints,
                tracking_metrics=exp_def.classification_tracking_metrics_single,
                acquisition_func_class=qLogNoisyExpectedImprovement,
                train_loader=trainloader_MNIST,
                valid_loader=validloader_MNIST,
                test_loader=testloader_MNIST,
                input_shape=(1, 28, 28),
                number_input_features=784,
                number_output_features=10,
                loss=nn.CrossEntropyLoss(),
                max_trials=50,
                num_reps_per_trial=1,
                max_epochs=5,
                experiment_name=f"qLogNEI_MNIST_50_{i+1}",
                global_early_stop=False,
                fully_random=False,
                interactive_plots=True,
                seed=SEED,
            )
        )

        # non-random regression, single objective, MC-sampler based Logarithm of Noisy Expected Improvement
        Super_experiment_df, Super_client_object, Super_experiment_object = (
            conduct_experiment(
                task="regression",
                parameter_space=exp_def.Superconductivity_parameters,
                objective=exp_def.Superconductivity_single_objective,
                param_constraints=exp_def.p_constraints,
                out_constraints=exp_def.o_constraints,
                tracking_metrics=exp_def.regression_tracking_metrics_single,
                acquisition_func_class=qLogNoisyExpectedImprovement,
                train_loader=trainloader_Super,
                valid_loader=validloader_Super,
                test_loader=testloader_Super,
                input_shape=(1, 1, 81),
                number_input_features=81,
                number_output_features=1,
                loss=nn.HuberLoss(),
                max_trials=50,
                num_reps_per_trial=1,
                max_epochs=5,
                experiment_name=f"qLogNEI_Super_50_{i+1}",
                global_early_stop=False,
                fully_random=False,
                interactive_plots=True,
                seed=None,
            )
        )

        # non-random classification, single objective, MC-sampler based Joint Entropy Search
        MNIST_experiment_df, MNIST_client_object, MNIST_experiment_object = (
            conduct_experiment(
                task="classification",
                parameter_space=exp_def.MNIST_parameters,
                objective=exp_def.MNIST_single_objective,
                param_constraints=exp_def.p_constraints,
                out_constraints=exp_def.o_constraints,
                tracking_metrics=exp_def.classification_tracking_metrics_single,
                acquisition_func_class=qJointEntropySearch,
                train_loader=trainloader_MNIST,
                valid_loader=validloader_MNIST,
                test_loader=testloader_MNIST,
                input_shape=(1, 28, 28),
                number_input_features=784,
                number_output_features=10,
                loss=nn.CrossEntropyLoss(),
                max_trials=50,
                num_reps_per_trial=1,
                max_epochs=5,
                experiment_name=f"qJES_MNIST_50_{i+1}",
                global_early_stop=False,
                fully_random=False,
                interactive_plots=True,
                seed=None,
            )
        )

        # non-random regression, single objective, MC-sampler based Joint Entropy Search
        Super_experiment_df, Super_client_object, Super_experiment_object = (
            conduct_experiment(
                task="regression",
                parameter_space=exp_def.Superconductivity_parameters,
                objective=exp_def.Superconductivity_single_objective,
                param_constraints=exp_def.p_constraints,
                out_constraints=exp_def.o_constraints,
                tracking_metrics=exp_def.regression_tracking_metrics_single,
                acquisition_func_class=qJointEntropySearch,
                train_loader=trainloader_Super,
                valid_loader=validloader_Super,
                test_loader=testloader_Super,
                input_shape=(1, 1, 81),
                number_input_features=81,
                number_output_features=1,
                loss=nn.HuberLoss(),
                max_trials=50,
                num_reps_per_trial=1,
                max_epochs=5,
                experiment_name=f"qJES_Super_50_{i+1}",
                global_early_stop=False,
                fully_random=False,
                interactive_plots=True,
                seed=None,
            )
        )
