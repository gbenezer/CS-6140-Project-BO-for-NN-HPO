# Functions to output custom neural networks using PyTorch Lightning
# current restrictions/assumptions are:
# - 3 layers (could be changed in theory)
# - feedforward (can't be changed easily)
# - uses only Adam optimizer (could be changed in theory)

# main import statements
import torch
from torch import nn
import lightning as L
from typing_extensions import Literal
from torchmetrics.functional.classification.accuracy import multiclass_accuracy
from torchmetrics.functional.regression.mse import mean_squared_error
from torchmetrics.functional.regression.nrmse import normalized_root_mean_squared_error

# Some code adapted from https://github.com/pytorch/tutorials/blob/main/intermediate_source/mnist_train_nas.py
# to be able to handle more general feedforward architecture for both classification and regression

# TODO: Docstring
# TODO: further commenting


def create_ff_model(
    task: Literal["regression", "classification"],
    input_shape: tuple,
    number_input_features: int,
    number_output_features: int,
    input_dropout_probability: float,
    hidden_dropout_probability: float,
    output_dropout_probability: float,
    hidden_layer_nodes_1: int,
    hidden_layer_nodes_2: int,
    hidden_layer_nodes_3: int,
    activation: Literal["swish", "sigmoid", "relu", "leaky_relu"] | nn.Module,
    loss: nn.modules.loss._Loss,
    learning_rate: float,
    beta1: float,
    beta2: float,
    w_decay: float,
) -> L.LightningModule:
    """Factory function for taking a set of neural network hyperparameters and creating a PyTorch Lightning Module defining a network and its training

    Args:
        task (Literal["regression", "classification"]): what task the feedforward network should be trained for
        input_shape (tuple): the shape of an input datum (needs to be at least 3 dimensional, even if several dimensions are ones)
        number_input_features (int): number of features for the network to process at the input
        number_output_features (int): number of classes, probabilities, or values for the network to output
        input_dropout_probability (float): probability of dropout between input and first hidden layer
        hidden_dropout_probability (float): probability of dropout between hidden layers
        output_dropout_probability (float): probability of dropout between last hidden layer and output
        hidden_layer_nodes_1 (int): number of hidden nodes in layer 1 of feedforward network 
        hidden_layer_nodes_2 (int): number of hidden nodes in layer 2 of feedforward network 
        hidden_layer_nodes_3 (int): number of hidden nodes in layer 3 of feedforward network 
        activation (Literal["swish", "sigmoid", "relu", "leaky_relu"] | nn.Module): activation function to use
        loss (nn.modules.loss._Loss): loss function to use
        learning_rate (float): learning rate for the Adam optimizer
        beta1 (float): momentum constant for the Adam optimizer
        beta2 (float): adaptive learning rate constant for the Adam optimizer
        w_decay (float): weight decay constant for the Adam optimizer

    Returns:
        L.LightningModule: A PyTorch Lightning Module with appropriate methods of training and testing that contains a PyTorch 
        feedforward neural network architecture coupled with a configured Adam optimizer
    """
    # allow for interoperation with Ax parameterization
    if activation == "sigmoid":
        activation = nn.Sigmoid()
    elif activation == "swish":
        activation = nn.SiLU()
    elif activation == "relu":
        activation = nn.ReLU()
    elif activation == "leaky_relu":
        activation = (
            nn.LeakyReLU()
        )  # set to default negative slope given hierarchical/conditional nature of parameter

    class FeedForwardModel(L.LightningModule):
        def __init__(self):
            super().__init__()

            # Create a PyTorch model
            layers = [nn.Flatten(), nn.Dropout(p=input_dropout_probability)]
            width = number_input_features

            # following could be changed so that a list of hidden layer node numbers is passed in
            # to allow for a variable number of hidden layers
            # (and same with hidden layer dropout probabilities along with activation functions)

            # dimensionality of hyperparameter space would change as the number of hidden layers was varied
            # may explode
            hidden_layers = [
                hidden_layer_nodes_1,
                hidden_layer_nodes_2,
                hidden_layer_nodes_3,
            ]

            num_params = 0
            for hidden_size in hidden_layers:
                if hidden_size > 0:
                    layers.append(nn.Linear(width, hidden_size))
                    layers.append(activation)
                    layers.append(nn.Dropout(p=hidden_dropout_probability))
                    num_params += width * hidden_size
                    width = hidden_size
            # neccessary pop to remove the last dropoout and replace it with
            # output probability
            layers.pop()
            layers.append(nn.Dropout(p=output_dropout_probability))
            layers.append(nn.Linear(width, number_output_features))
            num_params += width * number_output_features

            # Save the model and parameter counts
            self.num_params = num_params
            self.model = nn.Sequential(*layers)

            # for graph tracing
            self.example_input_array = torch.rand(size=input_shape)

        def forward(self, x):
            x = self.model(x)
            x = torch.squeeze(x)
            return x

        def training_step(self, batch, batch_idx):

            x, y = batch
            yhat = self(x)
            training_loss = loss(yhat, y)
            self.log("training_loss", training_loss)
            self.log("training_loss_total", training_loss, reduce_fx="sum")

            if task == "classification":
                preds = torch.argmax(yhat, dim=1)
                acc = multiclass_accuracy(preds, y, num_classes=number_output_features)
                self.log("training_accuracy", acc)

            elif task == "regression":
                mse = mean_squared_error(yhat, y, num_outputs=number_output_features)
                nrmse_mean = normalized_root_mean_squared_error(
                    yhat, y, normalization="mean", num_outputs=number_output_features
                )
                nrmse_range = normalized_root_mean_squared_error(
                    yhat, y, normalization="range", num_outputs=number_output_features
                )
                nrmse_std = normalized_root_mean_squared_error(
                    yhat, y, normalization="std", num_outputs=number_output_features
                )
                self.log("training_mse", mse)
                self.log("training_nrmse_mean", nrmse_mean)
                self.log("training_nrmse_range", nrmse_range)
                self.log("training_nrmse_std", nrmse_std)

            return training_loss

        def validation_step(self, batch, batch_idx):

            x, y = batch
            yhat = self(x)
            validation_loss = loss(yhat, y)
            self.log("mean_validation_loss", validation_loss)
            self.log(
                "cumulative_validation_loss",
                validation_loss,
                reduce_fx="sum",
            )
            self.log("number_parameters", self.num_params, reduce_fx="max")

            if task == "classification":
                preds = torch.argmax(yhat, dim=1)
                acc = multiclass_accuracy(preds, y, num_classes=number_output_features)
                self.log("validation_accuracy", acc)

            elif task == "regression":
                mse = mean_squared_error(yhat, y, num_outputs=number_output_features)
                nrmse_mean = normalized_root_mean_squared_error(
                    yhat, y, normalization="mean", num_outputs=number_output_features
                )
                nrmse_range = normalized_root_mean_squared_error(
                    yhat, y, normalization="range", num_outputs=number_output_features
                )
                nrmse_std = normalized_root_mean_squared_error(
                    yhat, y, normalization="std", num_outputs=number_output_features
                )
                self.log("validation_mse", mse)
                self.log("validation_nrmse_mean", nrmse_mean)
                self.log("validation_nrmse_range", nrmse_range)
                self.log("validation_nrmse_std", nrmse_std)
            return validation_loss

        def test_step(self, batch, batch_idx):

            x, y = batch
            yhat = self(x)
            test_loss = loss(yhat, y)
            self.log("mean_test_loss", test_loss)
            self.log("cumulative_test_loss", test_loss, reduce_fx="sum")
            self.log("number_parameters", self.num_params, reduce_fx="max")

            if task == "classification":
                preds = torch.argmax(yhat, dim=1)
                acc = multiclass_accuracy(preds, y, num_classes=number_output_features)
                self.log("test_accuracy", acc)

            elif task == "regression":
                mse = mean_squared_error(yhat, y, num_outputs=number_output_features)
                nrmse_mean = normalized_root_mean_squared_error(
                    yhat, y, normalization="mean", num_outputs=number_output_features
                )
                nrmse_range = normalized_root_mean_squared_error(
                    yhat, y, normalization="range", num_outputs=number_output_features
                )
                nrmse_std = normalized_root_mean_squared_error(
                    yhat, y, normalization="std", num_outputs=number_output_features
                )
                self.log("test_mse", mse)
                self.log("test_nrmse_mean", nrmse_mean)
                self.log("test_nrmse_range", nrmse_range)
                self.log("test_nrmse_std", nrmse_std)

            return test_loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2),
                weight_decay=w_decay,
            )
            return optimizer

    return FeedForwardModel()

# NOT USED, meant to allow for variation in number of neural network layers
# def create_ff_model_varied_layers(
#     task: Literal["regression", "classification"],
#     input_shape: tuple,
#     number_input_features: int,
#     number_output_features: int,
#     input_dropout_probability: float,
#     hidden_dropout_probability: float,
#     output_dropout_probability: float,
#     hidden_layer_nodes: tuple,
#     activations: list,
#     loss: nn.modules.loss._Loss,
#     learning_rate: float,
#     beta1: float,
#     beta2: float,
#     w_decay: float,
# ):
#     """_summary_

#     Args:
#         task (Literal["regression", "classification"]): _description_
#         input_shape (tuple): _description_
#         number_input_features (int): _description_
#         number_output_features (int): _description_
#         input_dropout_probability (float): _description_
#         hidden_dropout_probability (float): _description_
#         output_dropout_probability (float): _description_
#         hidden_layer_nodes (tuple): _description_
#         activations (list): _description_
#         loss (nn.modules.loss._Loss): _description_
#         learning_rate (float): _description_
#         beta1 (float): _description_
#         beta2 (float): _description_
#         w_decay (float): _description_

#     Returns:
#         _type_: _description_
#     """
#     # TODO: more explicitly restrict activations variable
#     # TODO: explicitly check on number of activations versus number of layers

#     class FeedForwardModel(L.LightningModule):
#         def __init__(self):
#             super().__init__()

#             # Create a PyTorch model
#             layers = [nn.Flatten(), nn.Dropout(p=input_dropout_probability)]
#             width = number_input_features

#             num_params = 0
#             for idx, hidden_size in enumerate(hidden_layer_nodes):
#                 if hidden_size > 0:
#                     layers.append(nn.Linear(width, hidden_size))
#                     layers.append(activations[idx])
#                     layers.append(nn.Dropout(p=hidden_dropout_probability))
#                     num_params += width * hidden_size
#                     width = hidden_size
#             # neccessary pop to remove the last dropoout and replace it with
#             # output probability
#             layers.pop()
#             layers.append(nn.Dropout(p=output_dropout_probability))
#             layers.append(nn.Linear(width, number_output_features))
#             num_params += width * number_output_features

#             # Save the model and parameter counts
#             self.num_params = num_params
#             self.model = nn.Sequential(*layers)

#             # for graph tracing
#             self.example_input_array = torch.rand(size=input_shape)

#         def forward(self, x):
#             x = self.model(x)
#             x = torch.squeeze(x)
#             return x

#         def training_step(self, batch, batch_idx):

#             x, y = batch
#             yhat = self(x)
#             training_loss = loss(yhat, y)
#             self.log("training_loss", training_loss)
#             self.log("training_loss_total", training_loss, reduce_fx="sum")

#             if task == "classification":
#                 preds = torch.argmax(yhat, dim=1)
#                 acc = multiclass_accuracy(preds, y, num_classes=number_output_features)
#                 self.log("training_accuracy", acc)

#             elif task == "regression":
#                 mse = mean_squared_error(yhat, y, num_outputs=number_output_features)
#                 nrmse_mean = normalized_root_mean_squared_error(
#                     yhat, y, normalization="mean", num_outputs=number_output_features
#                 )
#                 nrmse_range = normalized_root_mean_squared_error(
#                     yhat, y, normalization="range", num_outputs=number_output_features
#                 )
#                 nrmse_std = normalized_root_mean_squared_error(
#                     yhat, y, normalization="std", num_outputs=number_output_features
#                 )
#                 self.log("training_mse", mse)
#                 self.log("training_nrmse_mean", nrmse_mean)
#                 self.log("training_nrmse_range", nrmse_range)
#                 self.log("training_nrmse_std", nrmse_std)

#             return training_loss

#         def validation_step(self, batch, batch_idx):

#             x, y = batch
#             yhat = self(x)
#             validation_loss = loss(yhat, y)
#             self.log("mean_validation_loss", validation_loss)
#             self.log(
#                 "cumulative_validation_loss",
#                 validation_loss,
#                 reduce_fx="sum",
#             )

#             if task == "classification":
#                 preds = torch.argmax(yhat, dim=1)
#                 acc = multiclass_accuracy(preds, y, num_classes=number_output_features)
#                 self.log("validation_accuracy", acc)

#             elif task == "regression":
#                 mse = mean_squared_error(yhat, y, num_outputs=number_output_features)
#                 nrmse_mean = normalized_root_mean_squared_error(
#                     yhat, y, normalization="mean", num_outputs=number_output_features
#                 )
#                 nrmse_range = normalized_root_mean_squared_error(
#                     yhat, y, normalization="range", num_outputs=number_output_features
#                 )
#                 nrmse_std = normalized_root_mean_squared_error(
#                     yhat, y, normalization="std", num_outputs=number_output_features
#                 )
#                 self.log("validation_mse", mse)
#                 self.log("validation_nrmse_mean", nrmse_mean)
#                 self.log("validation_nrmse_range", nrmse_range)
#                 self.log("validation_nrmse_std", nrmse_std)
#             return validation_loss

#         def test_step(self, batch, batch_idx):

#             x, y = batch
#             yhat = self(x)
#             test_loss = loss(yhat, y)
#             self.log("mean_test_loss", test_loss)
#             self.log("cumulative_test_loss", test_loss, reduce_fx="sum")

#             if task == "classification":
#                 preds = torch.argmax(yhat, dim=1)
#                 acc = multiclass_accuracy(preds, y, num_classes=number_output_features)
#                 self.log("test_accuracy", acc)

#             elif task == "regression":
#                 mse = mean_squared_error(yhat, y, num_outputs=number_output_features)
#                 nrmse_mean = normalized_root_mean_squared_error(
#                     yhat, y, normalization="mean", num_outputs=number_output_features
#                 )
#                 nrmse_range = normalized_root_mean_squared_error(
#                     yhat, y, normalization="range", num_outputs=number_output_features
#                 )
#                 nrmse_std = normalized_root_mean_squared_error(
#                     yhat, y, normalization="std", num_outputs=number_output_features
#                 )
#                 self.log("test_mse", mse)
#                 self.log("test_nrmse_mean", nrmse_mean)
#                 self.log("test_nrmse_range", nrmse_range)
#                 self.log("test_nrmse_std", nrmse_std)

#             return test_loss

#         def configure_optimizers(self):
#             optimizer = torch.optim.Adam(
#                 self.parameters(),
#                 lr=learning_rate,
#                 betas=(beta1, beta2),
#                 weight_decay=w_decay,
#             )
#             return optimizer

#     return FeedForwardModel()
