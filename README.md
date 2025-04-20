# Influence of Acquisition Functions on Bayesian Optimization of Neural Network Hyperparameters

This project aims to evaluate Bayesian optimization for tuning neural network hyperparameters and architecture search using different acquisition functions and using environmental variables (like training set size, number of epochs, or function evaluations) as tunable hyperparameters in and of themselves.

The current plan is to use [PyTorch Lightning](https://pytorch.org/) for the implementation of the base neural network generation functionality, [Ax](https://ax.dev/) and [BoTorch](https://botorch.org/) for Bayesian optimization using Gaussian Process-based surrogate models and implemented acquisition functions.

## General Overview of Bayesian Optimization

The general workflow for Bayesian optimization in the context of neural network hyperparameter optimization and architecture search is as follows

1. Define the hyparparameter space $\mathcal{S}$ and any initial hyper-hyperparameters
2. Sample the space randomly to generate an initial set of hyperparameter combinations
    - Generate these combinations in a space-filling manner if possible
3. Evaluate the objective function(s) for each generated and trained neural network
4. Use these initial data to generate estimates for the posterior probability distribution(s) of how the objective function(s) vary with respect to neural network hyperparameters of interest
5. Generate a next set of hyperparameters to test by maximizing an acquisition function using the posterior(s)
6. Evaluate the generated set of hyperparameters
7. Update posterior probability distribution(s)
8. Repeat steps 5 to 7

## Necessary Components

The project will need implementation of the following components

- Initial hyperparameter combination generation
- Neural network generation
- Neural network training and evaluation with respect to objective function(s)
- Gaussian Process construction and Bayesian updating
- Result visualization

## Restrictions on Neural Network Architectures
