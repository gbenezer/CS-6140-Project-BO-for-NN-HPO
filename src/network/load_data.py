# file to define how the data are imported as DataSets and DataLoaders
from pathlib import Path
import torchvision.transforms.v2 as transforms
from sklearn.preprocessing import normalize, scale
import torch.utils.data as data
import torch
import pandas as pd
import numpy as np
from torchvision import datasets
from ucimlrepo import fetch_ucirepo


def get_MNIST_data(
    valid_fraction: float,
    random_seed: int,
    n_workers: int,
    batch_n: int,
    download_data: bool,
):
    """_summary_

    Args:
        valid_fraction (float): _description_
        random_seed (int): _description_
        n_workers (int): _description_
        batch_n (int): _description_
        download_data (bool): _description_

    Returns:
        _type_: _description_
    """
    # define the image transformation
    image_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float, scale=True),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # define the data directory and getting the datasets from the directory
    data_directory = Path(__file__).parent.parent.parent.resolve() / "torch_data"
    trainset = datasets.MNIST(
        root=data_directory,
        train=True,
        download=download_data,
        transform=image_transform,
    )
    testset = datasets.MNIST(
        root=data_directory,
        train=False,
        download=download_data,
        transform=image_transform,
    )

    # creating the test data DataLoader
    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        num_workers=n_workers,
        batch_size=batch_n,
        persistent_workers=True,
    )

    # splitting off the validation set from the larger training set
    valid_set_size = int(len(trainset) * valid_fraction)
    train_set_size = len(trainset) - valid_set_size
    seed = torch.Generator().manual_seed(random_seed)
    train_set, valid_set = data.random_split(
        trainset, [train_set_size, valid_set_size], generator=seed
    )

    # defining the training data and validation data DataLoader
    trainloader = torch.utils.data.DataLoader(
        dataset=train_set,
        num_workers=n_workers,
        batch_size=batch_n,
        shuffle=True,
        persistent_workers=True,
    )

    validloader = torch.utils.data.DataLoader(
        dataset=valid_set,
        num_workers=n_workers,
        batch_size=batch_n,
        persistent_workers=True,
    )

    return train_set, valid_set, testset, trainloader, validloader, testloader


# define Dataset class for Superconductivity data
class SuperconductivityDataset(data.Dataset):
    """_summary_

    Args:
        data (_type_): _description_
    """
    def __init__(
        self,
        transform=transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float)]
        ),
        target_transform=transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float)]
        ),
        normalize_samples=True,
        standardize_features=True,
    ):
        super().__init__()
        self.data_object = fetch_ucirepo(id=464)
        self.features = self.data_object.data.features
        self.targets = self.data_object.data.targets
        self.feature_ndarray = self.features.to_numpy()
        if standardize_features:
            self.feature_ndarray = scale(self.feature_ndarray)
        self.target_ndarray = self.targets.to_numpy().squeeze().astype(dtype=np.float32)
        self.transform = transform
        self.normalize_samples = normalize_samples
        self.target_transform = target_transform
        self.metadata = self.data_object.metadata
        self.variables = self.data_object.variables
        self.number_samples = self.features.shape[0]
        self.number_features = self.features.shape[1]

    def __len__(self):
        return self.number_samples

    def __getitem__(self, index):

        # get sample and target
        sample = self.feature_ndarray[index, :].reshape(1, -1)
        target = self.target_ndarray[index]

        # normalize vector if necessary
        if self.normalize_samples:
            sample = normalize(sample)

        # transform sample and target to torch.Tensor
        sample = self.transform(sample)
        target = self.target_transform(target)
        return sample, target


class LocalSuperconductivityDataset(data.Dataset):
    """_summary_

    Args:
        data (_type_): _description_
    """
    def __init__(
        self,
        transform=transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float)]
        ),
        target_transform=transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float)]
        ),
        normalize_samples=True,
        standardize_features=True,
    ):
        super().__init__()
        self.data_object = pd.read_csv(
            "C:\\Users\\Gil\\Documents\\Repositories\\Python\\CS_6140\\Project\\external_data\\superconductivty_data\\train.csv"
        )
        self.unique_molecules = pd.read_csv(
            "C:\\Users\\Gil\\Documents\\Repositories\\Python\\CS_6140\\Project\\external_data\\superconductivty_data\\unique_m.csv"
        )
        self.features = self.data_object.iloc[:, 0:81]
        self.targets = self.data_object.iloc[:, 81]
        self.feature_ndarray = self.features.to_numpy()
        if standardize_features:
            self.feature_ndarray = scale(self.feature_ndarray)
        self.target_ndarray = self.targets.to_numpy().squeeze().astype(dtype=np.float32)
        self.transform = transform
        self.normalize_samples = normalize_samples
        self.target_transform = target_transform
        self.number_samples = self.features.shape[0]
        self.number_features = self.features.shape[1]

    def __len__(self):
        return self.number_samples

    def __getitem__(self, index):

        # get sample and target
        sample = self.feature_ndarray[index, :].reshape(1, -1)
        target = self.target_ndarray[index]

        # normalize vector if necessary
        if self.normalize_samples:
            sample = normalize(sample)

        # transform sample and target to torch.Tensor
        sample = self.transform(sample)
        target = self.target_transform(target)
        return sample, target


# TODO: possibly incorporate external_data/superconductivity_data/unique_m.csv
def get_Superconductivity_data(
    valid_fraction: float,
    test_fraction: float,
    random_seed: int,
    n_workers: int,
    batch_n: int,
    local=False,
):
    """_summary_

    Args:
        valid_fraction (float): _description_
        test_fraction (float): _description_
        random_seed (int): _description_
        n_workers (int): _description_
        batch_n (int): _description_
        local (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # instantiating the full dataset
    if local:
        full_dataset = LocalSuperconductivityDataset()
    else:
        full_dataset = SuperconductivityDataset()

    # splitting off the test dataset
    test_size = int(len(full_dataset) * test_fraction)
    non_test_size = len(full_dataset) - test_size
    seed = torch.Generator().manual_seed(random_seed)
    test_set, non_test_set = data.random_split(
        full_dataset, [test_size, non_test_size], generator=seed
    )

    # splitting the remaining set into training and validation sets
    valid_size = int(len(non_test_set) * valid_fraction)
    train_size = len(non_test_set) - valid_size
    train_set, valid_set = data.random_split(
        non_test_set, [train_size, valid_size], generator=seed
    )

    # creating the DataLoader objects
    # creating the test data DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        num_workers=n_workers,
        batch_size=batch_n,
        persistent_workers=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_set,
        num_workers=n_workers,
        batch_size=batch_n,
        persistent_workers=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        num_workers=n_workers,
        batch_size=batch_n,
        persistent_workers=True,
    )

    return (
        full_dataset,
        train_set,
        valid_set,
        test_set,
        train_loader,
        valid_loader,
        test_loader,
    )
