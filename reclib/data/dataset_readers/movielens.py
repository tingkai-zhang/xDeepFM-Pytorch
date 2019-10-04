import numpy as np
import pandas as pd
import torch.utils.data


class MovieLens20MDataset(torch.utils.data.Dataset):
    """
    MovieLens 20M Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    Parameters
    ----------
    dataset_path: ``string``
        MovieLens dataset path

    Attributes
    ----------
    items : ``array``
        MovieLens raw feature columns(fields), discrete values
    targets:  ``array``
        processed binary targets. 0, 1
    field_sizes: ``List``
        The max numbers of each column, defining the size of each field

    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path, sep=','):
        data = pd.read_csv(dataset_path, sep=sep,
                           engine='python').to_numpy()[:, :3]
        # ID starts from 1
        self.items = data[:, :2].astype(np.int) - 1
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        self.field_sizes = np.max(self.items, axis=0) + 1

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target


class MovieLens1MDataset(MovieLens20MDataset):
    """
    MovieLens 1M Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path, '::')
