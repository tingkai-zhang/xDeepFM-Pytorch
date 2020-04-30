from typing import Dict, Generic, TypeVar

import torch

DataArray = TypeVar("DataArray", torch.Tensor, Dict[str, torch.Tensor])


class Field(Generic[DataArray]):
    """
    A ``Field`` is some piece of a data instance that ends up as an tensor in a model (either as an
    input or an output).  Data instances are just collections of fields.
    The reason have a ``Field`` class is, there could be multiple types of input data: text, image, ids
    """

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented
