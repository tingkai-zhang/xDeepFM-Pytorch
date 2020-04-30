import logging
from typing import Dict, Union

import torch
from torch.utils.data import DataLoader

from reclib.data.dataset import Batch
from reclib.data.fields import MetadataField

logger = logging.getLogger(__name__)

TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]


def add_epoch_number(batch: Batch, epoch: int) -> Batch:
    """
    Add the epoch number to the batch instances as a MetadataField.
    """
    for instance in batch.instances:
        instance.fields["epoch_num"] = MetadataField(epoch)
    return batch


class DataIterator(DataLoader):
    def __init__(self):
        pass
