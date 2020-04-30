"""
:py:class:`Model` is an abstract class representing
an reclib model.
"""
import logging

import torch

from reclib.common.registrable import Registrable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# When training a model, many sets of weights are saved. By default we want to
# save/load this set of weights.
_DEFAULT_WEIGHTS = "best.th"


class Model(torch.nn.Module, Registrable):
    """
    This abstract class represents a model to be trained. Rather than relying completely
    on the Pytorch Module, we modify the output spec of ``forward`` to be a dictionary.

    Models built using this API are still compatible with other pytorch models and can
    be used naturally as modules within other models - outputs are dictionaries, which
    can be unpacked and passed into other layers. One caveat to this is that if you
    wish to use an reclib model inside a Container (such as nn.Sequential), you must
    interleave the models with a wrapper module which unpacks the dictionary into
    a list of tensors.

    In order for your model to be trained using the :class:`~reclib.training.Trainer`
    api, the output dictionary of your Model must include a "loss" key, which will be
    optimised during the training process.

    Finally, you can optionally implement :func:`Model.get_metrics` in order to make use
    of early stopping and best-model serialization based on a validation metric in
    :class:`~reclib.training.Trainer`. Metrics that begin with "_" will not be logged
    to the progress bar by :class:`~reclib.training.Trainer`.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *inputs):
        raise NotImplementedError
