import argparse
import logging
from typing import Dict

from overrides import overrides
from reclib.commands.configure import Configure
from reclib.commands.dry_run import DryRun
from reclib.commands.elmo import Elmo
from reclib.commands.evaluate import Evaluate
from reclib.commands.find_learning_rate import FindLearningRate
from reclib.commands.fine_tune import FineTune
from reclib.commands.make_vocab import MakeVocab
from reclib.commands.print_results import PrintResults
from reclib.commands.test_install import TestInstall

from reclib import __version__
from reclib.commands.predict import Predict
from reclib.commands.subcommand import Subcommand
from reclib.commands.train import Train
from reclib.common.util import import_submodules

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ArgumentParserWithDefaults(argparse.ArgumentParser):
    """
    Custom argument parser that will display the default value for an argument
    in the help message.
    """

    _action_defaults_to_ignore = {"help", "store_true", "store_false", "store_const"}

    @staticmethod
    def _is_empty_default(default):
        if default is None:
            return True
        if isinstance(default, (str, list, tuple, set)):
            return not bool(default)
        return False

    @overrides
    def add_argument(self, *args, **kwargs):
        # pylint: disable=arguments-differ
        # Add default value to the help message when the default is meaningful.
        default = kwargs.get("default")
        if kwargs.get("action") not in self._action_defaults_to_ignore and not self._is_empty_default(default):
            description = kwargs.get("help") or ""
            kwargs["help"] = f"{description} (default = {default})"
        super().add_argument(*args, **kwargs)


def main(prog: str = None,
         subcommand_overrides: Dict[str, Subcommand] = {}) -> None:
    """
    The :mod:`~reclib.run` command only knows about the registered classes in the ``reclib``
    codebase. In particular, once you start creating your own ``Model`` s and so forth, it won't
    work for them, unless you use the ``--include-package`` flag.
    """
    # pylint: disable=dangerous-default-value
    parser = ArgumentParserWithDefaults(description="Run RecLib", usage='%(prog)s', prog=prog)
    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)

    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
        # Default commands
        "configure": Configure(),
        "train": Train(),
        "evaluate": Evaluate(),
        "predict": Predict(),
        "make-vocab": MakeVocab(),
        "elmo": Elmo(),
        "fine-tune": FineTune(),
        "dry-run": DryRun(),
        "test-install": TestInstall(),
        "find-lr": FindLearningRate(),
        "print-results": PrintResults(),
        # Superseded by overrides
        **subcommand_overrides
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        # configure doesn't need include-package because it imports
        # whatever classes it needs.
        if name != "configure":
            subparser.add_argument('--include-package',
                                   type=str,
                                   action='append',
                                   default=[],
                                   help='additional packages to include')

    args = parser.parse_args()

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in getattr(args, 'include_package', ()):
            import_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()
