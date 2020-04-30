from typing import Dict, MutableMapping, Mapping

from reclib.data.fields.field import Field


class Instance(Mapping[str, Field]):
    """
    An ``Instance`` is a collection of :class:`~reclib.data.fields.field.Field` objects,
    specifying the inputs and outputs to some model.
    We don't make a distinction between inputs and outputs here, though - all
    operations are done on all fields, and when we return arrays, we return them as dictionaries
    keyed by field name.  A model can then decide which fields it wants to use as inputs as which
    as outputs.

    Parameters
    ----------
    fields : ``Dict[str, Field]``
        The ``Field`` objects that will be used to produce data arrays for this instance.
    """

    def __init__(self, fields: MutableMapping[str, Field]) -> None:
        self.fields = fields
        self.indexed = False

    # Add methods for ``Mapping``.  Note, even though the fields are
    # mutable, we don't implement ``MutableMapping`` because we want
    # you to use ``add_field`` and supply a vocabulary.
    def __getitem__(self, key: str) -> Field:
        return self.fields[key]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def add_field(self, field_name: str, field: Field) -> None:
        """
        Add the field to the existing fields mapping.
        If we have already indexed the Instance, then we also index `field`, so
        it is necessary to supply the vocab.
        """
        self.fields[field_name] = field

    def __str__(self) -> str:
        base_string = f"Instance with fields:\n"
        return " ".join(
            [base_string] + [f"\t {name}: {field} \n" for name, field in self.fields.items()]
        )
