# -*- coding: utf-8 -*-
__all__ = ('Candidate',)

from typing import Dict, List, Tuple
import typing

from bugzoo.core.patch import Patch
import attr

from .core import Replacement, FileLine
from .transformation import Transformation
from .util import tuple_from_iterable

if typing.TYPE_CHECKING:
    from .problem import Problem


@attr.s(frozen=True, repr=False, slots=True, auto_attribs=True)
class Candidate:
    """Represents a repair as a set of atomic program transformations."""
    problem: 'Problem' = attr.ib(hash=False, eq=False)
    transformations: Tuple[Transformation, ...] = \
        attr.ib(converter=tuple_from_iterable)

    def __hash__(self) -> int:
        return hash(self.to_diff())

    def to_diff(self) -> Patch:
        """Transforms this candidate patch into a concrete, unified diff."""
        replacements = \
            map(lambda t: t.to_replacement(), self.transformations)
        replacements_by_file: Dict[str, List[Replacement]] = {}
        for rep in replacements:
            fn = rep.location.filename
            if fn not in replacements_by_file:
                replacements_by_file[fn] = []
            replacements_by_file[fn].append(rep)
        # FIXME order each collection of replacements by location
        return self.problem.sources.replacements_to_diff(replacements_by_file)

    def lines_changed(self) -> List[FileLine]:
        """
        Returns a list of source lines that are changed by this candidate
        patch.
        """
        return [t.line for t in self.transformations]

    @property
    def id(self) -> str:
        """An eight-character hexadecimal identifier for this candidate."""
        hex_hash = hex(abs(hash(self)))
        return hex_hash[2:10]

    def __repr__(self) -> str:
        return "Candidate<#{}>".format(self.id)
