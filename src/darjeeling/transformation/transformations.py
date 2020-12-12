# -*- coding: utf-8 -*-
__all__ = ('ProgramTransformations',)

from typing import Collection, Iterator
import typing

from loguru import logger
import attr

from .base import Transformation, TransformationSchema
from .index import TransformationIndex
from ..snippet import SnippetDatabase
from ..localization import Localization

if typing.TYPE_CHECKING:
    from ..problem import Problem


@attr.s
class ProgramTransformations:
    """Represents the space of possible transformations to a given program.

    Attributes
    ----------
    schemas: Collection[TransformationSchema]
        The schemas that compose the space of transformations.
    """
    schemas: Collection[TransformationSchema] = attr.ib()
    _index: TransformationIndex = \
        attr.ib(repr=False, eq=False, hash=False)

    @classmethod
    def build(cls,
              schemas: Collection[TransformationSchema],
              problem: 'Problem',
              snippets: SnippetDatabase,
              localization: Localization,
              consider_all_lines: bool
              ) -> 'ProgramTransformations':
        logger.debug("generating program transformations")

        if not consider_all_lines:
            # Only the lines that have been implicated
            lines = list(localization)
        else:
            # All lines, regardless of implication
            lines = list(problem.sources.filelines())

        index = TransformationIndex.build(schemas, problem, snippets, lines)
        logger.debug("generated program transformations")
        return ProgramTransformations(schemas, index)

    def __len__(self) -> int:
        """Returns a count of the number of possible transformations."""
        return len(self._index)

    def __iter__(self) -> Iterator[Transformation]:
        """Returns an iterator over all possible transformations."""
        yield from self._index

    def choice(self) -> Transformation:
        """Selects a single transformation at random"""
        return self._index.choice()
