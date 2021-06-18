# -*- coding: utf-8 -*-
__all__ = ('SearcherConfig',)

from typing import Dict, Optional, Any, Type, Iterator
import abc
import typing

from ..util import dynamically_registered
from ..outcome import CandidateOutcomeStore

if typing.TYPE_CHECKING:
    from .base import Searcher
    from ..problem import Problem
    from ..resources import ResourceUsageTracker
    from ..transformation import ProgramTransformations


@dynamically_registered(lookup='lookup')
class SearcherConfig(abc.ABC):
    """Describes a search algorithm configuration."""
    @staticmethod
    def __iter__() -> Iterator[str]:
        ...

    @staticmethod
    def __len__() -> int:
        ...

    @staticmethod
    def lookup(name: str) -> Type['SearcherConfig']:
        ...

    @classmethod
    @abc.abstractmethod
    def from_dict(cls,
                  d: Dict[str, Any],
                  dir_: Optional[str] = None
                  ) -> 'SearcherConfig':
        name_type: str = d['type']
        type_: Type[SearcherConfig] = SearcherConfig.lookup(name_type)
        return type_.from_dict(d, dir_)

    @abc.abstractmethod
    def build(self,
              problem: 'Problem',
              resources: 'ResourceUsageTracker',
              transformations: 'ProgramTransformations',
              *,
              outcomes: Optional[CandidateOutcomeStore],
              threads: int = 1,
              run_redundant_tests: bool = False,
              allow_partial_patches: bool = False
              ) -> 'Searcher':
        ...
