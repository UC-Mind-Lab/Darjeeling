# -*- coding: utf-8 -*-
"""
This module implements data structures and containers for describing the
outcome of test and build attempts.
"""
__all__ = (
    'TestOutcome',
    'TestOutcomeSet',
    'BuildOutcome',
    'CandidateOutcome',
    'CandidateOutcomeStore'
)

from typing import Any, Dict, Iterator, Mapping, Optional

import attr

from .core import BuildOutcome, TestOutcome, TestOutcomeSet
from .candidate import Candidate


@attr.s(frozen=True, slots=True, auto_attribs=True)
class CandidateOutcome:
    """Records the outcome of a candidate patch evaluation."""
    build: BuildOutcome
    tests: TestOutcomeSet
    is_repair: bool

    def with_test_outcome(self,
                          test: str,
                          successful: bool,
                          output: Optional[str],
                          time_taken: float
                          ) -> 'CandidateOutcome':
        outcome = TestOutcome(test, successful, time_taken, output)
        test_outcomes = self.tests.with_outcome(test, outcome)
        return CandidateOutcome(self.build, test_outcomes, self.is_repair)

    def merge(self,
              other: 'CandidateOutcome'
              ) -> 'CandidateOutcome':
        other_is_repair = all(other.tests[t].successful for t in other.tests)
        return CandidateOutcome(self.build,
                                self.tests.merge(other.tests),
                                self.is_repair and other_is_repair)

    def to_dict(self) -> Dict[str, Any]:
        return {'build': self.build.to_dict(),
                'tests': self.tests.to_dict(),
                'is-repair': self.is_repair}

    @classmethod
    def from_dict(cls, _dict) -> 'CandidateOutcome':
        return cls(
                build=BuildOutcome.from_dict(_dict['build']),
                tests=TestOutcomeSet.from_dict(_dict['tests']),
                is_repair=_dict['is-repair'])


class CandidateOutcomeStore(Mapping[int, CandidateOutcome]):
    """Maintains a record of candidate patch evaluation outcomes."""
    def __init__(self) -> None:
        self.__outcomes: Dict[int, CandidateOutcome] = {}

    def to_dict(self) -> dict:
        _dict = {}
        for candidate_hash in self.__outcomes:
            _dict[candidate_hash] = self.__outcomes[candidate_hash].to_dict()
        return _dict

    @classmethod
    def from_dict(cls, _dict: dict) -> "CandidateOutcomeStore":
        COS = cls()
        for candidate_hash in _dict:
            COS.record(candidate_hash, 
                       CandidateOutcome.from_dict(_dict[candidate_hash]))
        return COS

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __contains__(self, candidate_hash: Any) -> bool:
        if not isinstance(candidate_hash, int):
            return False
        return candidate_hash in self.__outcomes

    def __getitem__(self, candidate_hash: int) -> CandidateOutcome:
        return self.__outcomes[candidate_hash]

    def __iter__(self) -> Iterator[int]:
        yield from self.__outcomes

    def __len__(self) -> int:
        """Returns a count of the number of represented candidate patches."""
        return len(self.__outcomes)

    def record(self,
               candidate: Candidate,
               outcome: CandidateOutcome
               ) -> None:
        if candidate not in self.__outcomes:
            self.__outcomes[hash(candidate)] = outcome
        else:
            self.__outcomes[hash(candidate)] = \
                self.__outcomes[hash(candidate)].merge(outcome)
