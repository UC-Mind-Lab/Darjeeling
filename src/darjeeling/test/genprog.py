# -*- coding: utf-8 -*-
__all__ = ('GenProgTest', 'GenProgTestSuite', 'GenProgTestSuiteConfig')

from typing import Optional, Sequence, Dict, Any
import os
import typing

import attr

from .. import exceptions as exc
from .base import TestSuite
from .config import TestSuiteConfig
from ..core import TestOutcome, Test

if typing.TYPE_CHECKING:
    from ..container import ProgramContainer
    from ..environment import Environment


@attr.s(frozen=True, slots=True, auto_attribs=True)
class GenProgTest(Test):
    name: str
    target: Optional[str]
    timelimit: int
    allow_timeouts: bool


@attr.s(frozen=True, slots=True, auto_attribs=True)
class GenProgTestSuiteConfig(TestSuiteConfig):
    NAME = 'genprog'
    workdir: str
    number_failing_tests: int
    number_passing_tests: int
    time_limit_seconds: int
    target: Optional[str]
    allow_timeouts: bool

    @classmethod
    def from_dict(cls,
                  d: Dict[str, Any],
                  dir_: Optional[str] = None
                  ) -> TestSuiteConfig:
        # FIXME if no workdir is specified, use source directory specified
        # by program configuration
        workdir = d['workdir']
        number_failing_tests: int = d['number-of-failing-tests']
        number_passing_tests: int = d['number-of-passing-tests']
        target: Optional[str] = d.get('target', None)
        allow_timeouts: bool = d.get('allow-timeouts', False)

        if not os.path.isabs(workdir):
            m = "'workdir' property must be an absolute path"
            raise exc.BadConfigurationException(m)

        if 'time-limit' not in d:
            time_limit_seconds = 300
        else:
            time_limit_seconds = d['time-limit']

        if allow_timeouts:
            # To allow the test script to mark timeouts as intended we
            # must have the test script catch the time out before the
            # container's run command, which we can do be by having the
            # test script run for one second less than the specified
            # time_limit_seconds
            time_limit_seconds += 1

        return GenProgTestSuiteConfig(workdir=workdir,
                                      number_failing_tests=number_failing_tests,
                                      number_passing_tests=number_passing_tests,
                                      time_limit_seconds=time_limit_seconds,
                                      target=target,
                                      allow_timeouts=allow_timeouts)

    def build(self, environment: 'Environment') -> 'TestSuite':
        failing_test_numbers = range(1, self.number_failing_tests + 1)
        passing_test_numbers = range(1, self.number_passing_tests + 1)
        failing_test_names = [f'n{i}' for i in failing_test_numbers]
        passing_test_names = [f'p{i}' for i in passing_test_numbers]
        failing_tests = tuple(GenProgTest(name, self.target,
                                          self.time_limit_seconds - 1,
                                          self.allow_timeouts) for name in
                              failing_test_names)
        passing_tests = tuple(GenProgTest(name, self.target,
                                          self.time_limit_seconds - 1,
                                          self.allow_timeouts) for name in
                              passing_test_names)
        tests = failing_tests + passing_tests
        return GenProgTestSuite(environment=environment,
                                tests=tests,
                                workdir=self.workdir,
                                time_limit_seconds=self.time_limit_seconds)


class GenProgTestSuite(TestSuite[GenProgTest]):
    def __init__(self,
                 environment: 'Environment',
                 tests: Sequence[GenProgTest],
                 workdir: str,
                 time_limit_seconds: int
                 ) -> None:
        super().__init__(environment, tests)
        self._workdir = workdir
        self._time_limit_seconds = time_limit_seconds

    def execute(self,
                container: 'ProgramContainer',
                test: GenProgTest,
                *,
                coverage: bool = False
                ) -> TestOutcome:
        args = ""
        if test.target is not None:
            args += f" --target {test.target} "

        if test.allow_timeouts:
            args += f" --allow_timeouts {test.timelimit} "

        command = f'./test.sh {args} {test.name}'
        print(command)
        outcome = container.shell.run(command,
                                      cwd=self._workdir,
                                      time_limit=self._time_limit_seconds)  # noqa
        successful = outcome.returncode == 0

        output_path = os.path.join(self._workdir, "test_output")
        if container.filesystem.isfile(output_path):
            output: Optional[str] = container.filesystem.read(output_path)
        else:
            output = None

        return TestOutcome(test.name, successful=successful,
                           time_taken=outcome.duration, output=output)
