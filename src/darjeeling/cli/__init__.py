# -*- coding: utf-8 -*-
from bugzoo.core.patch import Patch
from collections import namedtuple
from typing import Optional, Sequence
import functools
import glob
import json
from kaskara.clang.analysis import ClangStatement
import os
from sourcelocation import Location, FileLocation, FileLocationRange
import sys


from loguru import logger
import attr
import bugzoo
import bugzoo.server
import cement
import pyroglyph
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import yaml

from ..environment import Environment
from ..problem import Problem
from ..version import __version__ as VERSION
from ..config import Config
from ..core import TestCoverageMap
from ..events import CsvEventLogger, WebSocketEventHandler
from ..plugins import LOADED_PLUGINS
from ..resources import ResourceUsageTracker
from ..session import Session
from ..exceptions import BadConfigurationException
from ..util import duration_str

BANNER = 'DARJEELING'


@attr.s(auto_attribs=True)
class ResourcesBlock(pyroglyph.Block):
    resources: ResourceUsageTracker

    @staticmethod
    def for_session(session: Session) -> 'ResourcesBlock':
        return ResourcesBlock(session.resources)

    @property
    def title(self) -> str:
        return 'Resources Used'

    @property
    def contents(self) -> Sequence[str]:
        duration_seconds = self.resources.wall_clock.duration
        l_time = f'Running Time: {duration_str(duration_seconds)}'
        l_candidates = f'Num. Candidates: {self.resources.candidates}'
        l_tests = f'Num. Tests: {self.resources.tests}'
        l_patches = 'Num. Acceptable Patches: TODO'
        return [l_time, l_candidates, l_tests, l_patches]


class ProblemBlock(pyroglyph.BasicBlock):
    def __init__(self, problem: Problem) -> None:
        title = f'Problem [{problem.bug.name}]'
        num_failing = len(list(problem.failing_tests))
        num_passing = len(list(problem.passing_tests))
        num_lines = len(list(problem.lines))
        num_files = len(list(problem.implicated_files))
        contents = [
            f'Passing Tests: {num_passing}',
            f'Failing Tests: {num_failing}',
            f'Implicated Lines: {num_lines} ({num_files} files)'
        ]
        super().__init__(title, contents)


class UI(pyroglyph.Window):
    def __init__(self, session: Session, **kwargs) -> None:
        title = f' Darjeeling [v{VERSION}] '
        blocks_left = [ResourcesBlock.for_session(session)]
        blocks_right = [ProblemBlock(session.problem)]
        super().__init__(title, blocks_left, blocks_right, **kwargs)


class BaseController(cement.Controller):
    class Meta:
        label = 'base'
        description = 'Language-independent automated program repair'
        arguments = [
            (['--version'], {'action': 'version', 'version': BANNER}),
        ]

    def default(self):
        # type: () -> None
        self.app.args.print_help()

    @property
    def _default_log_filename(self) -> str:
        # find all log file numbers that have been used in this directory
        used_numbers = [int(s.rpartition('.')[-1])
                        for s in glob.glob('darjeeling.log.*')]

        if not used_numbers:
            return os.path.join(os.getcwd(), 'darjeeling.log.0')

        num = max(used_numbers) + 1
        return os.path.join(os.getcwd(), 'darjeeling.log.{}'.format(num))

    @cement.ex(
        help='generates a test suite coverage report for a given problem',
        arguments=[
            (['filename'],
             {'help': ('a Darjeeling configuration file describing a faulty '
                       'program and how it should be repaired.')}),
            (['--format'],
             {'help': 'the format that should be used for the coverage report',
              'default': 'text',
              'choices': ('text', 'yaml', 'json')})
        ]
    )
    def coverage(self) -> None:
        """Generates a coverage report for a given program."""
        # load the configuration file
        filename = self.app.pargs.filename
        filename = os.path.abspath(filename)
        cfg_dir = os.path.dirname(filename)
        with open(filename, 'r') as f:
            yml = yaml.safe_load(f)
        cfg = Config.from_yml(yml, dir_=cfg_dir)

        with bugzoo.server.ephemeral(timeout_connection=120) as client_bugzoo:
            environment = Environment(bugzoo=client_bugzoo)
            try:
                session = Session.from_config(environment, cfg)
            except BadConfigurationException as exp:
                print(f"ERROR: bad configuration file:\n{exp}")
                sys.exit(1)

            coverage = session.coverage
            formatter = ({
                'text': lambda c: str(c),
                'yaml': lambda c: yaml.safe_dump(c.to_dict(), default_flow_style=False),
                'json': lambda c: json.dumps(c.to_dict(), indent=2)
            })[self.app.pargs.format]
            print(formatter(coverage))

    @cement.ex(
        help='attempt to automatically repair a given program',
        arguments=[
            (['filename'],
             {'help': ('a Darjeeling configuration file describing the faulty '
                       'program and how it should be repaired.')}),
            (['--interactive'],
             {'help': 'enables an interactive user interface.',
              'action': 'store_true'}),
            (['--silent'],
             {'help': 'prevents output to the stdout',
              'action': 'store_true'}),
            (['--log-events-to-file'],
             {'help': 'path of the CSV file to which events should be logged.',
              'type': str}),
            (['--print-patch'],
             {'help': 'prints the first acceptable patch that was found',
              'action': 'store_true'}),
            (['--log-to-file'],
             {'help': 'path to store the log file.',
              'type': str}),
            (['--no-log-to-file'],
             {'help': 'disables logging to file.',
              'action': 'store_true'}),
            (['--patch-dir'],
             {'help': 'path to store the patches.',
              'dest': 'dir_patches',
              'type': str}),
            (['-v', '--verbose'],
             {'help': 'enables verbose DEBUG-level logging to the stdout',
              'action': 'store_true'}),
            (['--web'],
             {'help': 'enables a web interface',
              'action': 'store_true'}),
            (['--seed'],
             {'help': 'random number generator seed',
              'type': int}),
            (['--max-candidates'],
             {'dest': 'limit_candidates',
              'type': int,
              'help': ('the maximum number of candidate patches that may be '
                       'considered by the search.')}),
            (['--max-time-mins'],
             {'dest': 'limit_time_minutes',
              'type': int,
              'help': ('the maximum number of minutes that may be spent '
                       'searching for a patch.')}),
            (['--continue'],
             {'dest': 'terminate_early',
              'action': 'store_false',
              'help': ('continue to search for patches after an acceptable '
                       ' patch has been discovered.')}),
            (['--threads'],
             {'dest': 'threads',
              'type': int,
              'help': ('number of threads over which the repair workload '
                       'should be distributed')})
        ]
    )
    def repair(self) -> bool:
        """Performs repair on a given scenario.

        Returns
        -------
        bool
            :code:`True` if at least one patch was found, else :code:`False`.
        """
        filename: str = self.app.pargs.filename
        interactive: bool = self.app.pargs.interactive
        seed: Optional[int] = self.app.pargs.seed
        terminate_early: bool = self.app.pargs.terminate_early
        threads: Optional[int] = self.app.pargs.threads
        limit_candidates: Optional[int] = \
            self.app.pargs.limit_candidates
        limit_time_minutes: Optional[int] = \
            self.app.pargs.limit_time_minutes
        dir_patches: Optional[str] = self.app.pargs.dir_patches
        log_to_filename: Optional[str] = self.app.pargs.log_to_file
        should_log_to_file: bool = not self.app.pargs.no_log_to_file
        verbose_logging: bool = self.app.pargs.verbose

        # remove all existing loggers
        logger.remove()
        logger.enable('darjeeling')
        for plugin_name in LOADED_PLUGINS:
            logger.enable(plugin_name)

        # log to stdout, unless instructed not to do so
        if not self.app.pargs.silent:
            if interactive:
                stdout_logging_level = 'CRITICAL'
            elif verbose_logging:
                stdout_logging_level = 'DEBUG'
            else:
                stdout_logging_level = 'INFO'
            logger.add(sys.stdout, level=stdout_logging_level)

        # setup logging to file
        if should_log_to_file:
            if not log_to_filename:
                log_to_filename = self._default_log_filename
            logger.info(f'logging to file: {log_to_filename}')
            logger.add(log_to_filename, level='DEBUG')

        # load the configuration file
        filename = os.path.abspath(filename)
        cfg_dir = os.path.dirname(filename)
        with open(filename, 'r') as f:
            yml = yaml.safe_load(f)
        cfg = Config.from_yml(yml,
                              dir_=cfg_dir,
                              threads=threads,
                              seed=seed,
                              terminate_early=terminate_early,
                              limit_candidates=limit_candidates,
                              limit_time_minutes=limit_time_minutes,
                              dir_patches=dir_patches)
        logger.info(f"using configuration: {cfg}")

        # connect to BugZoo
        with Environment() as environment:
            try:
                session = Session.from_config(environment, cfg)
            except BadConfigurationException as err:
                logger.error(str(err))
                sys.exit(1)

            # create and attach handlers
            if self.app.pargs.log_events_to_file:
                csv_logger_fn = self.app.pargs.log_events_to_file
                if not os.path.isabs(csv_logger_fn):
                    csv_logger_fn = os.path.join(os.getcwd(), csv_logger_fn)
                csv_logger = CsvEventLogger(csv_logger_fn,
                                            session._problem)
                session.attach_handler(csv_logger)

            # add optional websocket handler
            if self.app.pargs.web:
                websocket_handler = WebSocketEventHandler()
                session.attach_handler(websocket_handler)

            if interactive:
                with UI(session):
                    session.run()
                    session.close()

            if not interactive:
                session.run()
                session.close()

            if self.app.pargs.print_patch and session.has_found_patch:
                first_patch = next(session.patches)
                print(str(first_patch))

            if session.has_found_patch:
                sys.exit(0)
            else:
                sys.exit(1)

    @cement.ex(
        help='Gives suggestions of groups of test cases to invalidate',
        arguments=[
            (['filename'],
             {'help': ('a Darjeeling configuration file describing a '
                       'non-faulty program and how it should be tested.')}),
            (['--format'],
             {'help': 'the format that should be used for the coverage report',
              'default': 'text',
              'choices': ('text', 'yaml', 'json')}),
            (['--linkage'],
             {'help': 'linkage method to use',
              'default': 'single',
              'choices': ('single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward')})
        ]
    )
    def negations(self) -> None:
        """Generates suggestions of test cases to negate for a given program."""
        # load the configuration file
        filename = self.app.pargs.filename
        filename = os.path.abspath(filename)
        cfg_dir = os.path.dirname(filename)
        with open(filename, 'r') as f:
            yml = yaml.safe_load(f)
        cfg = Config.from_yml(yml, dir_=cfg_dir)

        with bugzoo.server.ephemeral(timeout_connection=120) as client_bugzoo:
            environment = Environment(bugzoo=client_bugzoo)
            try:
                session = Session.from_config(environment, cfg)
            except BadConfigurationException as exp:
                print(f"ERROR: bad configuration file:\n{exp}")
                sys.exit(1)

            coverage = session.coverage

            # Clustering stuff
            def test_name_pair(name1, name2):
                if name1 < name2:
                    return name1, name2
                else:
                    return name2, name1

            def test_pairs(coverage: TestCoverageMap):
                pairs = set()
                for tc1 in coverage:
                    for tc2 in coverage:
                        pairs.add(test_name_pair(tc1, tc2))
                return pairs

            def test_observations(coverage: TestCoverageMap):
                return [[t] for t in coverage]

            def jaccard_indices(coverage: TestCoverageMap):
                indices = dict()
                for tc1Name, tc2Name in test_pairs(coverage):
                    tc1Lines = coverage[tc1Name].lines
                    tc2Lines = coverage[tc2Name].lines
                    intersection = tc1Lines.intersection(tc2Lines)
                    union = tc1Lines.union(tc2Lines)
                    jaccardIndex = len(intersection) / len(union)
                    indices[(tc1Name, tc2Name)] = jaccardIndex
                return indices

            indices = jaccard_indices(coverage)

            obs = test_observations(coverage)

            def pair_wise_distance(u, v, indices):
                # Return the jaccard distance
                return 1 - indices[test_name_pair(u[0], v[0])]

            p_hack = functools.partial(pair_wise_distance, indices=indices)
            processed_pair_wise_distance = pdist(obs, p_hack)

            def cluster_deconstructions(observations, calc_linkage):
                ClusterNode = namedtuple("ClusterNode", ["cluster1", "cluster2", "distance"])

                def flatten_cluster_node(cl_node):
                    def help(cl):
                        if isinstance(cl, list):
                            yield cl[0]
                        else:
                            for c in help(cl[0]):
                                yield c
                            for c in help(cl[1]):
                                yield c
                    return list(sorted(help(cl_node)))

                # Build the clusters
                for record in calc_linkage:
                    observations.append(ClusterNode(
                        observations[int(record[0])],
                        observations[int(record[1])],
                        record[2]
                    ))

                to_check = [observations[-1]]

                while len(to_check) > 0:
                    temp = to_check.pop()
                    if isinstance(temp, list):
                        yield flatten_cluster_node(temp)
                    else:
                        if temp[2] > 0:
                            t0 = flatten_cluster_node(temp[0])
                            t1 = flatten_cluster_node(temp[1])
                            if len(t0) < len(t1):
                                yield t0
                                yield t1
                                if isinstance(temp[0], ClusterNode):
                                    to_check.append(temp[0])
                                if isinstance(temp[1], ClusterNode):
                                    to_check.append(temp[1])
                            else:
                                yield t1
                                yield t0
                                if isinstance(temp[1], ClusterNode):
                                    to_check.append(temp[1])
                                if isinstance(temp[0], ClusterNode):
                                    to_check.append(temp[0])

            _linkage = hierarchy.linkage(processed_pair_wise_distance,
                                         method=self.app.pargs.linkage)
            out = {
                "linkage": self.app.pargs.linkage,
                "suggestions": list(cluster_deconstructions(obs, _linkage))
            }
            formatter = ({
                'text': lambda c: str(c),
                'yaml': lambda c: yaml.safe_dump(c, default_flow_style=False),
                'json': lambda c: json.dumps(c, indent=2)
            })[self.app.pargs.format]
            print(formatter(out))

    @cement.ex(
        help='Gather information about a patch.',
        arguments=[
            (['filename'],
             {'help': ('a Darjeeling configuration file describing a '
                       'program and how it should be tested.')}),
            (['patch'],
             {'help': ('Patch to load.')}),
            (['--format'],
             {'help': 'the format that should be used for the coverage report',
              'default': 'text',
              'choices': ('text', 'yaml', 'json')})
        ]
    )
    def patch_information(self) -> None:
        """Generates suggestions of test cases to negate for a given program."""
        # load the configuration file
        filename: str = self.app.pargs.filename
        filename = os.path.abspath(filename)
        cfg_dir = os.path.dirname(filename)
        with open(filename, 'r') as f:
            yml = yaml.safe_load(f)
        cfg = Config.from_yml(yml, dir_=cfg_dir)

        # load the patch
        patch_path: str  = self.app.pargs.patch
        with open(patch_path, 'r') as fin:
            patch: Patch = Patch.from_unidiff(fin.read())

        patch_insertion_points = []
        for file_patch in patch._Patch__file_patches:
            for hunk in file_patch._FilePatch__hunks:
                patch_insertion_points.append(
                        FileLocation(
                            file_patch._FilePatch__old_fn,
                            Location(hunk._Hunk__old_start_at,1)))

        with bugzoo.server.ephemeral(timeout_connection=120) as client_bugzoo:
            environment = Environment(bugzoo=client_bugzoo)
            try:
                session = Session.from_config(environment, cfg)
            except BadConfigurationException as exp:
                print(f"ERROR: bad configuration file:\n{exp}")
                sys.exit(1)

            if session.problem.analysis is None:
                print(f"ERROR: No analysis of the problem created")
                sys.exit(1)

            def file_location_to_string(location: FileLocation) -> str:
                return f"{location.filename}:"\
                       f"{location.location.line}:{location.location.column}"

            def file_location_range_to_string(location: FileLocationRange)\
                    -> str:
                return f"{location.filename}:"\
                       f"{location.start.line}:{location.start.column}::"\
                       f"{location.stop.line}:{location.stop.column}"

            # Function names of patch
            locations_to_functions = {}
            for loc in patch_insertion_points:
                function = session.problem.analysis.functions.\
                        encloses(loc)
                locations_to_functions[file_location_to_string(loc)] = {
                        'name': function.name,
                        'location': file_location_range_to_string(
                            function.location),
                        'body_location': file_location_range_to_string(
                            function.body_location),
                        'return_type': function.return_type,
                        'is_global': function.is_global,
                        'is_pure': function.is_pure
                        }

            # Output information
            information = {
                    'locations_to_functions': locations_to_functions
                    }
            formatter = ({
                'text': lambda c: str(c),
                'yaml': lambda c: yaml.safe_dump(c,
                    default_flow_style=False),
                'json': lambda c: json.dumps(c, indent=2)
            })[self.app.pargs.format]
            print(formatter(information))

class CLI(cement.App):
    class Meta:
        label = 'darjeeling'
        catch_signals = None
        handlers = [BaseController]


def main():
    with CLI() as app:
        app.run()
