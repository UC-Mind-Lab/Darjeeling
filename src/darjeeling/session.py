# -*- coding: utf-8 -*-
__all__ = ('Session',)

from typing import Any, Dict, Iterator, List, Tuple
import glob
import json
import os
import random

import attr
import kaskara
from bugzoo.core import Patch
from bugzoo import Bug as Snapshot
from sourcelocation import Location, FileLocation, FileLocationRange
from loguru import logger

from .core import Language, TestCoverageMap
from .environment import Environment
from .candidate import Candidate
from .outcome import CandidateOutcome
from .resources import ResourceUsageTracker
from .searcher import Searcher
from .problem import Problem
from .config import Config
from .snippet import (SnippetDatabase, StatementSnippetDatabase,
                      LineSnippetDatabase)
from .localization import Localization
from .events import DarjeelingEventHandler, DarjeelingEventProducer


@attr.s
class Session(DarjeelingEventProducer):
    """Used to manage and inspect an interactive repair session."""
    dir_patches: str = attr.ib()
    searcher: Searcher = attr.ib()
    resources: ResourceUsageTracker = attr.ib()
    _problem: Problem = attr.ib()
    terminate_early: bool = attr.ib(default=True)
    _patches: List[Tuple[Candidate, CandidateOutcome]] = attr.ib(factory=list)

    def __attrs_post_init__(self) -> None:
        DarjeelingEventProducer.__init__(self)

    @staticmethod
    def from_config(environment: Environment, cfg: Config) -> 'Session':
        """Creates a new repair session according to a given configuration."""
        logger.debug('preparing patch directory')
        dir_patches = cfg.dir_patches
        if os.path.exists(dir_patches):
            logger.warning("clearing existing patch directory")
            for fn in glob.glob(f'{dir_patches}/*.diff'):
                if os.path.isfile(fn):
                    os.remove(fn)
        logger.debug('prepared patch directory')

        # ensure that Kaskara is installed
        logger.info('ensuring that kaskara installation is complete '
                    '(this may take 20 minutes if Kaskara is not up-to-date)')
        kaskara.post_install()
        logger.info('ensured that kaskara installation is complete')

        # seed the RNG
        # FIXME use separate RNG for each session
        random.seed(cfg.seed)

        logger.info(f"using {cfg.threads} threads")
        logger.info(f"using language: {cfg.program.language.value}")
        logger.info(f"using optimizations: {cfg.optimizations}")
        logger.info(f"using coverage config: {cfg.coverage}")
        logger.info(f"running redundant tests? {cfg.run_redundant_tests}")
        logger.info(f"allowing partial patches? {cfg.allow_partial_patches}")
        logger.info(f"considering all lines? {cfg.consider_all_lines}")
        logger.info(f"using random number generator seed: {cfg.seed}")

        if not cfg.terminate_early:
            logger.info("search will continue after an acceptable patch has been discovered")
        else:
            logger.info("search will terminate when an acceptable patch has been discovered")

        # create the resource tracker
        resources = ResourceUsageTracker.with_limits(cfg.resource_limits)
        logger.info(str(cfg.resource_limits))

        # build program
        logger.debug("building program...")
        program = cfg.program.build(environment)
        logger.debug(f"built program: {program}")

        # compute coverage
        logger.info("computing coverage information...")
        coverage = cfg.coverage.build(environment, program)
        logger.info("computed coverage information")
        logger.debug(f"coverage: {coverage}")

        # compute localization
        logger.info("computing fault localization...")
        localization = \
            Localization.from_config(coverage, cfg.localization)
        logger.info(f"computed fault localization:\n{localization}")

        # determine implicated files
        files = localization.files

        if program.language in (Language.CPP, Language.C):
            kaskara_project = kaskara.Project(dockerblade=environment.dockerblade,
                                              image=program.image,
                                              directory=program.source_directory,
                                              files=files)
            analyser = kaskara.clang.ClangAnalyser()
            analysis = analyser.analyse(kaskara_project)
        elif program.language == Language.PYTHON:
            kaskara_project = kaskara.Project(dockerblade=environment.dockerblade,
                                              image=program.image,
                                              directory=program.source_directory,
                                              files=files)
            analyser = kaskara.python.PythonAnalyser()
            analysis = analyser.analyse(kaskara_project)
        else:
            analysis = None

        # build problem
        problem = Problem.build(environment=environment,
                                config=cfg,
                                language=program.language,
                                program=program,
                                coverage=coverage,
                                analysis=analysis,
                                localization=localization)

        logger.info("constructing database of donor snippets...")
        snippets: SnippetDatabase
        if analysis is not None:
            snippets = StatementSnippetDatabase.from_kaskara(analysis, cfg)
        else:
            snippets = LineSnippetDatabase.for_problem(problem)
        logger.info(f"constructed database of donor snippets: {len(snippets)} snippets")

        transformations = cfg.transformations.build(problem, snippets, cfg.consider_all_lines)
        searcher = cfg.search.build(problem,
                                    resources=resources,
                                    transformations=transformations,
                                    threads=cfg.threads,
                                    run_redundant_tests=cfg.run_redundant_tests,
                                    allow_partial_patches=cfg.allow_partial_patches)

        # build session
        return Session(dir_patches=dir_patches,
                       resources=resources,
                       problem=problem,
                       searcher=searcher,
                       terminate_early=cfg.terminate_early)

    @property
    def snapshot(self) -> Snapshot:
        """The snapshot for the program being repaired."""
        return self.searcher.problem.bug

    @property
    def problem(self) -> Problem:
        """The repair problem that is being solved in this session."""
        return self.searcher.problem

    @property
    def coverage(self) -> TestCoverageMap:
        """The test suite coverage for the program under repair."""
        return self.problem.coverage

    def attach_handler(self, handler: DarjeelingEventHandler) -> None:
        super().attach_handler(handler)
        self.searcher.attach_handler(handler)

    def remove_handler(self, handler: DarjeelingEventHandler) -> None:
        super().remove_handler(handler)
        self.searcher.remove_handler(handler)

    def run(self) -> None:
        logger.info("beginning search process...")
        if self.terminate_early:
            try:
                self._patches.append(next(self.searcher.__iter__()))
            except StopIteration:
                pass
        else:
            self._patches = list(self.searcher)
        if not self._patches:
            logger.info("failed to find a patch")

    @property
    def has_found_patch(self) -> bool:
        """Returns :code:`True` if an acceptable patch has been found."""
        return len(self._patches) > 0

    @property
    def patches(self) -> Iterator[Patch]:
        """Returns an iterator over the patches found during this session."""
        for candidate, outcome in self._patches:
            yield candidate.to_diff()

    def close(self) -> None:
        """Closes the session."""
        # wait for threads to finish gracefully before exiting
        self.searcher.close()

        time_running_mins = self.resources.wall_clock.duration / 60
        logger.info(f"found {len(self._patches)} plausible patches")
        logger.info(f"time taken: {time_running_mins:.2f} minutes")
        logger.info(f"# test evaluations: {self.resources.tests}")
        logger.info(f"# candidate evaluations: {self.resources.candidates}")

        self._save_patches_to_disk()

    def pause(self) -> None:
        """Pauses the session."""
        raise NotImplementedError

    def _save_patches_to_disk(self) -> None:
        logger.debug("saving patches to disk...")
        os.makedirs(self.dir_patches, exist_ok=True)

        for i, (patch, outcome) in enumerate(self._patches):
            fn_patch = os.path.join(self.dir_patches, f'{i}.diff')
            logger.debug(f"writing patch to {fn_patch}")
            try:
                with open(fn_patch, 'w') as f:
                    f.write(str(patch.to_diff()))
            except OSError:
                logger.exception(f"failed to write patch: {fn_patch}")
                raise
            logger.debug(f"wrote patch to {fn_patch}")

            fn_outcome = os.path.join(self.dir_patches, f'{i}_outcomes.json')
            logger.debug(f"writing test outcomes to {fn_outcome}")
            try:
                with open(fn_outcome, 'w') as f:
                    json.dump(outcome.to_dict(), f)
            except OSError:
                logger.exception(f"failed to write test outcomes:"
                                 f"{fn_outcome}")
                raise
            logger.debug(f"wrote test outcomes to {fn_outcome}")

            fn_location = os.path.join(self.dir_patches, f'{i}_location.json')
            logger.debug(f"writing patch location information to {fn_location}")
            try:
                with open(fn_location, 'w') as f:
                    json.dump(
                            self.get_patched_function_location_information(
                                patch.to_diff()),
                            f)
            except OSError:
                logger.exception(f"failed to write patch location information:"
                                 f"{fn_location}")
                raise
            logger.debug(f"wrote patch location information to {fn_location}")

        logger.debug("saved patches to disk")

    def __enter__(self) -> 'Session':
        self.run()
        return self

    @staticmethod
    def file_location_to_string(location: FileLocation) -> str:
        return f"{location.filename}:"\
               f"{location.location.line}:{location.location.column}"

    @staticmethod
    def file_location_range_to_string(location: FileLocationRange)\
            -> str:
        return f"{location.filename}:"\
               f"{location.start.line}:{location.start.column}::"\
               f"{location.stop.line}:{location.stop.column}"

    @staticmethod
    def get_patch_insertion_points(patch: Patch) -> List[Patch]:
        patch_insertion_points = []
        for file_patch in patch._Patch__file_patches:
            for hunk in file_patch._FilePatch__hunks:
                patch_insertion_points.append(
                        FileLocation(
                            file_patch._FilePatch__old_fn,
                            Location(hunk._Hunk__old_start_at,1)))
        return patch_insertion_points

    def get_patched_function_location_information(self, patch: Patch) ->\
            Dict[str, Dict[str, Any]]:
        if self.problem.analysis is None:
            return {}
        # Function names of patch
        locations_to_functions = {}
        for loc in self.get_patch_insertion_points(patch):
            function = self.problem.analysis.functions.\
                    encloses(loc)
            locations_to_functions[self.file_location_to_string(loc)] = {
                    'name': function.name,
                    'location': self.file_location_range_to_string(
                        function.location),
                    'body_location': self.file_location_range_to_string(
                        function.body_location),
                    'return_type': function.return_type,
                    'is_global': function.is_global,
                    'is_pure': function.is_pure
                    }
        return locations_to_functions
