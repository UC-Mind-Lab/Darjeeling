# -*- coding: utf-8 -*-
"""
This module implements GenProg-style operators for individual source code
lines.
"""
__all__ = ('InsertLine', 'DeleteLine', 'ReplaceLine')

from typing import Any, Collection, Iterator, Mapping, Optional
import abc
import typing

import attr

from .base import Transformation, TransformationSchema
from .config import TransformationSchemaConfig
from ..snippet import SnippetDatabase, LineSnippetDatabase
from ..core import Replacement, FileLine, FileLocationRange, LocationRange
from ..exceptions import BadConfigurationException

if typing.TYPE_CHECKING:
    from ..problem import Problem


class LineTransformation(Transformation):
    """Base class for all line-based transformations."""


@attr.s(frozen=True, auto_attribs=True)
class LineTransformationSchema(TransformationSchema[LineTransformation]):
    _problem: 'Problem' = attr.ib(hash=False)
    _snippets: LineSnippetDatabase = attr.ib(hash=False)

    def find_all_in_file(self, filename: str) -> Iterator[Transformation]:
        m = "find_all_in_file is not required or supported by this schema"
        raise NotImplementedError(m)

    def find_all_at_lines_in_file(self,
                                  filename: str,
                                  lines: Collection[int]
                                  ) -> Iterator[Transformation]:
        for line_number in lines:
            file_line = FileLine(filename, line_number)
            yield from self.find_all_at_line(file_line)

    @abc.abstractmethod
    def find_all_at_line(self, line: FileLine) -> Iterator[Transformation]:
        ...

    def viable_insertions(self, context: FileLine) -> Iterator[FileLine]:
        sources = self._problem.sources
        filename = context.filename
        for line_num in range(1, sources.num_lines(filename) + 1):
            insertion = FileLine(filename, line_num)
            content = sources.read_line(insertion)
            if content.isspace():
                continue
            yield insertion


@attr.s(frozen=True, auto_attribs=True)
class DeleteLine(LineTransformation):
    _schema: LineTransformationSchema
    line: FileLine

    def to_replacement(self) -> Replacement:
        sources = self._schema._problem.sources
        loc = sources.line_to_location_range(self.line)
        return Replacement(loc, '')

    @property
    def schema(self) -> TransformationSchema:
        return self._schema

    class Schema(LineTransformationSchema):
        def find_all_at_line(self, line: FileLine) -> Iterator[Transformation]:
            yield DeleteLine(self, line)

    class SchemaConfig(TransformationSchemaConfig):
        NAME = 'delete-line'

        @classmethod
        def from_dict(cls,
                      d: Mapping[str, Any],
                      dir_: Optional[str] = None
                      ) -> 'TransformationSchemaConfig':
            return DeleteLine.SchemaConfig()

        def build(self,
                  problem: 'Problem',
                  snippets: SnippetDatabase
                  ) -> 'TransformationSchema':
            assert isinstance(snippets, LineSnippetDatabase)
            return DeleteLine.Schema(problem=problem, snippets=snippets)


@attr.s(frozen=True, auto_attribs=True)
class ReplaceLine(LineTransformation):
    _schema: LineTransformationSchema
    line: FileLine
    replacement: FileLine

    @property
    def schema(self) -> TransformationSchema:
        return self._schema

    def to_replacement(self) -> Replacement:
        sources = self._schema._problem.sources
        loc = sources.line_to_location_range(self.line)
        rep = sources.read_line(self.replacement, keep_newline=True)
        return Replacement(loc, rep)

    @attr.s(frozen=True, auto_attribs=True)
    class Schema(LineTransformationSchema):
        def find_all_at_line(self, line: FileLine) -> Iterator[Transformation]:
            for replacement in self.viable_insertions(line):
                if replacement != line:
                    yield ReplaceLine(self, line, replacement)

    class SchemaConfig(TransformationSchemaConfig):
        NAME = 'replace-line'

        @classmethod
        def from_dict(cls,
                      d: Mapping[str, Any],
                      dir_: Optional[str] = None
                      ) -> 'TransformationSchemaConfig':
            return ReplaceLine.SchemaConfig()

        def build(self,
                  problem: 'Problem',
                  snippets: SnippetDatabase
                  ) -> 'TransformationSchema':
            assert isinstance(snippets, LineSnippetDatabase)
            return ReplaceLine.Schema(problem=problem, snippets=snippets)


@attr.s(frozen=True, auto_attribs=True)
class InsertLine(LineTransformation):
    _schema: LineTransformationSchema
    line: FileLine
    insertion: FileLine

    @property
    def schema(self) -> TransformationSchema:
        return self._schema

    def to_replacement(self) -> Replacement:
        sources = self._schema._problem.sources
        r = sources.line_to_location_range(self.line)
        r = FileLocationRange(r.filename, LocationRange(r.start, r.start))
        ins = sources.read_line(self.insertion, keep_newline=True)
        return Replacement(r, ins)

    class Schema(LineTransformationSchema):
        def find_all_at_line(self, line: FileLine) -> Iterator[Transformation]:
            # TODO append after the last line!
            for ins in self.viable_insertions(line):
                yield InsertLine(self, line, ins)

    class SchemaConfig(TransformationSchemaConfig):
        NAME = 'insert-line'

        @classmethod
        def from_dict(cls,
                      d: Mapping[str, Any],
                      dir_: Optional[str] = None
                      ) -> 'TransformationSchemaConfig':
            return InsertLine.SchemaConfig()

        def build(self,
                  problem: 'Problem',
                  snippets: SnippetDatabase
                  ) -> 'TransformationSchema':
            assert isinstance(snippets, LineSnippetDatabase)
            return InsertLine.Schema(problem=problem, snippets=snippets)


@attr.s(frozen=True, auto_attribs=True)
class InsertReturnLine(LineTransformation):
    _schema: LineTransformationSchema
    line: FileLine
    insertion: str

    @property
    def schema(self) -> TransformationSchema:
        return self._schema

    def to_replacement(self) -> Replacement:
        sources = self._schema._problem.sources
        r = sources.line_to_location_range(self.line)
        r = FileLocationRange(r.filename, LocationRange(r.start, r.start))
        return Replacement(r, self.insertion)

    @attr.s(frozen=True, auto_attribs=True)
    class Schema(LineTransformationSchema):
        target: Any
        language: str

        def insertion(self) -> str:
            if self.language.lower() == "c" or self.language.lower() == "c++":
                return f"return {self.target};\n"
            elif self.language.lower() == "python":
                return f"return {self.target}\n"
            else:
                raise BadConfigurationException(f"Unknown language {self.language}")

        def find_all_at_line(self, line: FileLine) -> Iterator[Transformation]:
            yield InsertReturnLine(self, line, self.insertion())

    @attr.s(frozen=True, auto_attribs=True)
    class SchemaConfig(TransformationSchemaConfig):
        NAME = 'insert-return-line'
        target: Any
        language: str

        @classmethod
        def from_dict(cls,
                      d: Mapping[str, Any],
                      dir_: Optional[str] = None
                      ) -> 'TransformationSchemaConfig':
            target: Any = d["target"]
            language: str = d["language"]
            return InsertReturnLine.SchemaConfig(target, language)

        def build(self,
                  problem: 'Problem',
                  snippets: SnippetDatabase
                  ) -> 'TransformationSchema':
            assert isinstance(snippets, LineSnippetDatabase)
            return InsertReturnLine.Schema(problem=problem, snippets=snippets,
                                           target=self.target,
                                           language=self.language)
