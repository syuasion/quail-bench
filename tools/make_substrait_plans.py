"""Write the QUAIL-B query plans as Substrait ProtoJSON files.

    uv run python tools/make_substrait_plans.py [output_dir]

The output directory defaults to `quail_b/plans`. The checked-in files
there are the published query definitions; a test checks that they
equal this script's output, so a prompt or query shape is changed here
and in `quail_b/prompts.py`, then the files are regenerated.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from google.protobuf import json_format
from substrait import algebra_pb2 as algebra
from substrait import plan_pb2, type_pb2
from substrait.extensions import extensions_pb2

from quail_b import prompts
from quail_b.prompts import (
    AGENT_IMPLEMENTED_FIX,
    AGENT_RECOVERED,
    ASPECT_SENTIMENT,
    CARDIOVASCULAR_REACTION,
    DISCUSS_ASPECT,
    F1,
    F4,
    F5,
    F11,
    F12,
    F13,
    LEP1,
    LEP2,
    LEPJOIN,
    LEPS1,
    NEUROLOGICAL_REACTION,
    P_LOC,
    P_MSG,
    REACTION,
    REFUTE,
    SCENARIO_MATCH,
    SERIOUS_ADVERSE_EVENT,
    SUPPORT,
)
from quail_b.substrait import (
    AGGREGATE_GENERIC_EXTENSION_URN,
    AI_CLASSIFY_JOINED_NAME,
    AI_CLASSIFY_NAME,
    AI_EXTENSION_URN,
    AI_FILTER_NAME,
    AI_JOIN_NAME,
    AI_SCORE_NAME,
    AND_NAME,
    ARITHMETIC_EXTENSION_URN,
    BOOLEAN_EXTENSION_URN,
    COMPARISON_EXTENSION_URN,
    EQUAL_NAME,
    SUBSTRAIT_VERSION,
    output_name,
)

PLANS_DIR = Path(__file__).resolve().parents[1] / "quail_b" / "plans"

# Function anchors are fixed so expressions can reference them while
# the tree is built; only the functions a plan uses are declared.
_URN_ANCHORS = {
    AI_EXTENSION_URN: 1,
    COMPARISON_EXTENSION_URN: 2,
    BOOLEAN_EXTENSION_URN: 3,
    AGGREGATE_GENERIC_EXTENSION_URN: 4,
    ARITHMETIC_EXTENSION_URN: 5,
}
_FUNCTIONS = {
    AI_FILTER_NAME: (1, AI_EXTENSION_URN),
    AI_JOIN_NAME: (2, AI_EXTENSION_URN),
    EQUAL_NAME: (3, COMPARISON_EXTENSION_URN),
    AND_NAME: (4, BOOLEAN_EXTENSION_URN),
    AI_CLASSIFY_NAME: (5, AI_EXTENSION_URN),
    AI_CLASSIFY_JOINED_NAME: (6, AI_EXTENSION_URN),
    AI_SCORE_NAME: (7, AI_EXTENSION_URN),
    "not_equal:any_any": (8, COMPARISON_EXTENSION_URN),
    "lt:any_any": (9, COMPARISON_EXTENSION_URN),
    "lte:any_any": (10, COMPARISON_EXTENSION_URN),
    "gt:any_any": (11, COMPARISON_EXTENSION_URN),
    "gte:any_any": (12, COMPARISON_EXTENSION_URN),
    "count:any": (13, AGGREGATE_GENERIC_EXTENSION_URN),
    "sum:i32": (14, ARITHMETIC_EXTENSION_URN),
    "avg:i32": (15, ARITHMETIC_EXTENSION_URN),
    "avg:fp64": (16, ARITHMETIC_EXTENSION_URN),
    "min:i32": (17, ARITHMETIC_EXTENSION_URN),
    "max:i32": (18, ARITHMETIC_EXTENSION_URN),
}
_COMPARISONS = {
    "=": EQUAL_NAME, "<>": "not_equal:any_any", "<": "lt:any_any",
    "<=": "lte:any_any", ">": "gt:any_any", ">=": "gte:any_any",
}


@dataclass(frozen=True)
class Scan:
    """Read one document table.

    Attributes:
        table: The table name.
        alias: The relation alias, unique within a query.
        text: The column the AI functions read.
        columns: Further columns, for ordinary join conditions, column
            tests, keys, and measures.
        integers: The columns among `columns` typed as 32-bit integers.
    """

    table: str
    alias: str
    text: str
    columns: tuple[str, ...] = ()
    integers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Where:
    """Keep the documents whose column passes a test against a literal.

    The test runs before any AI function reads the relation, so it
    sits directly over the scan.
    """

    input: "Scan | Where"
    column: str
    comparison: str
    value: int | float | str


@dataclass(frozen=True)
class Score:
    """Add a number column: the model's belief that a document answers TRUE."""

    input: "Scan | Where | Filter | Classify | InList | Score"
    prompt: str
    output: str


@dataclass(frozen=True)
class Aggregate:
    """Group the rows by key fields and compute measures per group.

    Attributes:
        input: The AI tree.
        keys: Fields grouped by, as `alias.column` or a label name.
        measures: (name, function, argument) per measure, the argument
            a field or None for `count(*)`. No measures means DISTINCT.
    """

    input: object
    keys: tuple[str, ...]
    measures: tuple[tuple[str, str, str | None], ...] = ()


@dataclass(frozen=True)
class Having:
    """Keep the groups whose measures pass (name, comparison, value) tests."""

    input: Aggregate
    tests: tuple[tuple[str, str, int | float], ...]


@dataclass(frozen=True)
class Sort:
    """Order the rows by (field, descending) keys."""

    input: object
    keys: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class Fetch:
    """Skip `offset` rows and keep at most `count`."""

    input: object
    count: int
    offset: int = 0


@dataclass(frozen=True)
class Filter:
    """Keep the documents of one relation that answer a prompt TRUE."""

    input: "Scan | Where | Filter | Classify | InList | Score"
    prompt: str


@dataclass(frozen=True)
class Classify:
    """Add one label column to one relation from a fixed list of labels.

    Over a join of two relations, the call labels each row the join
    kept: the prompt names the anchor as `{0}` and its partner as `{1}`,
    and the label column belongs to the anchor.

    Attributes:
        input: The relation's scan, filters, or classifications, or a
            join of two relations to label its rows.
        prompt: The classification prompt.
        labels: The labels, in tie-breaking order.
        output: The name of the label column.
        descriptions: One description per label, or empty for none.
        documents: For joined rows, the anchor alias then the partner
            alias; None takes the join's aliases in order.
    """

    input: Scan | Filter | Classify | InList | Join
    prompt: str
    labels: tuple[str, ...]
    output: str
    descriptions: tuple[str, ...] = ()
    documents: tuple[str, str] | None = None


@dataclass(frozen=True)
class InList:
    """Keep the documents whose label column holds an accepted label."""

    input: Classify | InList
    output: str
    accepted: tuple[str, ...]


@dataclass(frozen=True)
class Join:
    """Keep the pairs of documents that answer a prompt TRUE.

    Attributes:
        left: The left input.
        right: The right input.
        aliases: The relations the prompt's `{0}` and `{1}` refer to.
        prompt: The join prompt.
        on: Column pairs, in `aliases` order, that must also be equal.
    """

    left: Scan | Filter | Join
    right: Scan | Filter | Join
    aliases: tuple[str, str]
    prompt: str
    on: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Query:
    """One benchmark query.

    Attributes:
        id: The query ID.
        description: A short description for the catalog.
        tree: The operator tree.
        privacy: Whether the query reads the privacy policy corpus.
        select: Returned columns: an alias for its id, `alias.column`
            for a label column; None returns every relation's id.
        labels_pending: Whether its reference labels are unpublished.
    """

    id: str
    description: str
    tree: Scan | Filter | Join | Classify | InList
    privacy: bool = False
    select: tuple[str, ...] | None = None
    labels_pending: bool = False


def _string_type():
    return type_pb2.Type(
        string=type_pb2.Type.String(
            nullability=type_pb2.Type.NULLABILITY_REQUIRED
        )
    )


def _i32_type():
    return type_pb2.Type(
        i32=type_pb2.Type.I32(nullability=type_pb2.Type.NULLABILITY_REQUIRED)
    )


def _i64_type():
    return type_pb2.Type(
        i64=type_pb2.Type.I64(nullability=type_pb2.Type.NULLABILITY_REQUIRED)
    )


def _fp64_type():
    return type_pb2.Type(
        fp64=type_pb2.Type.FP64(nullability=type_pb2.Type.NULLABILITY_REQUIRED)
    )


def _number(value):
    """Return a literal for a Python int, float, or string."""
    if isinstance(value, bool):
        raise TypeError("a column test compares with a number or string")
    if isinstance(value, int):
        return algebra.Expression(literal=algebra.Expression.Literal(i32=value))
    if isinstance(value, float):
        return algebra.Expression(literal=algebra.Expression.Literal(fp64=value))
    return _literal(value)


def _bool_type():
    return type_pb2.Type(
        bool=type_pb2.Type.Boolean(
            nullability=type_pb2.Type.NULLABILITY_REQUIRED
        )
    )


def _common(alias):
    return algebra.RelCommon(
        direct=algebra.RelCommon.Direct(),
        hint=algebra.RelCommon.Hint(alias=alias),
    )


def _field(index):
    return algebra.Expression(
        selection=algebra.Expression.FieldReference(
            direct_reference=algebra.Expression.ReferenceSegment(
                struct_field=algebra.Expression.ReferenceSegment.StructField(
                    field=index
                )
            ),
            root_reference=algebra.Expression.FieldReference.RootReference(),
        )
    )


def _literal(text):
    return algebra.Expression(
        literal=algebra.Expression.Literal(string=text)
    )


def _string_list(values):
    return algebra.Expression(
        literal=algebra.Expression.Literal(
            list=algebra.Expression.Literal.List(values=[
                algebra.Expression.Literal(string=value) for value in values
            ])
        )
    )


def _call(name, arguments, output_type=None):
    return algebra.Expression(
        scalar_function=algebra.Expression.ScalarFunction(
            function_reference=_FUNCTIONS[name][0],
            arguments=[
                algebra.FunctionArgument(value=argument)
                for argument in arguments
            ],
            output_type=output_type or _bool_type(),
        )
    )


class _Emitter:
    """Turn a query tree into Substrait relations.

    Operators are numbered in post-order (inputs before the operator,
    left before right), which is the order the benchmark's plan reader
    reports them in.
    """

    def __init__(self):
        self.counts = {"filter": 0, "join": 0, "classify": 0,
                       "in-list": 0, "where": 0, "score": 0,
                       "aggregate": 0, "having": 0, "sort": 0, "fetch": 0}
        self.functions = set()

    def _operator_id(self, kind):
        self.counts[kind] += 1
        return f"{kind}-{self.counts[kind]}"

    def emit(self, node):
        """Return (rel, fields, text columns by alias) for one tree."""
        if isinstance(node, Scan):
            names = ("id", node.text, *node.columns)
            read = algebra.ReadRel(
                common=_common(node.alias),
                base_schema=type_pb2.NamedStruct(
                    names=names,
                    struct=type_pb2.Type.Struct(
                        types=[_i32_type() if name in node.integers
                               else _string_type() for name in names],
                        nullability=type_pb2.Type.NULLABILITY_REQUIRED,
                    ),
                ),
                named_table=algebra.ReadRel.NamedTable(names=[node.table]),
            )
            fields = tuple((node.alias, name) for name in names)
            return algebra.Rel(read=read), fields, {node.alias: node.text}

        if isinstance(node, Where):
            rel, fields, text = self.emit(node.input)
            (alias,) = text
            name = _COMPARISONS[node.comparison]
            self.functions.add(name)
            relation = algebra.FilterRel(
                common=_common(self._operator_id("where")),
                input=rel,
                condition=_call(name, [
                    _field(fields.index((alias, node.column))),
                    _number(node.value),
                ]),
            )
            return algebra.Rel(filter=relation), fields, text

        if isinstance(node, Score):
            rel, fields, text = self.emit(node.input)
            (alias,) = text
            self.functions.add(AI_SCORE_NAME)
            output_fields = (*fields, (alias, node.output))
            common = _common(self._operator_id("score"))
            common.hint.output_names.extend(
                ".".join(field) for field in output_fields)
            relation = algebra.ProjectRel(
                common=common,
                input=rel,
                expressions=[_call(AI_SCORE_NAME, [
                    _literal(node.prompt),
                    _field(fields.index((alias, text[alias]))),
                ], _fp64_type())],
            )
            return algebra.Rel(project=relation), output_fields, text

        if isinstance(node, Classify):
            rel, fields, text = self.emit(node.input)
            if len(text) == 1:
                aliases = tuple(text)
                name = AI_CLASSIFY_NAME
            else:
                aliases = node.documents or _join_aliases(node.input)
                name = AI_CLASSIFY_JOINED_NAME
            self.functions.add(name)
            descriptions = node.descriptions or ("",) * len(node.labels)
            output_fields = (*fields, (aliases[0], node.output))
            common = _common(self._operator_id("classify"))
            common.hint.output_names.extend(
                ".".join(field) for field in output_fields)
            relation = algebra.ProjectRel(
                common=common,
                input=rel,
                expressions=[_call(name, [
                    _literal(node.prompt),
                    *(_field(fields.index((alias, text[alias])))
                      for alias in aliases),
                    _string_list(node.labels),
                    _string_list(descriptions),
                ], _string_type())],
            )
            return algebra.Rel(project=relation), output_fields, text

        if isinstance(node, InList):
            rel, fields, text = self.emit(node.input)
            # the label column may belong to any relation the input holds
            (field,) = [field for field in fields if field[1] == node.output]
            relation = algebra.FilterRel(
                common=_common(self._operator_id("in-list")),
                input=rel,
                condition=algebra.Expression(
                    singular_or_list=algebra.Expression.SingularOrList(
                        value=_field(fields.index(field)),
                        options=[_literal(label) for label in node.accepted],
                    )
                ),
            )
            return algebra.Rel(filter=relation), fields, text

        if isinstance(node, Filter):
            rel, fields, text = self.emit(node.input)
            (alias,) = text
            self.functions.add(AI_FILTER_NAME)
            relation = algebra.FilterRel(
                common=_common(self._operator_id("filter")),
                input=rel,
                condition=_call(AI_FILTER_NAME, [
                    _literal(node.prompt),
                    _field(fields.index((alias, text[alias]))),
                ]),
            )
            return algebra.Rel(filter=relation), fields, text

        left, left_fields, left_text = self.emit(node.left)
        right, right_fields, right_text = self.emit(node.right)
        fields = left_fields + right_fields
        text = {**left_text, **right_text}
        first, second = node.aliases
        self.functions.add(AI_JOIN_NAME)
        conditions = [_call(AI_JOIN_NAME, [
            _literal(node.prompt),
            _field(fields.index((first, text[first]))),
            _field(fields.index((second, text[second]))),
        ])]
        for left_column, right_column in node.on:
            self.functions.update({EQUAL_NAME, AND_NAME})
            conditions.append(_call(EQUAL_NAME, [
                _field(fields.index((first, left_column))),
                _field(fields.index((second, right_column))),
            ]))
        expression = (
            conditions[0] if len(conditions) == 1
            else _call(AND_NAME, conditions)
        )
        relation = algebra.JoinRel(
            common=_common(self._operator_id("join")),
            left=left,
            right=right,
            expression=expression,
            type=algebra.JoinRel.JOIN_TYPE_INNER,
        )
        return algebra.Rel(join=relation), fields, text


def _join_aliases(node) -> tuple[str, str]:
    """The aliases of the join under a classification, in prompt order."""
    while not isinstance(node, Join):
        node = node.input
    return node.aliases


def _emit_tail(emitter, node):
    """Return (rel, field names) for the relational steps over the AI tree."""
    if isinstance(node, Fetch):
        rel, names = _emit_tail(emitter, node.input)
        relation = algebra.FetchRel(
            common=_common(emitter._operator_id("fetch")),
            input=rel,
            offset_expr=algebra.Expression(
                literal=algebra.Expression.Literal(i64=node.offset)),
            count_expr=algebra.Expression(
                literal=algebra.Expression.Literal(i64=node.count)),
        )
        return algebra.Rel(fetch=relation), names
    if isinstance(node, Sort):
        rel, names = _emit_tail(emitter, node.input)
        relation = algebra.SortRel(
            common=_common(emitter._operator_id("sort")),
            input=rel,
            sorts=[algebra.SortField(
                expr=_field(names.index(name)),
                direction=(
                    algebra.SortField.SORT_DIRECTION_DESC_NULLS_LAST
                    if descending
                    else algebra.SortField.SORT_DIRECTION_ASC_NULLS_LAST))
                for name, descending in node.keys],
        )
        return algebra.Rel(sort=relation), names
    if isinstance(node, Having):
        rel, names = _emit_tail(emitter, node.input)
        conditions = []
        for name, comparison, value in node.tests:
            function = _COMPARISONS[comparison]
            emitter.functions.add(function)
            conditions.append(_call(function, [
                _field(names.index(name)), _number(value)]))
        if len(conditions) > 1:
            emitter.functions.add(AND_NAME)
        relation = algebra.FilterRel(
            common=_common(emitter._operator_id("having")),
            input=rel,
            condition=(conditions[0] if len(conditions) == 1
                       else _call(AND_NAME, conditions)),
        )
        return algebra.Rel(filter=relation), names
    if isinstance(node, Aggregate):
        rel, fields, _text = emitter.emit(node.input)
        names = tuple(".".join(field) for field in fields)
        output = (*node.keys, *(name for name, _, _ in node.measures))
        common = _common(emitter._operator_id("aggregate"))
        common.hint.output_names.extend(output)
        measures = []
        for _name, function, argument in node.measures:
            if argument is None:
                kernel = "count:any"
                arguments = []
            else:
                kernel = {
                    "count": "count:any", "count_distinct": "count:any",
                    "sum": "sum:i32", "min": "min:i32", "max": "max:i32",
                    "avg": ("avg:fp64" if argument.rpartition(".")[2]
                            in _score_columns(node.input) else "avg:i32"),
                }[function]
                arguments = [algebra.FunctionArgument(
                    value=_field(names.index(argument)))]
            emitter.functions.add(kernel)
            measures.append(algebra.AggregateRel.Measure(
                measure=algebra.AggregateFunction(
                    function_reference=_FUNCTIONS[kernel][0],
                    arguments=arguments,
                    output_type=(_fp64_type() if kernel == "avg:fp64"
                                 else _i64_type()),
                    phase=algebra.AGGREGATION_PHASE_INITIAL_TO_RESULT,
                    invocation=(
                        algebra.AggregateFunction.AGGREGATION_INVOCATION_DISTINCT
                        if function == "count_distinct"
                        else algebra.AggregateFunction.AGGREGATION_INVOCATION_ALL),
                )))
        relation = algebra.AggregateRel(
            common=common,
            input=rel,
            grouping_expressions=[_field(names.index(key)) for key in node.keys],
            groupings=[algebra.AggregateRel.Grouping(
                expression_references=list(range(len(node.keys))))],
            measures=measures,
        )
        return algebra.Rel(aggregate=relation), output
    rel, fields, _text = emitter.emit(node)
    return rel, tuple(".".join(field) for field in fields)


def _score_columns(node) -> set[str]:
    """Return the score column names an AI tree adds."""
    found = set()
    while node is not None and not isinstance(node, Join):
        if isinstance(node, Score):
            found.add(node.output)
        node = getattr(node, "input", None)
    return found


def build_plan(tree, select=None) -> plan_pb2.Plan:
    """Return the Substrait plan selecting the id of each relation.

    Args:
        tree: The query tree, with any relational steps on top.
        select: Returned columns, an alias for its id, `alias.column`
            for another column, or a bare measure name, or None for
            every relation's id in scan order.
    """
    emitter = _Emitter()
    rel, field_names = _emit_tail(emitter, tree)
    if select is None:
        select = [alias for alias, column in
                  (name.split(".", 1) for name in field_names if "." in name)
                  if column == "id"]
    columns = [name if name in field_names else f"{name}.id"
               for name in select]
    names = [output_name(column) for column in columns]
    expressions = [_field(field_names.index(column)) for column in columns]
    project = algebra.Rel(
        project=algebra.ProjectRel(
            common=algebra.RelCommon(
                emit=algebra.RelCommon.Emit(
                    output_mapping=[
                        len(field_names) + index
                        for index in range(len(expressions))
                    ]
                )
            ),
            input=rel,
            expressions=expressions,
        )
    )
    used = sorted(emitter.functions, key=lambda name: _FUNCTIONS[name][0])
    urns = sorted(
        {_FUNCTIONS[name][1] for name in used},
        key=_URN_ANCHORS.__getitem__,
    )
    return plan_pb2.Plan(
        version=plan_pb2.Version(
            major_number=SUBSTRAIT_VERSION[0],
            minor_number=SUBSTRAIT_VERSION[1],
            patch_number=SUBSTRAIT_VERSION[2],
            producer="quail-b",
        ),
        extension_urns=[
            extensions_pb2.SimpleExtensionURN(
                extension_urn_anchor=_URN_ANCHORS[urn],
                urn=urn,
            )
            for urn in urns
        ],
        extensions=[
            extensions_pb2.SimpleExtensionDeclaration(
                extension_function=(
                    extensions_pb2.SimpleExtensionDeclaration.ExtensionFunction(
                        extension_urn_reference=_URN_ANCHORS[_FUNCTIONS[name][1]],
                        function_anchor=_FUNCTIONS[name][0],
                        name=name,
                    )
                )
            )
            for name in used
        ],
        relations=[
            plan_pb2.PlanRel(
                root=algebra.RelRoot(input=project, names=names)
            )
        ],
    )


def _filters(node, *prompts):
    for prompt in prompts:
        node = Filter(node, prompt)
    return node


def _reviews(alias="r"):
    return Scan("reviews", alias, "body")


def _aspects(alias="a"):
    return Scan("aspects", alias, "aspect")


def _reports():
    return Scan("reports", "r", "report")


def _terms(alias="m"):
    return Scan("terms", alias, "term")


def _claims(alias="c", *columns):
    return Scan("claims", alias, "claim", columns)


def _evidence(alias="e"):
    return Scan("evidence", alias, "text")


def _contexts():
    return Scan("citation_contexts", "d", "destination_context")


def _passages():
    return Scan("citation_passages", "s", "passage_text")


def _traces():
    return Scan("agent_traces", "t", "trace")


def _traces_with_columns():
    """The agent traces with the columns the relational queries read."""
    return Scan("agent_traces", "t", "trace",
                ("trajectory_id", "turn_index", "token_count"),
                integers=("turn_index", "token_count"))


def _test_result(node):
    return _classify(node, prompts.AGENT_TEST_RESULT,
                     prompts.AGENT_TEST_RESULT_LABELS, "test_result",
                     prompts.AGENT_TEST_RESULT_DESCRIPTIONS)


def _policies():
    return Scan("policies", "p", "policy_text")


def _classify(node, prompt, labels, output, descriptions=()):
    return Classify(node, prompt, labels, output, descriptions)


def _sentiment(node):
    return _classify(node, prompts.IMDB_SENTIMENT,
                     prompts.IMDB_SENTIMENT_LABELS, "sentiment")


def _critical(node):
    """Reviews whose sentiment is negative or mixed."""
    return InList(_sentiment(node), "sentiment", ("negative", "mixed"))


def _organ_class(node):
    return _classify(node, prompts.BIO_ORGAN_CLASS,
                     prompts.BIO_ORGAN_CLASS_LABELS, "organ_class")


def _progress(node):
    return _classify(node, prompts.AGENT_PROGRESS,
                     prompts.AGENT_PROGRESS_LABELS, "progress",
                     prompts.AGENT_PROGRESS_DESCRIPTIONS)


def _imdb_chain(first):
    """Chain r1-a1-r2-a2: both reviews discuss a1, and r2 is positive about a2."""
    return Join(
        Join(
            Join(first, _aspects("a1"), ("r1", "a1"), DISCUSS_ASPECT),
            _reviews("r2"), ("r2", "a1"), DISCUSS_ASPECT,
        ),
        _aspects("a2"), ("r2", "a2"), ASPECT_SENTIMENT,
    )


def _fever_chain(c1, e1, c2, e2):
    """Chain c1-e1-c2-e2: e1 supports c1 but refutes c2, and e2 supports c2."""
    return Join(
        Join(Join(c1, e1, ("c1", "e1"), SUPPORT), c2, ("c2", "e1"), REFUTE),
        e2, ("c2", "e2"), SUPPORT,
    )


def _tripadvisor_reviews():
    return Scan("tripadvisor_reviews", "r", "body", ("hotel_id",))


QUERIES = (
    Query("IMDB-1", "filter: F1 (at least one positive aspect)",
          _filters(_reviews(), F1)),
    Query("IMDB-2", "join: J1 (reviews x aspects)",
          Join(_reviews(), _aspects(), ("r", "a"), DISCUSS_ASPECT)),
    Query("IMDB-3", "F1 -> J1, dependent",
          Join(_filters(_reviews(), F1), _aspects(), ("r", "a"),
               DISCUSS_ASPECT)),
    Query("IMDB-4", "F1 -> F4 -> J1, 2 filters then 1 join",
          Join(_filters(_reviews(), F1, F4), _aspects(), ("r", "a"),
               DISCUSS_ASPECT)),
    Query("IMDB-5", "F1 -> F4 -> F5 -> J1, 3 filters then 1 join",
          Join(_filters(_reviews(), F1, F4, F5), _aspects(), ("r", "a"),
               DISCUSS_ASPECT)),
    Query("IMDB-6", "F1 -> F4, 2 filters, no join",
          _filters(_reviews(), F1, F4)),
    Query("IMDB-7", "F1 -> F4 -> F5, 3 filters, no join",
          _filters(_reviews(), F1, F4, F5)),
    Query("IMDB-8", "2J, same anchor: J1 (DISCUSS_ASPECT) -> J2 "
          "(ASPECT_SENTIMENT), reviews x aspects x aspects",
          Join(Join(_reviews(), _aspects(), ("r", "a"), DISCUSS_ASPECT),
               _aspects("a2"), ("r", "a2"), ASPECT_SENTIMENT)),
    Query("IMDB-9", "3J chain r1-a1-r2-a2: two reviews discuss the same "
          "aspect, second review positive about another",
          _imdb_chain(_reviews("r1"))),
    Query("IMDB-10", "F1 -> 3J chain r1-a1-r2-a2",
          _imdb_chain(_filters(_reviews("r1"), F1))),
    Query("IMDB-11", "classify: the sentiment of every review",
          _sentiment(_reviews()), select=("r", "r.sentiment")),
    Query("IMDB-12", "F1 -> classify: the IMDb genre of each review's movie",
          _classify(_filters(_reviews(), F1), prompts.IMDB_GENRE,
                    prompts.IMDB_GENRE_LABELS, "genre"),
          select=("r", "r.genre")),
    Query("IMDB-13", "negative or mixed reviews -> J1: the aspects critical "
          "reviews discuss, with each review's sentiment",
          Join(_critical(_reviews()), _aspects(), ("r", "a"),
               DISCUSS_ASPECT),
          select=("r", "r.sentiment", "a")),
    Query("IMDB-14", "negative or mixed reviews -> classify: their main "
          "complaint, returned with their sentiment",
          _classify(_critical(_reviews()), prompts.IMDB_COMPLAINT,
                    prompts.IMDB_COMPLAINT_LABELS, "complaint"),
          select=("r", "r.sentiment", "r.complaint")),
    Query("IMDB-15", "negative or mixed reviews -> J1 -> classify: each "
          "review's sentiment toward the aspect it discusses",
          Classify(Join(_critical(_reviews()), _aspects(), ("r", "a"),
                        DISCUSS_ASPECT),
                   prompts.IMDB_ASPECT_SENTIMENT,
                   prompts.IMDB_ASPECT_SENTIMENT_LABELS, "aspect_sentiment"),
          select=("r", "r.sentiment", "a", "r.aspect_sentiment")),

    Query("BIO-1", "filter: serious adverse event",
          _filters(_reports(), SERIOUS_ADVERSE_EVENT)),
    Query("BIO-2", "join: J1 (reports x terms)",
          Join(_reports(), _terms(), ("r", "m"), REACTION)),
    Query("BIO-3", "serious adverse event -> reaction join",
          Join(_filters(_reports(), SERIOUS_ADVERSE_EVENT), _terms(),
               ("r", "m"), REACTION)),
    Query("BIO-4", "3F + 2J: serious reports with neurological and "
          "cardiovascular reactions",
          Join(Join(_filters(_reports(), SERIOUS_ADVERSE_EVENT),
                    _filters(_terms("n"), NEUROLOGICAL_REACTION),
                    ("r", "n"), REACTION),
               _filters(_terms("c"), CARDIOVASCULAR_REACTION),
               ("r", "c"), REACTION)),
    Query("BIO-5", "classify: the MedDRA system organ class of every "
          "reaction term",
          _organ_class(_terms()), select=("m", "m.organ_class")),
    Query("BIO-6", "serious adverse event reports x their cardiac or "
          "vascular reactions",
          Join(_filters(_reports(), SERIOUS_ADVERSE_EVENT),
               InList(_organ_class(_terms()), "organ_class",
                      ("cardiac disorders", "vascular disorders")),
               ("r", "m"), REACTION)),

    Query("FEV-1", "filter: F11 (about a person)", _filters(_claims(), F11)),
    Query("FEV-2", "join: J3 (claims x evidence)",
          Join(_claims(), _evidence(), ("c", "e"), SUPPORT)),
    Query("FEV-3", "F11 -> J3, dependent",
          Join(_filters(_claims(), F11), _evidence(), ("c", "e"), SUPPORT)),
    Query("FEV-4", "F11 -> F12 -> J3, 2 filters then 1 join",
          Join(_filters(_claims(), F11, F12), _evidence(), ("c", "e"),
               SUPPORT)),
    Query("FEV-5", "2F + 1J: two-sided pushdown - F11 on claims, F13 on "
          "evidence, each filtered before J3",
          Join(_filters(_claims(), F11), _filters(_evidence(), F13),
               ("c", "e"), SUPPORT)),
    Query("FEV-6", "3F + 1J: two-sided pushdown, deeper - F11 -> F12 on "
          "claims, F13 on evidence, each filtered before J3",
          Join(_filters(_claims(), F11, F12), _filters(_evidence(), F13),
               ("c", "e"), SUPPORT)),
    Query("FEV-7", "2J, same anchor: J1 (SUPPORT) -> J2 (REFUTE), "
          "claims x evidence x evidence",
          Join(Join(_claims(), _evidence(), ("c", "e"), SUPPORT),
               _evidence("e2"), ("c", "e2"), REFUTE)),
    Query("FEV-8", "3J chain c1-e1-c2-e2: evidence supports c1 but refutes "
          "c2, c2 supported by different evidence",
          _fever_chain(_claims("c1"), _evidence("e1"), _claims("c2"),
                       _evidence("e2"))),
    Query("FEV-9", "4F + 3J: F11 on c1 and c2, F13 on e1 and e2, then the "
          "c1-e1-c2-e2 join chain",
          _fever_chain(_filters(_claims("c1"), F11),
                       _filters(_evidence("e1"), F13),
                       _filters(_claims("c2"), F11),
                       _filters(_evidence("e2"), F13))),
    # A claim names its Wikipedia page, and an evidence row's id is its
    # page name, so SUPPORT is asked only of a claim and its own page.
    Query("FEV-10", "2F + 1J over pairs: F11 on claims, F13 on evidence, "
          "SUPPORT asked only of a claim and its own Wikipedia page",
          Join(_filters(_claims("c", "evidence_wiki_url"), F11),
               _filters(_evidence(), F13), ("c", "e"), SUPPORT,
               on=(("evidence_wiki_url", "id"),))),
    Query("FEV-11", "classify: claim topic; the political and historical "
          "claims, with their topic",
          InList(_classify(_claims(), prompts.FEV_TOPIC,
                           prompts.FEV_TOPIC_LABELS, "topic"),
                 "topic", ("politics", "history")),
          select=("c", "c.topic")),

    Query("LEP-1", "filter: LEP1 (reasoning does not apply)",
          _filters(_contexts(), LEP1)),
    Query("LEP-2", "join: citation contexts x cited passages",
          Join(_contexts(), _passages(), ("d", "s"), LEPJOIN)),
    Query("LEP-3", "LEP1 -> join, dependent",
          Join(_filters(_contexts(), LEP1), _passages(), ("d", "s"),
               LEPJOIN)),
    Query("LEP-4", "LEP1 -> LEP2 -> join, 2 filters then 1 join",
          Join(_filters(_contexts(), LEP1, LEP2), _passages(), ("d", "s"),
               LEPJOIN)),
    Query("LEP-5", "2F + 1J: two-sided pushdown - LEP1+LEP2 on excerpts, "
          "LEPS1 on passages, each filtered before the join",
          Join(_filters(_contexts(), LEP1, LEP2),
               _filters(_passages(), LEPS1), ("d", "s"), LEPJOIN)),
    Query("LEP-6", "constitutional or criminal law excerpts -> the "
          "passages they cite, with each excerpt's area of law",
          Join(InList(
                   _classify(_contexts(), prompts.LEP_AREA,
                             prompts.LEP_AREA_LABELS, "area"),
                   "area", ("constitutional law", "criminal law")),
               _passages(), ("d", "s"), LEPJOIN),
          select=("d", "d.area", "s")),
    Query("AGENT-1", "filter: recovered after an unsuccessful approach",
          _filters(_traces(), AGENT_RECOVERED)),
    Query("AGENT-2", "filter: implemented a plausible fix",
          _filters(_traces(), AGENT_IMPLEMENTED_FIX)),
    Query("AGENT-3", "recovered -> classify: how far the agent has gotten",
          _progress(_filters(_traces(), AGENT_RECOVERED)),
          select=("t", "t.progress")),
    Query("AGENT-4", "traces that changed the code -> classify: what the "
          "latest test or reproduction run showed",
          _classify(InList(_progress(_traces()), "progress",
                           prompts.AGENT_CHANGED_CODE),
                    prompts.AGENT_TEST_RESULT, prompts.AGENT_TEST_RESULT_LABELS,
                    "test_result", prompts.AGENT_TEST_RESULT_DESCRIPTIONS),
          select=("t", "t.progress", "t.test_result")),
    Query("AGENT-5", "classify every trace three ways: how far the agent "
          "has gotten, the project's PyPI topic, and the defect type of the "
          "bug",
          _classify(_classify(_progress(_traces()), prompts.AGENT_DOMAIN,
                              prompts.AGENT_DOMAIN_LABELS, "domain"),
                    prompts.AGENT_ROOT_CAUSE, prompts.AGENT_ROOT_CAUSE_LABELS,
                    "root_cause", prompts.AGENT_ROOT_CAUSE_DESCRIPTIONS),
          select=("t", "t.progress", "t.domain", "t.root_cause")),

    # Relational operators over the agent traces. Each query returns a
    # result small enough to read, and runs on Quail only: the stock
    # backends refuse column tests, sorts, and aggregates.
    Query("REL-AGENT-1", "column tests: snapshots past turn 10 of at most "
          "6,000 tokens -> filter: recovered",
          _filters(Where(Where(_traces_with_columns(), "turn_index", ">=", 10),
                         "token_count", "<=", 6000), AGENT_RECOVERED)),
    Query("REL-AGENT-2", "recovered -> the second page of ten, shortest "
          "trace first",
          Fetch(Sort(_filters(_traces_with_columns(), AGENT_RECOVERED),
                     (("t.token_count", False), ("t.id", False))),
                count=10, offset=10),
          select=("t", "t.token_count")),
    Query("REL-AGENT-3", "implemented a fix -> distinct trajectories",
          Aggregate(_filters(_traces_with_columns(), AGENT_IMPLEMENTED_FIX),
                    ("t.trajectory_id",)),
          select=("t.trajectory_id",)),
    Query("REL-AGENT-4", "score: recovered -> the 20 highest scores",
          Fetch(Sort(Score(_traces_with_columns(), AGENT_RECOVERED,
                           "recovered_score"),
                     (("t.recovered_score", True), ("t.id", False))), 20),
          select=("t", "t.recovered_score")),
    Query("REL-AGENT-5", "traces that changed the code -> classify: test "
          "result -> snapshots and trajectories per result, at least 50 "
          "snapshots, most first",
          Sort(Having(Aggregate(
              _test_result(InList(_progress(_traces_with_columns()),
                                  "progress", prompts.AGENT_CHANGED_CODE)),
              ("t.test_result",),
              (("n", "count", None),
               ("trajectories", "count_distinct", "t.trajectory_id"))),
              (("n", ">=", 50),)),
               (("n", True), ("t.test_result", False))),
          select=("t.test_result", "n", "trajectories")),
    Query("REL-AGENT-6", "implemented a fix -> per trajectory: fixes, first "
          "fix turn, longest trace; at least two fixes, earliest first, 50",
          Fetch(Sort(Having(Aggregate(
              _filters(_traces_with_columns(), AGENT_IMPLEMENTED_FIX),
              ("t.trajectory_id",),
              (("fixes", "count", None), ("first_fix", "min", "t.turn_index"),
               ("longest", "max", "t.token_count"))),
              (("fixes", ">=", 2),)),
              (("first_fix", False), ("t.trajectory_id", False))), 50),
          select=("t.trajectory_id", "fixes", "first_fix", "longest")),
    Query("REL-AGENT-7", "score: implemented a fix -> per trajectory of at "
          "least five snapshots: the mean score; the ten highest",
          Fetch(Sort(Having(Aggregate(
              Score(_traces_with_columns(), AGENT_IMPLEMENTED_FIX, "fix_score"),
              ("t.trajectory_id",),
              (("mean_fix_score", "avg", "t.fix_score"), ("n", "count", None))),
              (("n", ">=", 5),)),
              (("mean_fix_score", True), ("t.trajectory_id", False))), 10),
          select=("t.trajectory_id", "mean_fix_score", "n")),

    # PrivacyPolicies: only when that corpus is available.
    Query("PRIV-1", "2 filters: P_MSG + P_LOC",
          _filters(_policies(), P_MSG, P_LOC), privacy=True),
    Query("PRIV-2", "2 filters + 1 join: P_MSG + P_LOC -> scenarios",
          Join(_filters(_policies(), P_MSG, P_LOC),
               Scan("scenarios", "s", "scenario"), ("p", "s"),
               SCENARIO_MATCH), privacy=True),

    # TripAdvisor starts without reference labels and stays opt-in.
    Query("TA-1", "TA_1_1 -> TA_1_2, 2 filters",
          _filters(_tripadvisor_reviews(), prompts.TA_1_1,
                   prompts.TA_1_2), labels_pending=True),
    Query("TA-2", "classify: the department responsible for each complaint",
          Classify(_tripadvisor_reviews(), prompts.TA_2,
                   prompts.TA_2_LABELS, "department",
                   prompts.TA_2_DESCRIPTIONS),
          select=("r", "r.department"), labels_pending=True),
    Query("TA-3", "two TA_3_1 complaint filters -> TA_3 join: unique "
          "same-hotel pairs report the same problem",
          Join(_filters(Scan("tripadvisor_pair_left", "a", "body", ("pair_id",)),
                        prompts.TA_3_1),
               _filters(Scan("tripadvisor_pair_right", "b", "body", ("pair_id",)),
                        prompts.TA_3_1),
               ("a", "b"), prompts.TA_3,
               on=(("pair_id", "pair_id"),)), labels_pending=True),
    Query("REL-TA-1", "REL_TA_1 score: top 10 reviews",
          Fetch(Sort(Score(_tripadvisor_reviews(), prompts.REL_TA_1,
                           "forced_move_score"),
                     (("r.forced_move_score", True), ("r.id", False))), 10),
          select=("r", "r.forced_move_score"), labels_pending=True),
)


def plan_json(plan: plan_pb2.Plan) -> str:
    """Return a plan as ProtoJSON with sorted keys and a trailing newline."""
    return json_format.MessageToJson(
        plan,
        preserving_proto_field_name=True,
        indent=2,
        sort_keys=True,
    ) + "\n"


def write_plans(directory: Path) -> None:
    """Write every query plan and the catalog into a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    catalog = []
    for query in QUERIES:
        (directory / f"{query.id}.json").write_text(
            plan_json(build_plan(query.tree, query.select))
        )
        entry = {
            "id": query.id,
            "description": query.description,
            "privacy": query.privacy,
        }
        if query.labels_pending:
            entry["labels_pending"] = True
        catalog.append(entry)
    (directory / "catalog.json").write_text(
        json.dumps(catalog, indent=2) + "\n"
    )


def main(argv=None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    directory = Path(arguments[0]) if arguments else PLANS_DIR
    write_plans(directory)
    print(f"wrote {len(QUERIES)} plans to {directory}")


if __name__ == "__main__":
    main()
