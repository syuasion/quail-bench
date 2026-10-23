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

from quail_b import prompts, sec
from quail_b.prompts import (
    AGENT_IMPLEMENTED_FIX,
    AGENT_RECOVERED,
    ASPECT_SENTIMENT,
    CARDIOVASCULAR_REACTION,
    CUAD_GOVERNING_LAW,
    CUAD_GOVERNING_LAW_DESCRIPTIONS,
    CUAD_GOVERNING_LAW_LABELS,
    CUAD_HAS_CLAUSE,
    CUAD_QUESTIONS,
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
    RUNS_DIFFERENT_APPROACH,
    RUNS_REPRODUCED,
    RUNS_TEST_STEP,
    SALES_CHANGED_OFFER,
    SALES_COMMITS,
    SALES_COMPETITOR,
    SALES_DISCOUNT,
    SCENARIO_MATCH,
    SERIOUS_ADVERSE_EVENT,
    SUPPORT,
    SUPPORT_DIFFERENT_APPROACH,
    SUPPORT_FRUSTRATED,
    SUPPORT_PUSHBACK,
    WRENCH_EXPLOITED,
    WRENCH_STEP_EXPLOIT,
    cuad_filter,
)
from quail_b.substrait import (
    AGGREGATE_GENERIC_EXTENSION_URN,
    AI_CLASSIFY_JOINED_NAME,
    AI_CLASSIFY_NAME,
    AI_EXTENSION_URN,
    AI_EXTRACT_NAME,
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
    AI_EXTRACT_NAME: (19, AI_EXTENSION_URN),
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
        floats: The columns among `columns` typed as 64-bit floats.
    """

    table: str
    alias: str
    text: str
    columns: tuple[str, ...] = ()
    integers: tuple[str, ...] = ()
    floats: tuple[str, ...] = ()


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
class Extract:
    """Add one answer column to one relation: the words that answer a question.

    Attributes:
        input: The relation's scan, filters, classifications, or
            extractions.
        question: The question the answer is copied for.
        output: The name of the answer column, null where the document
            does not answer the question.
    """

    input: "Scan | Where | Filter | Classify | InList | Extract"
    question: str
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

    input: "Scan | Where | Filter | Classify | InList | Score | Extract"
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

    input: "Scan | Filter | Classify | InList | Join | Extract"
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
        comparisons: Other conditions, as (left column, comparison, right
            column) in `aliases` order; for example ("id", "<", "id").
    """

    left: Scan | Filter | Join
    right: Scan | Filter | Join
    aliases: tuple[str, str]
    prompt: str
    on: tuple[tuple[str, str], ...] = ()
    comparisons: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class Struct:
    """Return answer columns as the fields of one struct column.

    Attributes:
        name: The struct column's name.
        fields: The answer columns, as `alias.column`, in field order.
    """

    name: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class Query:
    """One benchmark query.

    Attributes:
        id: The query ID.
        description: A short description for the catalog.
        tree: The operator tree.
        privacy: Whether the query reads the privacy policy corpus.
        select: Returned columns: an alias for its id, `alias.column`
            for a label column, or a Struct of answer columns; None
            returns every relation's id.
        draft: Exclude this query from the default published suite.
    """

    id: str
    description: str
    tree: "Scan | Filter | Join | Classify | InList | Extract"
    privacy: bool = False
    select: tuple[str, ...] | None = None
    draft: bool = False


def _string_type():
    return type_pb2.Type(
        string=type_pb2.Type.String(
            nullability=type_pb2.Type.NULLABILITY_REQUIRED
        )
    )


def _nullable_string_type():
    return type_pb2.Type(
        string=type_pb2.Type.String(
            nullability=type_pb2.Type.NULLABILITY_NULLABLE
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
        raise TypeError("a WHERE condition compares with a number or string")
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
                       "in-list": 0, "where": 0, "score": 0, "extract": 0,
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
                               else _fp64_type() if name in node.floats
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

        if isinstance(node, Extract):
            rel, fields, text = self.emit(node.input)
            (alias,) = text
            self.functions.add(AI_EXTRACT_NAME)
            output_fields = (*fields, (alias, node.output))
            common = _common(self._operator_id("extract"))
            common.hint.output_names.extend(
                ".".join(field) for field in output_fields)
            relation = algebra.ProjectRel(
                common=common,
                input=rel,
                expressions=[_call(AI_EXTRACT_NAME, [
                    _literal(node.question),
                    _field(fields.index((alias, text[alias]))),
                ], _nullable_string_type())],
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
        for left_column, comparison, right_column in (
                *((column, "=", partner) for column, partner in node.on),
                *node.comparisons):
            name = _COMPARISONS[comparison]
            self.functions.update({name, AND_NAME})
            conditions.append(_call(name, [
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
            for another column, a bare measure name, or a Struct of
            answer columns, or None for every relation's id in scan
            order.
    """
    emitter = _Emitter()
    rel, field_names = _emit_tail(emitter, tree)
    if select is None:
        select = [alias for alias, column in
                  (name.split(".", 1) for name in field_names if "." in name)
                  if column == "id"]
    names, expressions = [], []
    for item in select:
        if isinstance(item, Struct):
            names.extend((item.name, *map(output_name, item.fields)))
            expressions.append(algebra.Expression(nested=algebra.Expression.Nested(
                struct=algebra.Expression.Nested.Struct(fields=[
                    _field(field_names.index(name)) for name in item.fields]))))
            continue
        column = item if item in field_names else f"{item}.id"
        names.append(output_name(column))
        expressions.append(_field(field_names.index(column)))
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


def _support_traces(alias, text="transcript"):
    return Scan("support_traces", alias, text, ("task_id", "reward"),
                floats=("reward",))


def _user_messages():
    """The customer messages, with the id of the agent message each answers."""
    return Where(Scan("support_messages", "u", "content",
                      ("role", "prev_assistant_id")), "role", "=", "user")


def _pushback_pairs():
    """Each customer reply joined to the agent message before it."""
    return Join(Scan("support_messages", "a", "content"), _user_messages(),
                ("a", "u"), SUPPORT_PUSHBACK, on=(("id", "prev_assistant_id"),))


def _issue_runs(alias, text="transcript"):
    return Scan("issue_runs", alias, text, ("instance_id", "resolved"),
                integers=("resolved",))


def _wrench_runs(*columns, integers=()):
    return Scan("wrench_runs", "w", "transcript", columns, integers=integers)


def _wrench_steps():
    return Scan("wrench_steps", "s", "text", ("run_id",))


def _sales_calls(alias="c", *columns):
    return Scan("sales_calls", alias, "transcript", columns)


def _contracts(alias="c"):
    return Scan("contracts", alias, "text")


def _sec_filings():
    return Scan("sec_filings", "f", "text", ("filing_month",))


def _sec_event_filings():
    """The filings of 9,600 to 90,000 characters, filed in 2023 or later."""
    scan = Scan("sec_filings", "f", "text", ("filing_month", "text_chars"),
                integers=("text_chars",))
    return Where(Where(Where(scan, "text_chars", ">=", sec.SEC_LONG_MIN_CHARS),
                       "text_chars", "<=", sec.SEC_LONG_MAX_CHARS),
                 "filing_month", ">=", sec.SEC_EVENT_FIRST_MONTH)


def _sec_event(prompt, output):
    """One extraction over each long, recent filing."""
    return Extract(_sec_event_filings(), prompt, output)


def _sec_departures():
    return Filter(_sec_filings(), prompts.SEC_CEO_DEPARTURE)


def _sec_successors():
    return _classify(
        _sec_departures(), prompts.SEC_CEO_SUCCESSOR,
        prompts.SEC_CEO_SUCCESSOR_LABELS, "successor_status",
        prompts.SEC_CEO_SUCCESSOR_DESCRIPTIONS)


def _clause_types():
    return Scan("clause_types", "k", "question")


def _cuad_filters(node, *categories):
    for category in categories:
        node = Filter(node, cuad_filter(category))
    return node


def _cuad_extract(node, category, output):
    return Extract(node, CUAD_QUESTIONS[category], output)


def _governing_law(node):
    return _classify(node, CUAD_GOVERNING_LAW, CUAD_GOVERNING_LAW_LABELS,
                     "governing_law_state", CUAD_GOVERNING_LAW_DESCRIPTIONS)


def _law_counts(node):
    """Contracts per governing law label, most first."""
    return Sort(Aggregate(_governing_law(node), ("c.governing_law_state",),
                          (("contracts", "count", None),)),
                (("contracts", True), ("c.governing_law_state", False)))


def _next_call_pairs():
    """Each sales call joined to the next call of the same deal."""
    return Join(_sales_calls("c1"), _sales_calls("c2", "prev_call_id"),
                ("c1", "c2"), SALES_CHANGED_OFFER,
                on=(("id", "prev_call_id"),))


def _outcome_pairs():
    """Each successful run's patch joined to each failed run's patch."""
    return Join(Where(_issue_runs("s", "patch"), "resolved", "=", 1),
                Where(_issue_runs("f", "patch"), "resolved", "=", 0), ("s", "f"),
                RUNS_DIFFERENT_APPROACH, on=(("instance_id", "instance_id"),))


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
    Query("IMDB-16", "extract: the actor each review names",
          Extract(_reviews(), prompts.IMDB_ACTOR, "actor"),
          select=("r", Struct("review", ("r.actor",))), draft=True),
    Query("IMDB-17", "extract: what each review says about the acting",
          Extract(_reviews(), prompts.IMDB_ACTING, "acting"),
          select=("r", Struct("review", ("r.acting",))), draft=True),

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
    # backends refuse WHERE conditions, sorts, and aggregates.
    Query("REL-AGENT-1", "where: snapshots past turn 10 of at most "
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

    # Customer support traces and coding agent runs, for agent trace
    # analytics. Their reference labels are not published yet.
    Query("SUPPORT-1", "customer messages -> filter: frustrated with the agent",
          _filters(_user_messages(), SUPPORT_FRUSTRATED)),
    Query("SUPPORT-2", "join over pairs: each customer reply x the agent "
          "message before it, pushback",
          _pushback_pairs()),
    Query("SUPPORT-3", "pushback pairs -> classify: what the disagreement "
          "is about",
          Classify(_pushback_pairs(), prompts.SUPPORT_DISAGREEMENT,
                   prompts.SUPPORT_DISAGREEMENT_LABELS, "disagreement",
                   prompts.SUPPORT_DISAGREEMENT_DESCRIPTIONS,
                   documents=("u", "a")),
          select=("u", "a", "u.disagreement")),
    Query("SUPPORT-4", "classify each opening request: intent -> "
          "conversations per intent, most first",
          Sort(Aggregate(
              _classify(_support_traces("t", "request"),
                        prompts.SUPPORT_INTENT, prompts.SUPPORT_INTENT_LABELS,
                        "intent", prompts.SUPPORT_INTENT_DESCRIPTIONS),
              ("t.intent",), (("n", "count", None),)),
               (("n", True), ("t.intent", False))),
          select=("t.intent", "n")),
    Query("SUPPORT-5", "join over pairs: runs of the same task that handle "
          "the request differently",
          Join(_support_traces("r1"), _support_traces("r2"), ("r1", "r2"),
               SUPPORT_DIFFERENT_APPROACH, on=(("task_id", "task_id"),))),
    Query("SUPPORT-6", "classify each conversation: how the agent handled "
          "the request -> per outcome: conversations and mean reward, at "
          "least five, best first",
          Sort(Having(Aggregate(
              _classify(_support_traces("t"), prompts.SUPPORT_OUTCOME,
                        prompts.SUPPORT_OUTCOME_LABELS, "outcome",
                        prompts.SUPPORT_OUTCOME_DESCRIPTIONS),
              ("t.outcome",),
              (("n", "count", None), ("mean_reward", "avg", "t.reward"))),
              (("n", ">=", 5),)),
               (("mean_reward", True), ("t.outcome", False))),
          select=("t.outcome", "n", "mean_reward")),
    Query("RUNS-1", "filter: reproduced the issue before changing the code",
          _filters(_issue_runs("r"), RUNS_REPRODUCED)),
    Query("RUNS-2", "classify each run: the kind of change -> per kind: runs "
          "and the share resolved, most runs first",
          Sort(Aggregate(
              _classify(_issue_runs("r"), prompts.RUNS_STRATEGY,
                        prompts.RUNS_STRATEGY_LABELS, "strategy",
                        prompts.RUNS_STRATEGY_DESCRIPTIONS),
              ("r.strategy",),
              (("n", "count", None), ("resolved_share", "avg", "r.resolved"))),
               (("n", True), ("r.strategy", False))),
          select=("r.strategy", "n", "resolved_share")),
    Query("RUNS-3", "join over pairs: each successful run's patch x each "
          "failed run's patch for the same issue, different approach",
          _outcome_pairs()),
    Query("RUNS-4", "agent steps -> filter: runs tests -> per run: test "
          "steps, at least three, most first, 50",
          Fetch(Sort(Having(Aggregate(
              _filters(Where(Scan("issue_messages", "m", "content",
                                  ("trace_id", "role")), "role", "=",
                             "assistant"), RUNS_TEST_STEP),
              ("m.trace_id",), (("test_steps", "count", None),)),
              (("test_steps", ">=", 3),)),
              (("test_steps", True), ("m.trace_id", False))), 50),
          select=("m.trace_id", "test_steps")),

    # Terminal Wrench runs and their agent steps. Their reference labels
    # are not published yet.
    Query("WRENCH-1", "filter: runs that exploited the verifier",
          _filters(_wrench_runs(), WRENCH_EXPLOITED)),
    Query("WRENCH-2", "score: exploited the verifier -> the 100 highest "
          "scores",
          Fetch(Sort(Score(_wrench_runs("token_count",
                                        integers=("token_count",)),
                           WRENCH_EXPLOITED, "exploit_score"),
                     (("w.exploit_score", True), ("w.id", False))), 100),
          select=("w", "w.exploit_score")),
    Query("WRENCH-3", "exploited the verifier -> classify: the kind of "
          "exploit",
          _classify(_filters(_wrench_runs(), WRENCH_EXPLOITED),
                    prompts.WRENCH_EXPLOIT_KIND,
                    prompts.WRENCH_EXPLOIT_KIND_LABELS, "exploit_kind",
                    prompts.WRENCH_EXPLOIT_KIND_DESCRIPTIONS),
          select=("w", "w.exploit_kind")),
    Query("WRENCH-4", "agent steps -> filter: part of an exploit -> per run: "
          "flagged steps, at least two, most first, 50",
          Fetch(Sort(Having(Aggregate(
              _filters(_wrench_steps(), WRENCH_STEP_EXPLOIT),
              ("s.run_id",), (("flagged_steps", "count", None),)),
              (("flagged_steps", ">=", 2),)),
              (("flagged_steps", True), ("s.run_id", False))), 50),
          select=("s.run_id", "flagged_steps")),
    Query("WRENCH-5", "where: baseline runs -> filter: exploited the "
          "verifier -> flagged runs per agent model, most first",
          Sort(Aggregate(
              _filters(Where(_wrench_runs("model", "mode"), "mode", "=",
                             "baseline"), WRENCH_EXPLOITED),
              ("w.model",), (("flagged", "count", None),)),
               (("flagged", True), ("w.model", False))),
          select=("w.model", "flagged")),
    Query("WRENCH-6", "filter: runs that exploited the verifier -> extract: "
          "the exploiting command or edit",
          Extract(_filters(_wrench_runs(), WRENCH_EXPLOITED),
                  prompts.WRENCH_EXPLOIT_ACTION, "exploit_action"),
          select=("w", Struct("exploit", ("w.exploit_action",))), draft=True),
    Query("WRENCH-7", "filter: steps that are part of an exploit -> extract: "
          "the file or check each step tampers with",
          Extract(_filters(_wrench_steps(), WRENCH_STEP_EXPLOIT),
                  prompts.WRENCH_EXPLOIT_TARGET, "exploit_target"),
          select=("s", Struct("exploit", ("s.exploit_target",))),
          draft=True),
    Query("WRENCH-8", "agent steps -> extract: the thought that plans to "
          "cheat and the command that peeks at the tests",
          Extract(Extract(_wrench_steps(), prompts.WRENCH_CHEAT_PLAN, "plan"),
                  prompts.WRENCH_CHEAT_COMMAND, "command"),
          select=("s", Struct("cheat", ("s.plan", "s.command"))), draft=True),

    # CRMArena-Pro sales calls. Their reference labels are not published
    # yet.
    Query("SALES-1", "filter: calls that name a competitor",
          _filters(_sales_calls(), SALES_COMPETITOR)),
    Query("SALES-2", "classify each call: the customer's main concern -> "
          "calls per concern, most first",
          Sort(Aggregate(
              _classify(_sales_calls(), prompts.SALES_CONCERN,
                        prompts.SALES_CONCERN_LABELS, "concern",
                        prompts.SALES_CONCERN_DESCRIPTIONS),
              ("c.concern",), (("n", "count", None),)),
               (("n", True), ("c.concern", False))),
          select=("c.concern", "n")),
    Query("SALES-3", "join over pairs: each call x the next call of the same "
          "deal, the rep quotes a different price or discount",
          _next_call_pairs()),
    Query("SALES-4", "2 filters: the rep offers a discount + the customer "
          "commits to buying",
          _filters(_sales_calls(), SALES_DISCOUNT, SALES_COMMITS)),
    Query("SALES-5", "where: deals in negotiation -> score: the "
          "customer commits to buying -> the 25 highest scores",
          Fetch(Sort(Score(Where(_sales_calls("c", "deal_stage"),
                                 "deal_stage", "=", "Negotiation"),
                           SALES_COMMITS, "commit_score"),
                     (("c.commit_score", True), ("c.id", False))), 25),
          select=("c", "c.commit_score")),

    # CUAD contracts, with the lawyers' labels. Their reference labels
    # are not published yet.
    Query("CUAD-1", "extract: the governing law of each contract",
          _cuad_extract(_contracts(), "Governing Law", "governing_law"),
          select=("c", Struct("contract", ("c.governing_law",)))),
    Query("CUAD-2", "extract: the agreement date of each contract",
          _cuad_extract(_contracts(), "Agreement Date", "agreement_date"),
          select=("c", Struct("contract", ("c.agreement_date",)))),
    Query("CUAD-3", "extract: the cap on liability clause, null without one",
          _cuad_extract(_contracts(), "Cap On Liability", "liability_cap"),
          select=("c", Struct("contract", ("c.liability_cap",)))),
    Query("CUAD-4", "extract: the parties, several spans per contract",
          _cuad_extract(_contracts(), "Parties", "parties"),
          select=("c", Struct("contract", ("c.parties",)))),
    Query("CUAD-5", "filter: termination for convenience",
          _cuad_filters(_contracts(), "Termination For Convenience")),
    Query("CUAD-6", "2 filters: non-compete + exclusivity",
          _cuad_filters(_contracts(), "Non-Compete", "Exclusivity")),
    Query("CUAD-7", "filter: audit rights -> extract: the audit rights clause",
          _cuad_extract(_cuad_filters(_contracts(), "Audit Rights"),
                        "Audit Rights", "audit_rights"),
          select=("c", Struct("contract", ("c.audit_rights",)))),
    Query("CUAD-8", "2 filters: cap on liability + insurance -> extract: "
          "the cap on liability clause",
          _cuad_extract(_cuad_filters(_contracts(), "Cap On Liability",
                                      "Insurance"),
                        "Cap On Liability", "liability_cap"),
          select=("c", Struct("contract", ("c.liability_cap",)))),
    Query("CUAD-9", "2 filters: license grant + non-transferable license -> "
          "extract: the license grant",
          _cuad_extract(_cuad_filters(_contracts(), "License Grant",
                                      "Non-Transferable License"),
                        "License Grant", "license_grant"),
          select=("c", Struct("contract", ("c.license_grant",)))),
    Query("CUAD-10", "3 extracts: the agreement date, the expiration date, "
          "and the governing law of each contract",
          _cuad_extract(_cuad_extract(_cuad_extract(
              _contracts(), "Agreement Date", "agreement_date"),
              "Expiration Date", "expiration_date"),
              "Governing Law", "governing_law"),
          select=("c", Struct("contract", (
              "c.agreement_date", "c.expiration_date", "c.governing_law")))),
    Query("CUAD-11", "classify each contract: its governing law -> contracts "
          "per law, most first",
          _law_counts(_contracts()),
          select=("c.governing_law_state", "contracts")),
    Query("CUAD-12", "filter: insurance -> classify: the governing law -> "
          "contracts per law, most first",
          _law_counts(_cuad_filters(_contracts(), "Insurance")),
          select=("c.governing_law_state", "contracts")),
    Query("CUAD-13", "join: contracts x clause types (33 types)",
          Join(_contracts(), _clause_types(), ("c", "k"), CUAD_HAS_CLAUSE)),
    Query("CUAD-14", "filter: change of control -> join: clause types",
          Join(_cuad_filters(_contracts(), "Change Of Control"),
               _clause_types(), ("c", "k"), CUAD_HAS_CLAUSE)),
    Query("CUAD-15", "filter: termination for convenience -> extract: the "
          "expiration date -> by id, 20",
          Fetch(Sort(_cuad_extract(
              _cuad_filters(_contracts(), "Termination For Convenience"),
              "Expiration Date", "expiration_date"),
              (("c.id", False),)), 20),
          select=("c", Struct("contract", ("c.expiration_date",)))),

    # SEC inputs are built locally until reference labels are published.
    Query("SEC-1", "filter: announced CEO departure", _sec_departures(),
          draft=True),
    Query("SEC-2", "where: long, recent filing -> extract: party in the reported event",
          _sec_event(prompts.SEC_EVENT_PARTY, "event_party"),
          select=("f", Struct("report", ("f.event_party",))), draft=True),
    Query("SEC-3", "where: long, recent filing -> extract: dollar amount of the event",
          _sec_event(prompts.SEC_EVENT_AMOUNT, "event_amount"),
          select=("f", Struct("report", ("f.event_amount",))), draft=True),
    Query("SEC-4", "filter: CEO departure -> extract: name, date, reason",
          Extract(Extract(Extract(
              _sec_departures(), prompts.SEC_CEO_NAME, "ceo_name"),
              prompts.SEC_CEO_DATE, "departure_date"),
              prompts.SEC_CEO_REASON, "departure_reason"),
          select=("f", Struct("departure", (
              "f.ceo_name", "f.departure_date", "f.departure_reason"))),
          draft=True),
    Query("SEC-5", "filter: CEO departure -> classify: successor status",
          _sec_successors(), select=("f", "f.successor_status"), draft=True),
    Query("SEC-6", "filter: CEO departure -> classify: successor status -> "
          "filings per month and status",
          Sort(Aggregate(_sec_successors(),
                         ("f.filing_month", "f.successor_status"),
                         (("filings", "count", None),)),
               (("f.filing_month", False), ("f.successor_status", False))),
          select=("f.filing_month", "f.successor_status", "filings"), draft=True),
    Query("SEC-7", "where: long, recent filing -> extract: sentence stating the event",
          _sec_event(prompts.SEC_EVENT_SENTENCE, "event_sentence"),
          select=("f", Struct("report", ("f.event_sentence",))), draft=True),
    Query("SEC-8", "where: long, recent filing -> extract: what the event changes",
          _sec_event(prompts.SEC_EVENT_CHANGE, "event_change"),
          select=("f", Struct("report", ("f.event_change",))), draft=True),

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
                   prompts.TA_1_2), draft=True),
    Query("TA-2", "classify: the department responsible for each complaint",
          Classify(_tripadvisor_reviews(), prompts.TA_2,
                   prompts.TA_2_LABELS, "department",
                   prompts.TA_2_DESCRIPTIONS),
          select=("r", "r.department"), draft=True),
    Query("TA-3", "two TA_3_1 complaint filters -> TA_3 join: unique "
          "same-hotel review pairs report the same problem",
          Join(_filters(Scan("tripadvisor_reviews", "r1", "body", ("hotel_id",)),
                        prompts.TA_3_1),
               _filters(Scan("tripadvisor_reviews", "r2", "body", ("hotel_id",)),
                        prompts.TA_3_1),
               ("r1", "r2"), prompts.TA_3,
               on=(("hotel_id", "hotel_id"),),
               comparisons=(("id", "<", "id"),)), draft=True),
    Query("REL-TA-1", "REL_TA_1 score: top 10 reviews",
          Fetch(Sort(Score(_tripadvisor_reviews(), prompts.REL_TA_1,
                           "forced_move_score"),
                     (("r.forced_move_score", True), ("r.id", False))), 10),
          select=("r", "r.forced_move_score"), draft=True),
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
        if query.draft:
            entry["draft"] = True
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
