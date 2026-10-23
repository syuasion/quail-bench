"""Read the benchmark information in a QUAIL-B Substrait plan."""

from __future__ import annotations

from dataclasses import dataclass

from substrait import algebra_pb2, plan_pb2

SUBSTRAIT_VERSION = (0, 103, 0)
AI_EXTENSION_URN = "extension:org.fsdatalab.quail_b:functions_ai"
COMPARISON_EXTENSION_URN = "extension:io.substrait:functions_comparison"
BOOLEAN_EXTENSION_URN = "extension:io.substrait:functions_boolean"

AI_FILTER_NAME = "ai_filter:str_str"
AI_JOIN_NAME = "ai_join:str_str_str"
AI_CLASSIFY_NAME = "ai_classify:str_str_list_list"
AI_CLASSIFY_JOINED_NAME = "ai_classify:str_str_str_list_list"
AI_SCORE_NAME = "ai_score:str_str"
AI_EXTRACT_NAME = "ai_extract:str_str"
EQUAL_NAME = "equal:any_any"
AND_NAME = "and:bool"
AGGREGATE_GENERIC_EXTENSION_URN = (
    "extension:io.substrait:functions_aggregate_generic")
ARITHMETIC_EXTENSION_URN = "extension:io.substrait:functions_arithmetic"
# comparisons in WHERE and join conditions, by Substrait function name
COMPARISON_NAMES = {
    "equal:any_any": "=", "not_equal:any_any": "<>",
    "lt:any_any": "<", "lte:any_any": "<=",
    "gt:any_any": ">", "gte:any_any": ">=",
}
# the comparison with its two sides swapped
_REVERSED = {"=": "=", "<>": "<>", "<": ">", "<=": ">=", ">": "<", ">=": "<="}
# aggregate measures, by Substrait function name
AGGREGATE_NAMES = {
    "count:any": "count", "sum:i32": "sum", "avg:i32": "avg",
    "avg:fp64": "avg", "min:i32": "min", "max:i32": "max",
}

_FUNCTION_URNS = {
    AI_FILTER_NAME: AI_EXTENSION_URN,
    AI_JOIN_NAME: AI_EXTENSION_URN,
    AI_CLASSIFY_NAME: AI_EXTENSION_URN,
    AI_CLASSIFY_JOINED_NAME: AI_EXTENSION_URN,
    AI_SCORE_NAME: AI_EXTENSION_URN,
    AI_EXTRACT_NAME: AI_EXTENSION_URN,
    EQUAL_NAME: COMPARISON_EXTENSION_URN,
    AND_NAME: BOOLEAN_EXTENSION_URN,
    **{name: COMPARISON_EXTENSION_URN for name in COMPARISON_NAMES},
    "count:any": AGGREGATE_GENERIC_EXTENSION_URN,
    **{name: ARITHMETIC_EXTENSION_URN for name in AGGREGATE_NAMES
       if name != "count:any"},
}


@dataclass(frozen=True)
class Relation:
    alias: str
    table: str
    text_column: str


@dataclass(frozen=True)
class Filter:
    id: str
    relation: str
    prompt: str


@dataclass(frozen=True)
class Join:
    """One ai_join call and the ordinary conditions beside it.

    Attributes:
        id: The operator ID.
        relations: The aliases of `{0}` and `{1}`.
        prompt: The prompt template.
        on: Equal column pairs, left column first.
        comparisons: Other column comparisons, as (left column,
            comparison, right column); for example ("id", "<", "id").
    """

    id: str
    relations: tuple[str, str]
    prompt: str
    on: tuple[tuple[str, str], ...] = ()
    comparisons: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class Classify:
    """One ai_classify call that adds a label column to one relation.

    A one-document classification labels each document of `relation`.
    A classification of joined rows labels each row of the join of
    `relation` and `partner`; its label column still belongs to
    `relation`.

    Attributes:
        id: The operator ID.
        relation: The alias of the classified documents, `{0}`.
        prompt: The prompt template, with the document as `{0}` and,
            for joined rows, the partner as `{1}`.
        labels: The categories, in the order that breaks ties.
        descriptions: One description per label; empty means none.
        output: The name of the added label column.
        partner: The alias of the partner documents, `{1}`, or None.
    """

    id: str
    relation: str
    prompt: str
    labels: tuple[str, ...]
    descriptions: tuple[str, ...]
    output: str
    partner: str | None = None

    @property
    def relations(self) -> tuple[str, ...]:
        """The classified aliases: the anchor, then the partner if any."""
        if self.partner is None:
            return (self.relation,)
        return (self.relation, self.partner)


@dataclass(frozen=True)
class InList:
    """Keep the documents whose label is one of the accepted labels."""

    id: str
    relation: str
    output: str
    accepted: tuple[str, ...]


@dataclass(frozen=True)
class Score:
    """One ai_score call that adds a number column to one relation.

    The prompt is a filter prompt; the number is the model's belief
    that the document answers it TRUE, between 0 and 1.
    """

    id: str
    relation: str
    prompt: str
    output: str


@dataclass(frozen=True)
class Extract:
    """One ai_extract call that adds a nullable string column to one relation.

    The prompt is the question; the answer is the words of the document
    that answer it, copied from the document, or null when the document
    does not answer it. A query returns the answer as a field of an
    AnswerStruct, named `output`.
    """

    id: str
    relation: str
    prompt: str
    output: str


@dataclass(frozen=True)
class AnswerStruct:
    """A result column that holds the answers of extractions as a struct.

    Attributes:
        name: The result column's name.
        fields: The selected answer fields, as `alias.column`, in struct
            field order. Each struct field is named by its column.
    """

    name: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class ColumnTest:
    """A test of one source column against a literal, before the model."""

    id: str
    relation: str
    column: str
    comparison: str
    value: int | float | str


@dataclass(frozen=True)
class Aggregate:
    """Group the rows by key columns and compute measures per group.

    Attributes:
        id: The operator ID.
        keys: The field names grouped by.
        measures: (name, function, argument field name or None for
            count(*)) per measure. No measures means DISTINCT.
    """

    id: str
    keys: tuple[str, ...]
    measures: tuple[tuple[str, str, str | None], ...]


@dataclass(frozen=True)
class Having:
    """Keep the groups whose measures pass (name, comparison, value) tests."""

    id: str
    tests: tuple[tuple[str, str, int | float], ...]


@dataclass(frozen=True)
class Sort:
    """Order the rows by (field name, descending) keys; nulls go last."""

    id: str
    keys: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class Fetch:
    """Skip `offset` rows and keep at most `count`."""

    id: str
    offset: int
    count: int


type Operator = (Filter | Join | Classify | InList | Score | ColumnTest
                 | Extract)
type Step = Aggregate | Having | Sort | Fetch


@dataclass(frozen=True)
class PlanInfo:
    """The benchmark reading of a plan.

    `operators` are the steps under the AI tree, in post-order.
    `tail` holds the relational steps between the AI tree and the
    root projection, bottom up: at most an aggregate, a having, a
    sort, and a fetch. `fields` names the columns the root projection
    selects from, as `alias.column` for source and label columns and a
    bare name for a measure. `select` lists the selected fields, the
    answer fields of `structs` included.
    """

    relations: tuple[Relation, ...]
    operators: tuple[Operator, ...]
    select: tuple[str, ...]
    tail: tuple[Step, ...] = ()
    fields: tuple[str, ...] = ()
    structs: tuple[AnswerStruct, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        """The result column names, in order: answers under their struct."""
        owner = {name: struct.name for struct in self.structs
                 for name in struct.fields}
        columns = []
        for name in self.select:
            column = owner.get(name, output_name(name))
            if column not in columns:
                columns.append(column)
        return tuple(columns)

    def struct_of(self, operator: "Extract") -> AnswerStruct:
        """Return the struct column that holds an extraction's answer."""
        name = f"{operator.relation}.{operator.output}"
        return next(struct for struct in self.structs if name in struct.fields)

    @property
    def filters(self) -> tuple[Filter, ...]:
        return tuple(
            operator
            for operator in self.operators
            if isinstance(operator, Filter)
        )

    @property
    def joins(self) -> tuple[Join, ...]:
        return tuple(
            operator
            for operator in self.operators
            if isinstance(operator, Join)
        )

    @property
    def ai_operators(
            self) -> tuple[Filter | Join | Classify | Score | Extract, ...]:
        """The operators that ask the model, each with its prompt."""
        return tuple(
            operator
            for operator in self.operators
            if not isinstance(operator, (InList, ColumnTest))
        )

    @property
    def scores(self) -> tuple[Score, ...]:
        return tuple(operator for operator in self.operators
                     if isinstance(operator, Score))

    @property
    def extracts(self) -> tuple[Extract, ...]:
        return tuple(operator for operator in self.operators
                     if isinstance(operator, Extract))

    @property
    def column_tests(self) -> tuple[ColumnTest, ...]:
        return tuple(operator for operator in self.operators
                     if isinstance(operator, ColumnTest))

    @property
    def relational(self) -> bool:
        """Whether the query is scored from its rows rather than per operator.

        A score or a tail step makes a query relational, and so does a
        WHERE condition over the one relation it reads. A WHERE condition on
        one relation of a join only limits the pairs the join asks.
        """
        return bool(self.scores or self.tail
                    or (self.column_tests and len(self.relations) == 1))

    @property
    def aggregate(self) -> Aggregate | None:
        return next((step for step in self.tail
                     if isinstance(step, Aggregate)), None)

    @property
    def having(self) -> Having | None:
        return next((step for step in self.tail
                     if isinstance(step, Having)), None)

    @property
    def sort(self) -> Sort | None:
        return next((step for step in self.tail
                     if isinstance(step, Sort)), None)

    @property
    def fetch(self) -> Fetch | None:
        return next((step for step in self.tail
                     if isinstance(step, Fetch)), None)

    @property
    def classifies(self) -> tuple[Classify, ...]:
        return tuple(
            operator
            for operator in self.operators
            if isinstance(operator, Classify)
        )

    @property
    def in_lists(self) -> tuple[InList, ...]:
        return tuple(
            operator
            for operator in self.operators
            if isinstance(operator, InList)
        )

    def classify_output(self, output: str) -> Classify:
        return next(
            operator for operator in self.classifies
            if operator.output == output
        )

    def extract_output(self, output: str) -> Extract:
        return next(
            operator for operator in self.extracts
            if operator.output == output
        )

    def pairing_join(self, operator: Classify) -> Join | None:
        """The join of exactly the two relations whose rows the call labels."""
        return next(
            (join for join in self.joins
             if set(join.relations) == set(operator.relations)),
            None)

    @property
    def base_alias(self) -> str:
        return self.relations[0].alias

    def relation(self, alias: str) -> Relation:
        return next(
            relation for relation in self.relations if relation.alias == alias
        )


@dataclass(frozen=True)
class _Decoded:
    fields: tuple[tuple[str, str], ...]
    tables: tuple[tuple[str, str], ...]
    text_columns: tuple[tuple[str, str], ...]
    operators: tuple[Operator, ...]


def _with_text_column(
    text_columns: tuple[tuple[str, str], ...],
    field: tuple[str, str],
) -> tuple[tuple[str, str], ...]:
    """Record the column an AI function reads from one relation."""
    alias, column = field
    known = dict(text_columns)
    if known.get(alias, column) != column:
        raise ValueError(f"relation {alias!r} is used with two text columns")
    if alias in known:
        return text_columns
    return (*text_columns, (alias, column))


def _function_names(plan: plan_pb2.Plan) -> dict[int, str]:
    urns = {}
    for declaration in plan.extension_urns:
        anchor = declaration.extension_urn_anchor
        if anchor in urns:
            raise ValueError(f"duplicate Substrait extension URN anchor {anchor}")
        urns[anchor] = declaration.urn
    names = {}
    for declaration in plan.extensions:
        if not declaration.HasField("extension_function"):
            raise ValueError("QUAIL-B plans support scalar functions only")
        function = declaration.extension_function
        if function.function_anchor in names:
            raise ValueError(
                f"duplicate Substrait function anchor {function.function_anchor}"
            )
        expected_urn = _FUNCTION_URNS.get(function.name)
        if expected_urn is None:
            raise ValueError(
                f"unsupported Substrait function {function.name!r}"
            )
        if urns.get(function.extension_urn_reference) != expected_urn:
            raise ValueError(
                f"Substrait function {function.name!r} has the wrong URN"
            )
        names[function.function_anchor] = function.name
    return names


def _selection_index(expression: algebra_pb2.Expression) -> int:
    if not expression.HasField("selection"):
        raise ValueError("Substrait argument must be a field selection")
    reference = expression.selection
    if not reference.HasField("direct_reference"):
        raise ValueError("Substrait field must use a direct reference")
    segment = reference.direct_reference
    if not segment.HasField("struct_field") or segment.struct_field.HasField(
        "child"
    ):
        raise ValueError("Substrait field must select one top-level column")
    return segment.struct_field.field


def _selected(
    fields: tuple[tuple[str, str], ...],
    expression: algebra_pb2.Expression,
) -> tuple[str, str]:
    index = _selection_index(expression)
    if index >= len(fields):
        raise ValueError(f"Substrait field index {index} is out of range")
    return fields[index]


def _arguments(
    expression: algebra_pb2.Expression,
) -> list[algebra_pb2.Expression]:
    if not expression.HasField("scalar_function"):
        raise ValueError("Substrait predicate must be a scalar function")
    return [
        argument.value for argument in expression.scalar_function.arguments
    ]


def _literal_string(expression: algebra_pb2.Expression) -> str:
    if not expression.HasField("literal"):
        raise ValueError("Substrait AI prompt must be a string literal")
    literal = expression.literal
    if literal.WhichOneof("literal_type") != "string":
        raise ValueError("Substrait AI prompt must be a string literal")
    return literal.string


def _string_list(expression: algebra_pb2.Expression) -> tuple[str, ...]:
    if (not expression.HasField("literal")
            or expression.literal.WhichOneof("literal_type") != "list"):
        raise ValueError("ai_classify labels must be a list literal")
    values = []
    for value in expression.literal.list.values:
        if value.WhichOneof("literal_type") != "string":
            raise ValueError("ai_classify labels must be strings")
        values.append(value.string)
    return tuple(values)


def _validate_labels(labels, descriptions) -> None:
    if len(labels) < 2:
        raise ValueError("ai_classify needs at least two labels")
    if len(descriptions) != len(labels):
        raise ValueError("ai_classify needs one description per label")
    folded = [" ".join(label.split()).casefold() for label in labels]
    if any(not label for label in folded) or len(set(folded)) != len(folded):
        raise ValueError("ai_classify labels must be nonempty and distinct")
    for label in folded:
        for other in folded:
            # the score of a label that continues another is never higher
            if other != label and other.startswith(label + " "):
                raise ValueError(
                    f"ai_classify label {other!r} extends label {label!r}")


def _flatten(
    expression: algebra_pb2.Expression,
    functions: dict[int, str],
) -> list[algebra_pb2.Expression]:
    if not expression.HasField("scalar_function"):
        return [expression]
    function = expression.scalar_function
    if functions.get(function.function_reference) != AND_NAME:
        return [expression]
    conditions = []
    for argument in function.arguments:
        conditions.extend(_flatten(argument.value, functions))
    return conditions


def _decode(
    rel: algebra_pb2.Rel,
    functions: dict[int, str],
) -> _Decoded:
    kind = rel.WhichOneof("rel_type")
    if kind == "read":
        read = rel.read
        alias = read.common.hint.alias
        if not alias:
            raise ValueError("QUAIL-B reads need a relation alias in hint.alias")
        if not read.HasField("named_table") or not read.named_table.names:
            raise ValueError("QUAIL-B reads need a named table")
        if len(read.base_schema.names) != len(read.base_schema.struct.types):
            raise ValueError("Substrait read schema names and types do not match")
        fields = tuple((alias, name) for name in read.base_schema.names)
        if (alias, "id") not in fields:
            raise ValueError("QUAIL-B reads need an id column")
        return _Decoded(fields, ((alias, read.named_table.names[-1]),), (), ())

    if kind == "project":
        return _decode_project(rel.project, functions)

    if kind == "filter" and rel.filter.condition.HasField("singular_or_list"):
        child = _decode(rel.filter.input, functions)
        membership = rel.filter.condition.singular_or_list
        alias, output = _selected(child.fields, membership.value)
        classify = next(
            (operator for operator in child.operators
             if isinstance(operator, Classify)
             and (operator.relation, operator.output) == (alias, output)),
            None)
        if classify is None:
            raise ValueError("an IN-list filter must test an ai_classify column")
        if classify.partner is not None:
            raise ValueError(
                "an IN-list filter tests a one-document classification")
        accepted = tuple(_literal_string(option) for option in membership.options)
        if not accepted or len(set(accepted)) != len(accepted):
            raise ValueError("an IN-list filter needs distinct accepted labels")
        if not set(accepted) <= set(classify.labels):
            raise ValueError("an IN-list filter accepts a label not in its call")
        operator = InList(
            rel.filter.common.hint.alias, alias, output, accepted)
        return _Decoded(child.fields, child.tables, child.text_columns,
                        (*child.operators, operator))

    if kind == "filter" and _comparison_name(rel.filter.condition, functions):
        child = _decode(rel.filter.input, functions)
        tests = []
        for condition in _flatten(rel.filter.condition, functions):
            comparison = _comparison_name(condition, functions)
            if comparison is None:
                raise ValueError(
                    "a WHERE condition FilterRel combines comparisons only")
            tests.append(_column_test(
                rel.filter.common.hint.alias, child.fields, condition,
                comparison))
        if len({test.relation for test in tests}) != 1:
            raise ValueError("a WHERE condition FilterRel tests one relation")
        if any(not isinstance(operator, ColumnTest)
               for operator in child.operators):
            raise ValueError(
                "a WHERE condition sits directly over its relation's scan")
        return _Decoded(child.fields, child.tables, child.text_columns,
                        (*child.operators, *tests))

    if kind == "filter":
        child = _decode(rel.filter.input, functions)
        condition = rel.filter.condition
        function = condition.scalar_function
        if functions.get(function.function_reference) != AI_FILTER_NAME:
            raise ValueError("QUAIL-B FilterRel must call ai_filter")
        arguments = _arguments(condition)
        if len(arguments) != 2:
            raise ValueError("ai_filter needs a prompt and document")
        prompt = _literal_string(arguments[0])
        field = _selected(child.fields, arguments[1])
        operator = Filter(rel.filter.common.hint.alias, field[0], prompt)
        return _Decoded(
            child.fields,
            child.tables,
            _with_text_column(child.text_columns, field),
            (*child.operators, operator),
        )

    if kind == "join":
        if rel.join.type != algebra_pb2.JoinRel.JOIN_TYPE_INNER:
            raise ValueError("QUAIL-B JoinRel must be an inner join")
        left = _decode(rel.join.left, functions)
        right = _decode(rel.join.right, functions)
        fields = left.fields + right.fields
        conditions = _flatten(rel.join.expression, functions)
        ai_conditions = [
            condition
            for condition in conditions
            if condition.HasField("scalar_function")
            and functions.get(
                condition.scalar_function.function_reference
            ) == AI_JOIN_NAME
        ]
        if len(ai_conditions) != 1:
            raise ValueError("QUAIL-B JoinRel needs one ai_join condition")
        arguments = _arguments(ai_conditions[0])
        if len(arguments) != 3:
            raise ValueError("ai_join needs a prompt and two documents")
        prompt = _literal_string(arguments[0])
        document_fields = (
            _selected(fields, arguments[1]),
            _selected(fields, arguments[2]),
        )
        relation_aliases = tuple(field[0] for field in document_fields)
        text_columns = (*left.text_columns, *right.text_columns)
        for field in document_fields:
            text_columns = _with_text_column(text_columns, field)
        on = []
        comparisons = []
        for condition in conditions:
            function = condition.scalar_function
            name = functions.get(function.function_reference)
            if name == AI_JOIN_NAME:
                continue
            if name not in COMPARISON_NAMES:
                raise ValueError("unsupported QUAIL-B join condition")
            arguments = _arguments(condition)
            if len(arguments) != 2:
                raise ValueError("a join condition compares two fields")
            first, second = (
                _selected(fields, expression) for expression in arguments
            )
            comparison = COMPARISON_NAMES[name]
            if (second[0], first[0]) == relation_aliases:
                first, second = second, first
                comparison = _REVERSED[comparison]
            elif (first[0], second[0]) != relation_aliases:
                raise ValueError("join condition uses unrelated relations")
            if comparison == "=":
                on.append((first[1], second[1]))
            else:
                comparisons.append((first[1], comparison, second[1]))
        operator = Join(
            rel.join.common.hint.alias,
            relation_aliases,
            prompt,
            tuple(on),
            tuple(comparisons),
        )
        return _Decoded(
            fields,
            (*left.tables, *right.tables),
            text_columns,
            (*left.operators, *right.operators, operator),
        )

    raise ValueError(f"unsupported QUAIL-B Substrait relation {kind!r}")


def _comparison_name(expression, functions) -> str | None:
    """Return the comparison a scalar call makes, or None for another call."""
    if not expression.HasField("scalar_function"):
        return None
    name = functions.get(expression.scalar_function.function_reference)
    if name == AND_NAME:
        inner = [_comparison_name(argument, functions)
                 for argument in _flatten(expression, functions)]
        return inner[0] if all(inner) else None
    return COMPARISON_NAMES.get(name)


def _literal_value(expression: algebra_pb2.Expression):
    """Return the Python value of a number or string literal."""
    if not expression.HasField("literal"):
        raise ValueError("a WHERE condition compares with a literal")
    literal = expression.literal
    kind = literal.WhichOneof("literal_type")
    if kind in ("i8", "i16", "i32", "i64", "fp32", "fp64", "string"):
        return getattr(literal, kind)
    raise ValueError(f"unsupported WHERE condition literal {kind!r}")


def _column_test(operator_id: str, fields, condition, comparison: str
                 ) -> ColumnTest:
    arguments = _arguments(condition)
    if len(arguments) != 2:
        raise ValueError("a WHERE condition compares a column with a literal")
    alias, column = _selected(fields, arguments[0])
    if column == "id":
        raise ValueError("a WHERE condition does not test the id column")
    return ColumnTest(operator_id, alias, column, comparison,
                       _literal_value(arguments[1]))


def _decode_project(project: algebra_pb2.ProjectRel,
                    functions: dict[int, str]) -> _Decoded:
    """Decode an inner ProjectRel: a classify label, a score, or an extract."""
    if len(project.expressions) == 1 and project.expressions[0].HasField(
            "scalar_function"):
        reference = project.expressions[0].scalar_function.function_reference
        if functions.get(reference) == AI_SCORE_NAME:
            return _decode_score(project, functions)
        if functions.get(reference) == AI_EXTRACT_NAME:
            return _decode_extract(project, functions)
    return _decode_classify(project, functions)


def _decode_extract(project: algebra_pb2.ProjectRel,
                    functions: dict[int, str]) -> _Decoded:
    """Decode a ProjectRel that adds one ai_extract answer column."""
    child = _decode(project.input, functions)
    if project.common.HasField("emit"):
        raise ValueError("an extract ProjectRel adds exactly one column")
    arguments = _arguments(project.expressions[0])
    if len(arguments) != 2:
        raise ValueError("ai_extract needs a question and a document field")
    prompt = _literal_string(arguments[0])
    field = _selected(child.fields, arguments[1])
    names = tuple(project.common.hint.output_names)
    expected = tuple(".".join(name) for name in child.fields)
    if len(names) != len(expected) + 1 or names[:-1] != expected:
        raise ValueError("an extract ProjectRel must name every output field")
    alias, _, output = names[-1].partition(".")
    if alias != field[0] or not output or "." in output:
        raise ValueError("an answer column must belong to the read relation")
    if output == "id" or (alias, output) in child.fields:
        raise ValueError(f"answer column {output!r} is already a field")
    operator = Extract(project.common.hint.alias, alias, prompt, output)
    return _Decoded(
        (*child.fields, (alias, output)),
        child.tables,
        _with_text_column(child.text_columns, field),
        (*child.operators, operator),
    )


def _decode_score(project: algebra_pb2.ProjectRel,
                  functions: dict[int, str]) -> _Decoded:
    """Decode a ProjectRel that adds one ai_score number column."""
    child = _decode(project.input, functions)
    if project.common.HasField("emit"):
        raise ValueError("a score ProjectRel adds exactly one column")
    arguments = _arguments(project.expressions[0])
    if len(arguments) != 2:
        raise ValueError("ai_score needs a prompt and a document field")
    prompt = _literal_string(arguments[0])
    field = _selected(child.fields, arguments[1])
    names = tuple(project.common.hint.output_names)
    expected = tuple(".".join(name) for name in child.fields)
    if len(names) != len(expected) + 1 or names[:-1] != expected:
        raise ValueError("a score ProjectRel must name every output field")
    alias, _, output = names[-1].partition(".")
    if alias != field[0] or not output or "." in output:
        raise ValueError("a score column must belong to the scored relation")
    if output == "id" or (alias, output) in child.fields:
        raise ValueError(f"score column {output!r} is already a field")
    operator = Score(project.common.hint.alias, alias, prompt, output)
    return _Decoded(
        (*child.fields, (alias, output)),
        child.tables,
        _with_text_column(child.text_columns, field),
        (*child.operators, operator),
    )


def _decode_classify(project: algebra_pb2.ProjectRel,
                     functions: dict[int, str]) -> _Decoded:
    """Decode a ProjectRel that adds one ai_classify label column.

    The four-argument call labels one document; the five-argument call
    labels a joined row, the anchor document then its partner, and
    needs a child whose fields hold both relations.
    """
    child = _decode(project.input, functions)
    if len(project.expressions) != 1 or project.common.HasField("emit"):
        raise ValueError("a classify ProjectRel adds exactly one column")
    expression = project.expressions[0]
    function = expression.scalar_function
    name = (functions.get(function.function_reference)
            if expression.HasField("scalar_function") else None)
    if name not in (AI_CLASSIFY_NAME, AI_CLASSIFY_JOINED_NAME):
        raise ValueError("an inner QUAIL-B ProjectRel must call ai_classify")
    arguments = _arguments(expression)
    documents = 2 if name == AI_CLASSIFY_JOINED_NAME else 1
    if len(arguments) != documents + 3:
        raise ValueError(
            f"{name} needs a prompt, {documents} document field(s), labels, "
            "and descriptions")
    prompt = _literal_string(arguments[0])
    fields = tuple(_selected(child.fields, argument)
                   for argument in arguments[1:1 + documents])
    labels = _string_list(arguments[1 + documents])
    descriptions = _string_list(arguments[2 + documents])
    _validate_labels(labels, descriptions)
    if len({field[0] for field in fields}) != documents:
        raise ValueError("a classification of joined rows reads two relations")
    names = tuple(project.common.hint.output_names)
    expected = tuple(".".join(name) for name in child.fields)
    if len(names) != len(expected) + 1 or names[:-1] != expected:
        raise ValueError("a classify ProjectRel must name every output field")
    alias, _, output = names[-1].partition(".")
    if alias != fields[0][0] or not output or "." in output:
        raise ValueError("a label column must belong to the classified relation")
    if output == "id" or (alias, output) in child.fields:
        raise ValueError(f"label column {output!r} is already a field")
    partner = fields[1][0] if documents == 2 else None
    operator = Classify(project.common.hint.alias, alias, prompt, labels,
                         descriptions, output, partner)
    text_columns = child.text_columns
    for field in fields:
        text_columns = _with_text_column(text_columns, field)
    return _Decoded(
        (*child.fields, (alias, output)),
        child.tables,
        text_columns,
        (*child.operators, operator),
    )


def _field_alias(name: str) -> str:
    alias, separator, column = name.partition(".")
    if not separator or not alias or not column or "." in column:
        raise ValueError(f"invalid field name {name!r}")
    return alias


def _decode_tail(rel: algebra_pb2.Rel, functions: dict[int, str]
                 ) -> tuple[_Decoded, tuple[Step, ...], tuple[str, ...]]:
    """Decode the relational steps between the root projection and the AI tree.

    Returns:
        The decoded AI tree, the steps bottom up, and the field names
        the root projection selects from.
    """
    kind = rel.WhichOneof("rel_type")
    if kind == "fetch":
        fetch = rel.fetch
        child, steps, fields = _decode_tail(fetch.input, functions)
        offset = _literal_value(fetch.offset_expr) if fetch.HasField(
            "offset_expr") else 0
        count = _literal_value(fetch.count_expr)
        if not isinstance(offset, int) or not isinstance(count, int) \
                or offset < 0 or count <= 0:
            raise ValueError("a fetch needs a nonnegative offset and a "
                             "positive count")
        return child, (*steps, Fetch(fetch.common.hint.alias, offset, count)), \
            fields
    if kind == "sort":
        sort = rel.sort
        child, steps, fields = _decode_tail(sort.input, functions)
        keys = []
        for item in sort.sorts:
            descending = item.direction in (
                algebra_pb2.SortField.SORT_DIRECTION_DESC_NULLS_FIRST,
                algebra_pb2.SortField.SORT_DIRECTION_DESC_NULLS_LAST)
            keys.append((fields[_selection_index(item.expr)], descending))
        if not keys:
            raise ValueError("a sort needs at least one key")
        return child, (*steps, Sort(sort.common.hint.alias, tuple(keys))), \
            fields
    if kind == "filter" and _comparison_name(rel.filter.condition, functions):
        child, steps, fields = _decode_tail(rel.filter.input, functions)
        if not steps or not isinstance(steps[-1], Aggregate):
            raise ValueError("a having FilterRel sits over an AggregateRel")
        tests = []
        for condition in _flatten(rel.filter.condition, functions):
            arguments = _arguments(condition)
            if len(arguments) != 2:
                raise ValueError("a having test compares a measure with a "
                                 "number")
            name = fields[_selection_index(arguments[0])]
            value = _literal_value(arguments[1])
            if isinstance(value, str):
                raise ValueError("a having test compares with a number")
            tests.append((name, _comparison_name(condition, functions), value))
        return child, (*steps, Having(rel.filter.common.hint.alias,
                                        tuple(tests))), fields
    if kind == "aggregate":
        aggregate = rel.aggregate
        child, steps, fields = _decode_tail(aggregate.input, functions)
        if steps:
            raise ValueError("an aggregate is the first relational step")
        if len(aggregate.groupings) != 1:
            raise ValueError("an aggregate has one grouping")
        keys = tuple(
            fields[_selection_index(aggregate.grouping_expressions[index])]
            for index in aggregate.groupings[0].expression_references)
        names = tuple(aggregate.common.hint.output_names)
        if len(names) != len(keys) + len(aggregate.measures):
            raise ValueError("an aggregate names each key and measure")
        measures = []
        for name, measure in zip(names[len(keys):], aggregate.measures):
            function = AGGREGATE_NAMES.get(
                functions.get(measure.measure.function_reference))
            if function is None:
                raise ValueError("unsupported aggregate function")
            arguments = [argument.value for argument in measure.measure.arguments]
            if function == "count" and measure.measure.invocation == (
                    algebra_pb2.AggregateFunction.AGGREGATION_INVOCATION_DISTINCT):
                function = "count_distinct"
            if not arguments and function != "count":
                raise ValueError(f"{function} needs an argument")
            argument = (fields[_selection_index(arguments[0])] if arguments
                        else None)
            if not name or "." in name:
                raise ValueError(f"a measure needs a bare name, got {name!r}")
            measures.append((name, function, argument))
        step = Aggregate(aggregate.common.hint.alias, keys, tuple(measures))
        return child, (step,), names
    decoded = _decode(rel, functions)
    return decoded, (), tuple(".".join(field) for field in decoded.fields)


def _validate_relational(info: PlanInfo) -> None:
    """Check the WHERE conditions, scores, tail, and projection of a relational plan."""
    if len(info.relations) != 1:
        raise ValueError("a relational QUAIL-B query reads one relation")
    kinds = [type(step) for step in info.tail]
    allowed = [Aggregate, Having, Sort, Fetch]
    order = [allowed.index(kind) for kind in kinds]
    if order != sorted(order) or len(set(order)) != len(order):
        raise ValueError("relational steps are at most one aggregate, "
                         "having, sort, and fetch, in that order")
    if info.having is not None and info.aggregate is None:
        raise ValueError("a having needs an aggregate")
    aggregate = info.aggregate
    if aggregate is not None:
        names = [*aggregate.keys, *(name for name, _, _ in aggregate.measures)]
        if len(set(names)) != len(names):
            raise ValueError("aggregate keys and measure names must be unique")
        for name, _, _ in aggregate.measures:
            if name in info.fields[:len(aggregate.keys)]:
                raise ValueError(f"measure {name!r} is also a key")
    for name, _comparison, _value in (
            info.having.tests if info.having is not None else ()):
        if aggregate is None or name not in {
                name for name, _, _ in aggregate.measures}:
            raise ValueError(f"having tests {name!r}, which is not a measure")
    if info.sort is not None:
        for name, _descending in info.sort.keys:
            if name not in info.fields:
                raise ValueError(f"sort key {name!r} is not a field")
    if not info.select or any(name not in info.fields for name in info.select):
        raise ValueError("a relational projection selects fields of its "
                         "last step")
    if len(set(info.select)) != len(info.select):
        raise ValueError("a projection selects each field once")
    outputs = [operator.output for operator in info.scores]
    if len(set(outputs)) != len(outputs):
        raise ValueError("score column names must be distinct")
    for score in info.scores:
        if info.fetch is None and info.aggregate is None and (
                f"{score.relation}.{score.output}" not in info.select):
            raise ValueError(f"score column {score.output!r} is never used")
    for extract in info.extracts:
        if f"{extract.relation}.{extract.output}" not in info.select:
            raise ValueError(
                f"answer column {extract.output!r} is never used")


def _validate_info(info: PlanInfo) -> None:
    aliases = [relation.alias for relation in info.relations]
    if not aliases or len(set(aliases)) != len(aliases):
        raise ValueError("QUAIL-B relation aliases must be nonempty and unique")
    operator_ids = [operator.id for operator in info.operators]
    if any(not operator_id for operator_id in operator_ids):
        raise ValueError("QUAIL-B operator IDs must be nonempty")
    if len(set(operator_ids)) != len(operator_ids):
        raise ValueError("QUAIL-B operator IDs must be unique")
    if info.relational:
        _validate_relational(info)
        return
    if len(info.relations) != len(info.joins) + 1:
        raise ValueError("QUAIL-B plan must join every relation")
    joined = {info.base_alias}
    for join in info.joins:
        referenced = set(join.relations)
        if len(referenced & joined) != 1 or len(referenced - joined) != 1:
            raise ValueError(f"join {join.id!r} must add one relation")
        joined.update(referenced)
    first_join = {alias: len(info.operators) for alias in aliases}
    for index, operator in enumerate(info.operators):
        if isinstance(operator, Join):
            for alias in operator.relations:
                first_join[alias] = min(first_join[alias], index)
    join_index = {join.id: index for index, join in enumerate(info.operators)
                  if isinstance(join, Join)}
    for index, operator in enumerate(info.operators):
        if isinstance(operator, Classify) and operator.partner is not None:
            # a classification of joined rows labels the rows a join kept
            join = info.pairing_join(operator)
            if join is None or index < join_index[join.id]:
                raise ValueError(
                    f"classification {operator.id!r} of joined rows must follow the "
                    "join of its two relations")
        elif (
            isinstance(operator, (Filter, Classify, InList, ColumnTest,
                                  Extract))
            and index > first_join[operator.relation]
        ):
            raise ValueError(
                f"filter {operator.id!r} must precede joins on its relation"
            )
    outputs = [operator.output for operator in (*info.classifies,
                                                *info.extracts)]
    if len(set(outputs)) != len(outputs) or set(outputs) & set(aliases):
        raise ValueError("label and answer column names must be distinct "
                         "from each other and from relation aliases")
    selected = {_field_alias(name) for name in info.select}
    if not selected or not selected <= set(aliases):
        raise ValueError("QUAIL-B projection uses an unknown relation")
    selected_ids = {
        _field_alias(name) for name in info.select if name.endswith(".id")}
    if selected_ids != selected:
        raise ValueError("a query selecting a label column must select its "
                         "relation's id")
    for name in info.select:
        alias, column = name.split(".", 1)
        if column == "id":
            continue
        operator = next(
            (operator for operator in (*info.classifies, *info.extracts)
             if (operator.relation, operator.output) == (alias, column)),
            None)
        if operator is None:
            raise ValueError(f"QUAIL-B projection selects {name!r}; only ids, "
                             "label columns, and answer columns can be "
                             "selected")
        if isinstance(operator, Classify) and not (
                set(operator.relations) <= selected_ids):
            raise ValueError(f"a query selecting the joined-row label {name!r} must "
                             "select both relations' ids")
    used = {operator.output for operator in info.in_lists}
    used.update(name.split(".", 1)[1] for name in info.select)
    for operator in info.classifies:
        if operator.output not in used:
            raise ValueError(
                f"label column {operator.output!r} is never used")
    for operator in info.extracts:
        if operator.output not in used:
            raise ValueError(
                f"answer column {operator.output!r} is never used")


def inspect_plan(plan: plan_pb2.Plan) -> PlanInfo:
    """Return the benchmark information in a supported Substrait plan."""
    version = plan.version
    if (
        version.major_number,
        version.minor_number,
        version.patch_number,
    ) != SUBSTRAIT_VERSION:
        raise ValueError(
            f"QUAIL-B requires Substrait {'.'.join(map(str, SUBSTRAIT_VERSION))}"
        )
    if len(plan.relations) != 1 or not plan.relations[0].HasField("root"):
        raise ValueError("QUAIL-B needs one Substrait root relation")
    root = plan.relations[0].root
    if not root.input.HasField("project"):
        raise ValueError("QUAIL-B root must contain a ProjectRel")
    project = root.input.project
    decoded, tail, fields = _decode_tail(project.input, _function_names(plan))
    select, structs, names = [], [], []
    for expression in project.expressions:
        if expression.HasField("nested"):
            if len(root.names) <= len(names):
                raise ValueError("QUAIL-B root names do not name every struct")
            struct = _answer_struct(expression, fields, root.names[len(names)])
            structs.append(struct)
            select.extend(struct.fields)
            names.extend((struct.name, *map(output_name, struct.fields)))
            continue
        name = fields[_selection_index(expression)]
        select.append(name)
        names.append(output_name(name))
    expected_mapping = tuple(
        len(fields) + index
        for index in range(len(project.expressions))
    )
    if tuple(project.common.emit.output_mapping) != expected_mapping:
        raise ValueError("QUAIL-B ProjectRel has an invalid output mapping")
    if tuple(root.names) != tuple(names):
        raise ValueError("QUAIL-B root names do not match selected relations")
    text_columns = dict(decoded.text_columns)
    relations = []
    for alias, table in decoded.tables:
        if alias not in text_columns:
            raise ValueError(f"relation {alias!r} is not read by an AI function")
        relations.append(Relation(alias, table, text_columns[alias]))
    info = PlanInfo(tuple(relations), decoded.operators, tuple(select), tail,
                    fields, tuple(structs))
    _validate_info(info)
    _validate_structs(info)
    return info


def _answer_struct(expression: algebra_pb2.Expression, fields: tuple[str, ...],
                   name: str) -> AnswerStruct:
    """Decode a root struct expression over selected answer fields."""
    nested = expression.nested
    if nested.WhichOneof("nested_type") != "struct" or not nested.struct.fields:
        raise ValueError("a QUAIL-B root struct holds at least one field")
    return AnswerStruct(name, tuple(
        fields[_selection_index(field)] for field in nested.struct.fields))


def _validate_structs(info: PlanInfo) -> None:
    """Check that the answer structs hold every returned answer, and only them."""
    answers = {f"{operator.relation}.{operator.output}": operator
               for operator in info.extracts}
    held = [name for struct in info.structs for name in struct.fields]
    if len(set(held)) != len(held):
        raise ValueError("a QUAIL-B answer belongs to one struct")
    for struct in info.structs:
        if not struct.name or "." in struct.name:
            raise ValueError(f"a struct needs a bare name, got {struct.name!r}")
        if any(name not in answers for name in struct.fields):
            raise ValueError(f"struct {struct.name!r} holds only answer columns")
        if len({answers[name].relation for name in struct.fields}) != 1:
            raise ValueError(f"struct {struct.name!r} holds answers of one relation")
    for name in info.select:
        if name in answers and name not in held:
            raise ValueError(f"answer column {name!r} is returned in a struct")
    columns = info.columns
    plain = [output_name(name) for name in info.select if name not in held]
    if len(columns) != len(plain) + len(info.structs):
        raise ValueError("QUAIL-B result column names must be unique")


def output_name(field: str) -> str:
    """Return the result column name of a selected field.

    An id column is named by its alias, another column by its name
    after the alias, and a measure by its bare name.
    """
    alias, separator, column = field.partition(".")
    if not separator:
        return field
    return alias if column == "id" else column
