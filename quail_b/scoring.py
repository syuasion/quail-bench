"""Score one engine's run of one query against the saved labels.

The engine's runner hands over a `RunOutput`: every predicate answer it
produced, keyed by the query's operator IDs, and the final rows.
Everything here is in terms of the benchmark's own ids, so no engine
object is needed to score a run.
"""

from __future__ import annotations

import re
import string
import tempfile
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from quail_b._keys import lookup, tuple_keys
from quail_b.data import _ids
from quail_b.queries import QuerySpec
from quail_b.relational import relational_accuracy


@dataclass
class RunOutput:
    """What one engine returned for one query.

    Attributes:
        filter_answers: Filter operator ID to a table with the relation's
            alias column and a boolean `answer` column, one row per document
            the engine asked about.
        join_answers: Join operator ID to a table with one ID column
            per joined relation and a boolean `answer` column, one row per
            evaluated tuple.
        rows: The final rows: one ID column per selected alias, named by
            the alias, and one string column per selected label column,
            named by the query. None when a saved run is rescored from
            its answers alone.
        runtime_s: Completed query execution time, excluding result collection.
        measurements: Engine-reported numbers:

            - `fresh_tokens` counts positions processed by model forward
              passes. It is required with `prompt_pieces`.
            - `input_tokens` counts the complete inputs of all evaluated
              prompts, including positions read from KV. Report it when
              `prompt_pieces` is unavailable.
            - Other values are optional.
        prompt_pieces: The prompt token ids around each document, as
            `quail_b.minimum.validate_prompt_pieces` describes, or None.
            With the answers and `fresh_tokens`, scoring fills
            `input_tokens`, `input_tokens_per_second`, `minimum_tokens`,
            `regret_tokens`, and the minimum and regret by text.
        classify_answers: Classify operator ID to a table with the
            relation's alias column, the partner's alias column for a
            classification of joined rows, and a string `label`
            column, one row per document or joined row the engine
            classified. A document or joined row with no row cannot
            pass an IN-list filter or appear with its label.
        score_answers: Score operator ID to a table with the relation's
            alias column and a float `score` column, one row per
            document the engine scored. Token metrics need it for a
            query with a score.
        extract_answers: Extract operator ID to a table with the
            relation's alias column and a string `answer` column, null
            where the engine found no answer, one row per document the
            engine read. Answer accuracy and token metrics need it for
            a query with an extraction.
    """

    filter_answers: dict[str, pa.Table] | None
    join_answers: dict[str, pa.Table] | None
    rows: pa.Table
    runtime_s: float | None = None
    measurements: dict = field(default_factory=dict)
    prompt_pieces: dict | None = None
    classify_answers: dict[str, pa.Table] | None = None
    score_answers: dict[str, pa.Table] | None = None
    extract_answers: dict[str, pa.Table] | None = None


def output_columns(spec: QuerySpec) -> list[str]:
    """Return the result column names: aliases for ids, label and struct names."""
    return list(spec.info.columns)


def scored_columns(spec: QuerySpec) -> list[str]:
    """Return the result columns the rows are compared on: not an answer.

    An extracted answer is scored against its reference spans from the
    answer table, not as part of a result row.
    """
    answers = {struct.name for struct in spec.info.structs}
    return [name for name in output_columns(spec) if name not in answers]


def _selected_labels(spec: QuerySpec):
    """Return the classify operators whose label column the query returns."""
    selected = set(output_columns(spec))
    return [operator for operator in spec.info.classifies
            if operator.output in selected]


def selected_extracts(spec: QuerySpec):
    """Return the extract operators whose answer the query returns."""
    selected = set(spec.info.select)
    return [operator for operator in spec.info.extracts
            if f"{operator.relation}.{operator.output}" in selected]

@dataclass
class BinaryCounts:
    correct: int = 0
    evaluated: int = 0
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def add(self, predicted: bool, expected: bool) -> None:
        self.evaluated += 1
        self.correct += int(predicted == expected)
        if predicted and expected:
            self.true_positive += 1
        elif predicted:
            self.false_positive += 1
        elif expected:
            self.false_negative += 1
        else:
            self.true_negative += 1

    def merge(self, other: "BinaryCounts") -> None:
        for name in ("correct", "evaluated", "true_positive",
                     "true_negative", "false_positive", "false_negative"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> dict:
        accuracy = self.correct / self.evaluated if self.evaluated else 0.0
        pden = self.true_positive + self.false_positive
        rden = self.true_positive + self.false_negative
        precision = self.true_positive / pden if pden else 0.0
        recall = self.true_positive / rden if rden else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        return {
            "evaluated": self.evaluated,
            "correct": self.correct,
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
        }


@dataclass
class LabelCounts:
    """Agreement of classification answers with the reference labels.

    Attributes:
        correct: Answers that match their reference label.
        evaluated: Answers with a reference label.
        unlabeled: Answers over pairs the reference join does not keep,
            which have no reference label and are left out.
    """

    correct: int = 0
    evaluated: int = 0
    unlabeled: int = 0

    def merge(self, other: "LabelCounts") -> None:
        self.correct += other.correct
        self.evaluated += other.evaluated
        self.unlabeled += other.unlabeled

    def as_dict(self) -> dict:
        return {
            "correct": self.correct,
            "evaluated": self.evaluated,
            "unlabeled": self.unlabeled,
            "accuracy": (round(self.correct / self.evaluated, 6)
                         if self.evaluated else None),
        }


def normalize_answer(text: str) -> list[str]:
    """Return an answer's words for comparison, as SQuAD normalizes them.

    Case, punctuation, the articles a, an, and the, and extra spaces
    are dropped.
    """
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def span_scores(answer: str | None, references) -> tuple[float, float]:
    """Return (exact match, token F1) of one answer against reference spans.

    A document without a reference span is answered right by null
    alone; one with spans is answered against its best-matching span.
    """
    references = list(references)
    if answer is None:
        return (1.0, 1.0) if not references else (0.0, 0.0)
    if not references:
        return 0.0, 0.0
    predicted = normalize_answer(answer)
    best = (0.0, 0.0)
    for reference in references:
        expected = normalize_answer(reference)
        exact = float(predicted == expected)
        common = sum((Counter(predicted) & Counter(expected)).values())
        if not predicted or not expected:
            f1 = float(predicted == expected)
        elif common == 0:
            f1 = 0.0
        else:
            precision = common / len(predicted)
            recall = common / len(expected)
            f1 = 2 * precision * recall / (precision + recall)
        best = max(best, (exact, f1))
    return best


@dataclass
class SpanCounts:
    """Agreement of extraction answers with the reference spans.

    Attributes:
        evaluated: Answers with a reference row, null answers included.
        exact: Answers that match a reference span word for word, or
            are null where no span exists.
        f1_total: The sum of the answers' token F1 scores.
    """

    evaluated: int = 0
    exact: int = 0
    f1_total: float = 0.0

    def merge(self, other: "SpanCounts") -> None:
        self.evaluated += other.evaluated
        self.exact += other.exact
        self.f1_total += other.f1_total

    def as_dict(self) -> dict:
        return {
            "evaluated": self.evaluated,
            "exact": self.exact,
            "exact_match": (round(self.exact / self.evaluated, 6)
                            if self.evaluated else None),
            "token_f1": (round(self.f1_total / self.evaluated, 6)
                         if self.evaluated else None),
        }


@dataclass
class _PredicateCount:
    predicate_key: str
    op: str
    alias: str | None = None
    counts: BinaryCounts = field(default_factory=BinaryCounts)

    def as_dict(self) -> dict:
        return {
            "predicate_key": self.predicate_key,
            "op": self.op,
            "alias": self.alias,
            **self.counts.as_dict(),
        }


def _row_metrics(predicted_count: int, expected_count: int,
                 matched: int) -> dict:
    if not predicted_count and not expected_count:
        precision = recall = f1 = 1.0
    else:
        precision = matched / predicted_count if predicted_count else 0.0
        recall = matched / expected_count if expected_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
    return {
        "predicted_rows": predicted_count,
        "expected_rows": expected_count,
        "matching_rows": matched,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_match": (predicted_count == expected_count == matched),
        "false_positive_rows": predicted_count - matched,
        "false_negative_rows": expected_count - matched,
    }


def _id_table(columns: dict[str, list]) -> pa.Table:
    return pa.table({
        name: pa.array([str(value) for value in values], type=pa.string())
        for name, values in columns.items()
    })


def _distinct(table: pa.Table) -> pa.Table:
    if table.num_rows == 0:
        return table
    return table.group_by(table.column_names).aggregate([])


def _as_string_ids(table: pa.Table, aliases) -> pa.Table:
    return _distinct(pa.table({
        alias: table.column(alias).cast(pa.string()) for alias in aliases
    }))


def corpus_ids(spec: QuerySpec, corpus_rows) -> dict:
    """Per alias, the distinct corpus ids as strings, in one fixed order."""
    return {relation.alias: pc.unique(pa.array(
        [str(row_id) for row_id in _ids(corpus_rows[relation.table])],
        type=pa.string()))
        for relation in spec.info.relations}


def encode_ids(table: pa.Table, aliases, references: dict) -> pa.Table:
    """Replace each id column by its position in the alias's corpus ids.

    An id not in the corpus becomes null. The corpus ids are cast to
    the column's type when that works (integer ids stored as integers),
    so a result of hundreds of millions of rows is never cast to
    strings; only the small corpus side is.
    """
    columns = {}
    for alias in aliases:
        column = table.column(alias)
        reference = references[alias]
        if column.type != reference.type:
            try:
                reference = reference.cast(column.type)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                column = column.cast(pa.string())
        columns[alias] = pc.index_in(column, value_set=reference)
    return pa.table(columns)


def _join_all(tables: list[pa.Table]) -> pa.Table:
    """Inner join relations that share alias columns, in any order."""
    pending = list(tables)
    joined = pending.pop(0)
    while pending:
        for index, table in enumerate(pending):
            shared = sorted(set(joined.column_names) & set(table.column_names))
            if shared:
                joined = joined.join(table, keys=shared, join_type="inner")
                pending.pop(index)
                break
        else:
            raise ValueError("AI join relations form a disconnected graph")
    return joined


def _column_by_id(rows, column: str, ids) -> pa.Array:
    """Return a corpus column's value for each id, null for an unknown id."""
    if isinstance(rows, pa.Table):
        values = rows.column(column)
    else:
        values = pa.array([row[column] for row in rows])
    keys = pa.array([str(row_id) for row_id in _ids(rows)], pa.string())
    return pc.take(values, pc.index_in(ids, value_set=keys))


def _apply_conditions(table: pa.Table, join, spec: QuerySpec,
                      corpus_rows) -> pa.Table:
    """Keep the pairs of string ids that satisfy the join's column conditions.

    corpus_rows is needed only when the join has such conditions.
    """
    conditions = (*((left, "=", right) for left, right in join.on),
                  *join.comparisons)
    if not conditions:
        return table
    if corpus_rows is None:
        raise ValueError(
            f"{spec.id}: the join on {conditions} needs the corpus rows")
    left, right = join.relations
    left_rows = corpus_rows[spec.info.relation(left).table]
    right_rows = corpus_rows[spec.info.relation(right).table]
    mask = pa.array([True] * table.num_rows, pa.bool_())
    for left_column, comparison, right_column in conditions:
        passed = _COMPARE[comparison](
            _column_by_id(left_rows, left_column, table.column(left)),
            _column_by_id(right_rows, right_column, table.column(right)))
        mask = pc.and_(mask, pc.fill_null(passed, False))
    return table.filter(mask)


def _label_table(table: pa.Table, aliases, output: str) -> pa.Table:
    """Return an answer table as distinct string (aliases..., output) rows."""
    return _distinct(pa.table({
        **{alias: pc.cast(table.column(alias), pa.string())
           for alias in aliases},
        output: pc.cast(table.column("label"), pa.string()),
    }))


def answer_labels(spec: QuerySpec, classify_answers) -> dict | None:
    """Return label column name to the engine's (aliases..., label) rows."""
    if spec.info.classifies and classify_answers is None:
        return None
    labels = {}
    for operator in spec.info.classifies:
        table = classify_answers.get(operator.id)
        if table is None:
            return None
        labels[operator.output] = _label_table(
            table, operator.relations, operator.output)
    return labels


def _narrow(survivors: dict, alias: str, passed) -> None:
    kept = survivors[alias]
    survivors[alias] = set(passed) if kept is None else kept & set(passed)


def _apply_labels(spec: QuerySpec, survivors: dict, labels: dict) -> None:
    """Keep ids that have every used label and pass every IN-list filter.

    A label of joined rows never narrows an alias:
    `_apply_joined_labels` narrows the join's pairs instead.
    """
    used = {operator.output for operator in spec.info.in_lists}
    used.update(operator.output for operator in _selected_labels(spec))
    for operator in spec.info.classifies:
        if operator.output in used and operator.partner is None:
            table = labels[operator.output]
            _narrow(survivors, operator.relation,
                    table.column(operator.relation).to_pylist())
    for in_list in spec.info.in_lists:
        table = labels[in_list.output]
        mask = pc.is_in(table.column(in_list.output),
                        value_set=pa.array(in_list.accepted, pa.string()))
        _narrow(survivors, in_list.relation,
                table.filter(mask).column(in_list.relation).to_pylist())


def _joined_classifies(spec: QuerySpec, join):
    """Return the classifications of the rows of one join."""
    return [operator for operator in spec.info.classifies
            if operator.partner is not None
            and spec.info.pairing_join(operator) is join]


def _apply_joined_labels(spec: QuerySpec, join, pairs: pa.Table,
                       labels: dict) -> pa.Table:
    """Keep the join's pairs that have every joined-row label the query returns."""
    for operator in _joined_classifies(spec, join):
        pairs = pairs.join(labels[operator.output],
                           keys=list(operator.relations), join_type="left semi")
    return pairs


def answer_relations(spec: QuerySpec, filter_answers, join_answers,
                     corpus_rows=None, classify_answers=None,
                     extract_answers=None):
    """Return (survivors, relations) an engine's own answers imply.

    survivors maps each alias to the string ids that passed every filter
    and IN-list filter on it and have every label and answer column the
    query uses, or None when nothing narrows it. relations holds one table per join,
    in written order, with the pairs that answered TRUE, satisfy the
    join's equality conditions, survived, and have every joined-row label
    the query returns. corpus_rows is needed only
    when a join has equality conditions. Returns None when the answers
    are missing.
    """
    if join_answers is None or filter_answers is None:
        return None
    labels = answer_labels(spec, classify_answers)
    if labels is None:
        return None
    if spec.info.extracts and extract_answers is None:
        return None
    survivors = {
        relation.alias: None for relation in spec.info.relations
    }
    for filter_spec in spec.info.filters:
        table = filter_answers.get(filter_spec.id)
        if table is None:
            return None
        alias = filter_spec.relation
        passed = {
            str(row_id)
            for row_id, answer in zip(table.column(alias).to_pylist(),
                                      table.column("answer").to_pylist())
            if answer
        }
        kept = survivors[alias]
        survivors[alias] = passed if kept is None else kept & passed
    _apply_labels(spec, survivors, labels)
    # a document with no answer row cannot appear with its answer column
    for operator in selected_extracts(spec):
        table = extract_answers.get(operator.id)
        if table is None:
            return None
        _narrow(survivors, operator.relation,
                pc.cast(table.column(operator.relation), pa.string()).to_pylist())
    relations = []
    for join in spec.info.joins:
        table = join_answers.get(join.id)
        if table is None:
            return None
        mask = table.column("answer")
        true_pairs = _as_string_ids(table.filter(mask), list(join.relations))
        true_pairs = _apply_conditions(true_pairs, join, spec, corpus_rows)
        for alias in join.relations:
            if survivors[alias] is not None:
                true_pairs = true_pairs.filter(pc.is_in(
                    true_pairs.column(alias),
                    value_set=pa.array(sorted(survivors[alias]), pa.string())))
        relations.append(_apply_joined_labels(spec, join, true_pairs, labels))
    return survivors, relations


def _with_labels(rows: pa.Table, spec: QuerySpec, labels: dict) -> pa.Table:
    """Add the returned label columns to id rows and keep output columns."""
    for operator in _selected_labels(spec):
        rows = rows.join(labels[operator.output],
                         keys=list(operator.relations), join_type="inner")
    return _distinct(rows.select(sorted(scored_columns(spec))))


def rows_from_answers(spec: QuerySpec, filter_answers, join_answers,
                      corpus_rows=None, classify_answers=None,
                      extract_answers=None) -> pa.Table:
    """Return the final rows an engine's own answers imply.

    For a run that saved its predicate answers but not its rows: a row
    survives when every filter on its alias answered TRUE, every label
    filter accepted its label, every join it takes part in answered
    TRUE, and every join equality holds. corpus_rows is needed only
    when a join has equality conditions.
    """
    survivors, relations = answer_relations(
        spec, filter_answers, join_answers, corpus_rows, classify_answers,
        extract_answers)
    if relations:
        rows = _join_all(relations)
    else:
        alias = spec.info.base_alias
        rows = _id_table({alias: sorted(survivors[alias])})
    return _with_labels(rows, spec, answer_labels(spec, classify_answers))


def implied_row_count(spec: QuerySpec, survivors: dict, relations: list) -> int:
    """Count the rows the answers imply without building them.

    The relations are joined in the order _join_all uses, but each step
    keeps only the aliases a later relation still needs, with a weight
    per row that sums the rows it stands for. Every joined alias is a
    selected column, so the join's size is the distinct row count.
    """
    if not relations:
        return len(survivors[spec.info.base_alias])
    pending = list(relations)
    current = pending.pop(0)
    current = current.append_column(
        "weight", pa.array([1] * current.num_rows, pa.int64()))
    while pending:
        for index, table in enumerate(pending):
            shared = sorted((set(current.column_names) - {"weight"})
                            & set(table.column_names))
            if shared:
                pending.pop(index)
                break
        else:
            raise ValueError("AI join relations form a disconnected graph")
        joined = current.join(table, keys=shared, join_type="inner")
        needed = {alias for later in pending for alias in later.column_names}
        keep = sorted((set(joined.column_names) - {"weight"}) & needed)
        if keep:
            current = joined.group_by(keep).aggregate([("weight", "sum")])
            current = current.rename_columns(keep + ["weight"])
        else:
            current = joined
    return pc.sum(current.column("weight")).as_py() or 0


def _keys(table: pa.Table, columns) -> pa.ChunkedArray:
    """Join string columns into one key per row, for set membership."""
    return pc.binary_join_element_wise(
        *(table.column(column) for column in columns), "\x1f")


def implied_rows_mask(rows: pa.Table, survivors: dict, relations: list,
                      spec: QuerySpec, labels: dict | None = None
                      ) -> pa.ChunkedArray:
    """Per row of strings, whether the answers imply it.

    Args:
        rows: One string column per selected alias and label column.
        survivors: As answer_relations returns them.
        relations: As answer_relations returns them.
        spec: The query.
        labels: As answer_labels returns them; needed when the query
            returns a label column.
    """
    mask = pa.array([True] * rows.num_rows, pa.bool_())
    for operator in _selected_labels(spec):
        columns = [*operator.relations, operator.output]
        pairs = _keys(rows, columns)
        known = _keys(labels[operator.output], columns)
        mask = pc.and_(mask, pc.is_in(pairs, value_set=known))
    aliases = {relation.alias for relation in spec.info.relations}
    for alias in (name for name in rows.column_names if name in aliases):
        if survivors.get(alias) is not None:
            mask = pc.and_(mask, pc.is_in(
                rows.column(alias),
                value_set=pa.array(sorted(survivors[alias]), pa.string())))
    for join, relation in zip(spec.info.joins, relations):
        left, right = join.relations
        pairs = pc.binary_join_element_wise(
            rows.column(left), rows.column(right), "\x1f")
        known = pc.binary_join_element_wise(
            relation.column(left), relation.column(right), "\x1f")
        mask = pc.and_(mask, pc.is_in(pairs, value_set=pc.unique(known)))
    if not relations:
        alias = spec.info.base_alias
        base = pa.array(sorted(survivors[alias]), pa.string())
        mask = pc.and_(mask, pc.is_in(rows.column(alias),
                                      value_set=base))
    return mask


def scores_from_answers(spec: QuerySpec, output: RunOutput, corpus_rows):
    """Return (survivors, relations) when the answers can score the rows.

    That needs every answer the query produces, and the selected columns
    must hold every joined alias, so that a row is a join tuple.
    """
    answers = answer_relations(
        spec, output.filter_answers, output.join_answers, corpus_rows,
        output.classify_answers, output.extract_answers)
    if answers is None:
        return None
    selected = {name.split(".")[0] for name in spec.info.select}
    joined = {
        alias for join in spec.info.joins for alias in join.relations
    }
    if not joined <= selected or spec.info.base_alias not in selected:
        return None
    return answers


def reference_answer(ground_truth, template: str, ids: tuple[str, ...]) -> bool:
    """Return the saved label of one prompt template for one or two ids."""
    key = ground_truth.key_for_template(template)
    if len(ids) == 1:
        return ground_truth.answer(key, ids[0])
    if len(ids) == 2:
        return ground_truth.answer(key, ids[0], ids[1])
    raise NotImplementedError(
        "ground truth evaluation supports one or two prompt arguments")


def _labels(ground_truth, template: str):
    """Return the labels of one prompt template as written."""
    return ground_truth.predicates[ground_truth.key_for_template(template)]


def _extract_labels(ground_truth, operator):
    """Return the reference spans of one extract operator."""
    labels = _labels(ground_truth, operator.prompt)
    if labels.kind != "extract":
        raise ValueError(
            f"{labels.key} is not an extraction, as {operator.id} is")
    return labels


def _classify_labels(ground_truth, operator):
    """Return the reference labels of one classify operator."""
    labels = _labels(ground_truth, operator.prompt)
    if (labels.kind != "classify"
            or tuple(labels.predicate["labels"]) != operator.labels):
        raise ValueError(
            f"{labels.key} is not a classification with the labels of "
            f"{operator.id}")
    return labels


def expected_labels(spec: QuerySpec, ground_truth) -> dict:
    """Return label column name to the reference (aliases..., label) rows.

    A one-document classification's rows drop the reference table's
    null `right_id`; a classification of joined rows keeps both ids
    under the anchor and partner aliases.
    """
    labels = {}
    for operator in spec.info.classifies:
        reference = _classify_labels(ground_truth, operator).table
        names = [operator.relation, operator.partner or "right_id", "label"]
        labels[operator.output] = _label_table(
            reference.rename_columns(names), operator.relations,
            operator.output)
    return labels


def _expected_pairs(spec: QuerySpec, ground_truth, corpus_rows
                    ) -> list[pa.Table]:
    """Return, per join, the reference TRUE pairs that satisfy its equalities."""
    relations = []
    for join in spec.info.joins:
        pairs = _labels(ground_truth, join.prompt).true_pairs
        pairs = pairs.select(["left_id", "right_id"]).rename_columns(
            list(join.relations))
        relations.append(_apply_conditions(pairs, join, spec, corpus_rows))
    return relations


def _check_joined_labels(spec: QuerySpec, ground_truth, survivors: dict,
                       pairs: list[pa.Table]) -> None:
    """Check that every surviving reference pair has each joined-row label.

    Args:
        spec: The query.
        ground_truth: The label collection.
        survivors: Per alias, the ids that pass every filter on it.
        pairs: Per join, the reference pairs, as `_expected_pairs`.

    Raises:
        KeyError: A pair the query would return has no reference label.
    """
    for join, table in zip(spec.info.joins, pairs):
        for operator in _joined_classifies(spec, join):
            labels = _classify_labels(ground_truth, operator)
            for alias in operator.relations:
                table = table.filter(pc.is_in(
                    table.column(alias),
                    value_set=pa.array(survivors[alias], pa.string())))
            known = _keys(labels.table, ["left_id", "right_id"])
            unknown = pc.invert(pc.is_in(
                _keys(table, list(operator.relations)), value_set=known))
            if pc.any(unknown).as_py():
                raise KeyError(
                    f"no ground truth for {labels.key} and "
                    f"{pc.sum(unknown).as_py()} pairs of {join.id}")


_COMPARE = {
    "=": pc.equal, "<>": pc.not_equal, "<": pc.less, "<=": pc.less_equal,
    ">": pc.greater, ">=": pc.greater_equal,
}


def expected_survivors(spec: QuerySpec, ground_truth, corpus_rows
                       ) -> dict[str, list[str]]:
    """Return, per alias, the ids that pass every WHERE condition and filter on it."""
    survivors = {}
    for relation in spec.info.relations:
        rows = corpus_rows[relation.table]
        ids = pa.array([str(row_id) for row_id in _ids(rows)], pa.string())
        for test in spec.info.column_tests:
            if test.relation != relation.alias:
                continue
            passed = _COMPARE[test.comparison](
                _column_by_id(rows, test.column, ids), test.value)
            ids = ids.filter(pc.fill_null(passed, False))
        for filter_spec in spec.info.filters:
            if filter_spec.relation != relation.alias:
                continue
            labels = _labels(ground_truth, filter_spec.prompt)
            unknown = pc.invert(pc.is_in(
                ids, value_set=labels.table.column("left_id")))
            if pc.any(unknown).as_py():
                raise KeyError(
                    f"no ground truth for {labels.key} and "
                    f"{pc.sum(unknown).as_py()} rows of {relation.table}")
            ids = ids.filter(pc.is_in(
                ids, value_set=labels.true_pairs.column("left_id")))
        survivors[relation.alias] = ids.to_pylist()
    for operator in spec.info.classifies:
        if operator.partner is not None:
            continue
        labels = _classify_labels(ground_truth, operator)
        ids = pa.array(survivors[operator.relation], pa.string())
        unknown = pc.invert(pc.is_in(
            ids, value_set=labels.table.column("left_id")))
        if pc.any(unknown).as_py():
            raise KeyError(
                f"no ground truth for {labels.key} and "
                f"{pc.sum(unknown).as_py()} rows")
    kept = {alias: set(ids) for alias, ids in survivors.items()}
    _apply_labels(spec, kept, expected_labels(spec, ground_truth))
    return {alias: [row_id for row_id in ids if row_id in kept[alias]]
            for alias, ids in survivors.items()}


def expected_rows(spec: QuerySpec, ground_truth, corpus_rows) -> pa.Table:
    """Return the final rows the labels say the query should return.

    Raises:
        KeyError: A document or pair the query returns has no label.
    """
    survivors = expected_survivors(spec, ground_truth, corpus_rows)
    relations = _expected_pairs(spec, ground_truth, corpus_rows)
    _check_joined_labels(spec, ground_truth, survivors, relations)
    if relations:
        rows = _join_all(relations)
    else:
        alias = spec.info.base_alias
        rows = _id_table({alias: survivors[alias]})
    for alias in rows.column_names:
        rows = rows.join(
            _id_table({alias: survivors[alias]}),
            keys=[alias], join_type="inner")
    return _with_labels(rows, spec, expected_labels(spec, ground_truth))


def label_agreement(table: pa.Table, aliases, labels) -> LabelCounts:
    """Count a classify answer table's agreement with its reference labels.

    Args:
        table: One id column per alias and a string `label` column.
        aliases: The id columns: the anchor, then a pair's partner.
        labels: The `PredicateLabels` of the classification.

    Returns:
        The counts. A pair the reference join does not keep has no
        reference label; such a classified pair counts as unlabeled.

    Raises:
        KeyError: A document the engine classified has no label.
    """
    aliases = list(aliases)
    answers = _label_table(table, aliases, "predicted")
    reference = labels.table.select(
        [*("left_id", "right_id")[:len(aliases)], "label"]
    ).rename_columns([*aliases, "expected"])
    joined = answers.join(reference, keys=aliases, join_type="left outer")
    expected = joined.column("expected")
    unlabeled = expected.null_count
    if unlabeled and len(aliases) == 1:
        raise KeyError(
            f"no ground truth for {labels.key} and {unlabeled} "
            "classified rows")
    if unlabeled:
        joined = joined.filter(pc.is_valid(expected))
        expected = joined.column("expected")
    correct = pc.sum(pc.equal(joined.column("predicted"), expected)).as_py()
    return LabelCounts(correct=correct or 0, evaluated=joined.num_rows,
                       unlabeled=unlabeled)


def span_agreement(table: pa.Table, alias: str, labels) -> SpanCounts:
    """Count an extract answer table's agreement with its reference spans.

    Args:
        table: The alias's id column and a string `answer` column, null
            where the engine found no answer.
        alias: The id column.
        labels: The `PredicateLabels` of the extraction.

    Raises:
        KeyError: A document the engine answered has no reference row.
    """
    reference = dict(zip(labels.table.column("left_id").to_pylist(),
                         labels.table.column("answers").to_pylist()))
    counts = SpanCounts()
    ids = pc.cast(table.column(alias), pa.string()).to_pylist()
    answers = pc.cast(table.column("answer"), pa.string()).to_pylist()
    missing = sum(row_id not in reference for row_id in ids)
    if missing:
        raise KeyError(
            f"no ground truth for {labels.key} and {missing} answered rows")
    for row_id, answer in zip(ids, answers):
        exact, f1 = span_scores(answer, reference[row_id])
        counts.evaluated += 1
        counts.exact += int(exact)
        counts.f1_total += f1
    return counts


def agreement(table: pa.Table, aliases, labels) -> BinaryCounts:
    """Count an answer table's agreement with the labels, in one join.

    Args:
        table: One id column per alias and a boolean `answer` column.
        aliases: The id columns, in the labels' left then right order.
        labels: The `PredicateLabels` of the predicate.

    Raises:
        KeyError: A row the engine answered has no label.
    """
    aliases = list(aliases)
    reference = labels.table
    reference_keys, keys, count = tuple_keys(
        [reference.column(name)
         for name in ("left_id", "right_id")[:len(aliases)]],
        [table.column(alias) for alias in aliases])
    expected = lookup(
        reference_keys,
        np.asarray(pc.cast(reference.column("answer"), pa.int8())),
        count, keys)
    missing = int(np.count_nonzero(expected < 0))
    if missing:
        raise KeyError(
            f"no ground truth for {labels.key} and {missing} answered rows")
    expected = expected.astype(bool)
    predicted = np.asarray(pc.cast(table.column("answer"), pa.bool_()))

    def number(mask):
        return int(np.count_nonzero(mask))

    counts = BinaryCounts(
        evaluated=len(predicted),
        true_positive=number(predicted & expected),
        true_negative=number(~predicted & ~expected),
        false_positive=number(predicted & ~expected),
        false_negative=number(~predicted & expected),
    )
    counts.correct = counts.true_positive + counts.true_negative
    return counts


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _chain(pairs: list[str], survivors: list[str]) -> str:
    """Return the FROM clause joining pair and survivor tables by alias.

    Every table is keyed by alias columns, so a natural join is the
    join of the query: pairs meet on their shared alias, and a
    survivor table keeps the ids that passed that alias's filters.
    """
    return " NATURAL JOIN ".join(
        [f"(SELECT DISTINCT * FROM {_quoted(name)}) AS {_quoted(name)}"
         for name in pairs]
        + [_quoted(name) for name in survivors])


def _projection(selected, from_clause: str, distinct: bool) -> str:
    columns = ", ".join(_quoted(alias) for alias in selected)
    return (f"SELECT {'DISTINCT ' if distinct else ''}{columns} "
            f"FROM {from_clause}")


def _distinct(table: pa.Table) -> pa.Table:
    """Return the table's distinct rows, every column as a string."""
    table = pa.table({name: pc.cast(table.column(name), pa.string())
                      for name in table.column_names})
    return table.group_by(table.column_names).aggregate([])


def join_size(tables: list[pa.Table]) -> int:
    """Return the row count of the natural join of distinct tables.

    The join is never built. Each table gets a weight of 1 per row, and
    columns are summed out one at a time: the tables that hold the
    column are joined on their shared columns, their weights multiplied,
    and the result grouped by its other columns with the weights
    summed. The column held by the fewest other columns goes first, so
    for a chain of pair tables no intermediate outgrows a pair table.
    """
    factors = [_distinct(table) for table in tables]
    factors = [table.append_column(
        "weight", pa.array([1] * table.num_rows, pa.int64()))
        for table in factors]
    total = 1
    while True:
        columns = {name for table in factors for name in table.column_names
                   if name != "weight"}
        if not columns:
            break

        def neighbors(column):
            return {name for table in factors
                    if column in table.column_names
                    for name in table.column_names} - {column, "weight"}

        column = min(sorted(columns), key=lambda name: len(neighbors(name)))
        holding = [table for table in factors if column in table.column_names]
        factors = [table for table in factors
                   if column not in table.column_names]
        joined = holding[0]
        for table in holding[1:]:
            table = table.rename_columns(
                [name if name != "weight" else "partner_weight"
                 for name in table.column_names])
            shared = sorted((set(joined.column_names) - {"weight"})
                            & set(table.column_names))
            joined = joined.join(table, keys=shared, join_type="inner")
            joined = joined.set_column(
                joined.column_names.index("weight"), "weight",
                pc.multiply(joined.column("weight"),
                            joined.column("partner_weight")))
            joined = joined.drop_columns(["partner_weight"])
        keep = sorted(set(joined.column_names) - {column, "weight"})
        if keep:
            factors.append(joined.group_by(keep).aggregate(
                [("weight", "sum")]).rename_columns(keep + ["weight"]))
        else:
            total *= pc.sum(joined.column("weight")).as_py() or 0
    for table in factors:
        total *= pc.sum(table.column("weight")).as_py() or 0
    return total


def _intersection(left: pa.Table, right: pa.Table) -> pa.Table:
    """Return the distinct rows in both tables, which share their columns."""
    left, right = _distinct(left), _distinct(right)
    return left.join(right, keys=left.column_names, join_type="inner")


def row_counts(spec: QuerySpec, output: RunOutput, ground_truth,
               corpus_rows) -> tuple[int, int, int]:
    """Return (predicted, expected, matched) result rows, without building them.

    A result row is a tuple of one id per selected alias. The expected
    rows are the join of the labels' true pairs, each alias restricted
    to the ids that pass its filters; a traced run's predicted rows are
    the same join over the engine's true pairs and survivors. A row is
    in both exactly when every pair is true for both and every id
    survives both, so the matched rows are the join over the
    intersected pairs and survivors. DuckDB streams these joins and
    counts them, so a result of hundreds of millions of rows is never
    held in memory. When every alias is selected, the joined tuples are
    distinct and the counts are join sizes, computed without walking the
    rows (`join_size`). An untraced run's rows are counted as saved.
    """
    import duckdb

    selected = scored_columns(spec)
    selected_aliases = {name.split(".")[0] for name in spec.info.select}
    aliases = [relation.alias for relation in spec.info.relations]
    # with every alias selected, the joined tuples are already distinct;
    # a label column is a function of its alias's id, or of a pair's ids
    distinct = selected_aliases != set(aliases)
    returned_labels = [operator.output for operator in _selected_labels(spec)]
    with tempfile.TemporaryDirectory(prefix="quail_b_rows_") as spill:
        con = duckdb.connect()
        con.execute(f"SET temp_directory = '{spill}'")
        tables = {}

        def register(name, table):
            con.register(name, table)
            tables[name] = table

        def ids(name, values):
            register(name, pa.table({
                name.split(":", 1)[1]: pa.array(sorted(values), pa.string())}))

        def size(names) -> int:
            return join_size([tables[name] for name in names])

        survivors = expected_survivors(spec, ground_truth, corpus_rows)
        for alias, values in survivors.items():
            ids(f"expected:{alias}", values)
        pairs = _expected_pairs(spec, ground_truth, corpus_rows)
        _check_joined_labels(spec, ground_truth, survivors, pairs)
        for index, table in enumerate(pairs):
            register(f"expected_pairs:{index}", table)
        reference_labels = expected_labels(spec, ground_truth)
        for output_name in returned_labels:
            register(f"expected_labels:{output_name}",
                     reference_labels[output_name])
        expected_pairs = [f"expected_pairs:{i}"
                          for i in range(len(spec.info.joins))] + [
            f"expected_labels:{name}" for name in returned_labels]
        expected_from = _chain(
            expected_pairs, [f"expected:{alias}" for alias in aliases])

        def count(query: str) -> int:
            return con.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]

        expected_names = expected_pairs + [f"expected:{alias}" for alias in aliases]
        expected_count = (count(_projection(selected, expected_from, True))
                          if distinct else size(expected_names))
        answers = scores_from_answers(spec, output, corpus_rows)
        if answers is None:
            # an untraced run: its saved rows, as strings like the ids above
            if output.rows is None:
                raise ValueError("scoring a run without answers needs its rows")
            con.register("rows", output.rows)
            rows = "SELECT DISTINCT " + ", ".join(
                f"CAST({_quoted(alias)} AS VARCHAR) AS {_quoted(alias)}"
                for alias in selected) + " FROM rows"
            return (
                count(rows),
                expected_count,
                count(f"({rows}) INTERSECT "
                      f"({_projection(selected, expected_from, True)})"))
        survivors, relations = answers
        engine_ids = []
        for alias in aliases:
            if survivors[alias] is not None:
                ids(f"engine:{alias}", survivors[alias])
                engine_ids.append(f"engine:{alias}")
            else:
                ids(f"corpus:{alias}", corpus_ids(spec, corpus_rows)[alias]
                    .to_pylist())
                engine_ids.append(f"corpus:{alias}")
        for index, relation in enumerate(relations):
            register(f"engine_pairs:{index}", relation)
        engine_labels = answer_labels(spec, output.classify_answers)
        for output_name in returned_labels:
            register(f"engine_labels:{output_name}",
                     engine_labels[output_name])
        engine_pairs = [f"engine_pairs:{i}" for i in range(len(relations))] + [
            f"engine_labels:{name}" for name in returned_labels]
        engine_from = _chain(engine_pairs, engine_ids)
        if distinct:
            predicted_count = count(_projection(selected, engine_from, True))
            matched_count = count(
                f"({_projection(selected, expected_from, True)}) INTERSECT "
                f"({_projection(selected, engine_from, True)})")
        else:
            predicted_count = size(engine_pairs + engine_ids)
            matched_count = join_size(
                [_intersection(tables[expected_name], tables[engine_name])
                 for expected_name, engine_name in zip(expected_pairs,
                                                       engine_pairs)]
                + [tables[f"expected:{alias}"] for alias in aliases]
                + [tables[name] for name in engine_ids])
        return predicted_count, expected_count, matched_count


def evaluate(spec: QuerySpec, output: RunOutput, ground_truth, corpus_rows) -> dict:
    per_predicate = []
    total = BinaryCounts()
    filters = {
        filter_spec.id: filter_spec for filter_spec in spec.info.filters
    }
    for operator_id, table in (output.filter_answers or {}).items():
        filter_spec = filters[operator_id]
        labels = _labels(ground_truth, filter_spec.prompt)
        item = _PredicateCount(labels.key, "filter", filter_spec.relation)
        item.counts = agreement(table, [filter_spec.relation], labels)
        total.merge(item.counts)
        per_predicate.append(item.as_dict())
    joins = {join.id: join for join in spec.info.joins}
    for operator_id, table in (output.join_answers or {}).items():
        join = joins[operator_id]
        labels = _labels(ground_truth, join.prompt)
        item = _PredicateCount(labels.key, "join")
        item.counts = agreement(table, join.relations, labels)
        total.merge(item.counts)
        per_predicate.append(item.as_dict())
    classifies = {operator.id: operator for operator in spec.info.classifies}
    label_total = LabelCounts()
    for operator_id, table in (output.classify_answers or {}).items():
        operator = classifies[operator_id]
        labels = _classify_labels(ground_truth, operator)
        counts = label_agreement(table, operator.relations, labels)
        label_total.merge(counts)
        per_predicate.append({
            "predicate_key": labels.key, "op": "classify",
            "alias": operator.relation, **counts.as_dict()})
    extracts = {operator.id: operator for operator in spec.info.extracts}
    span_total = SpanCounts()
    for operator_id, table in (output.extract_answers or {}).items():
        operator = extracts[operator_id]
        labels = _extract_labels(ground_truth, operator)
        counts = span_agreement(table, operator.relation, labels)
        span_total.merge(counts)
        per_predicate.append({
            "predicate_key": labels.key, "op": "extract",
            "alias": operator.relation, **counts.as_dict()})

    if spec.info.relational:
        if output.rows is None:
            raise ValueError("a relational query is scored from its rows")
        output_accuracy = relational_accuracy(
            spec, output.rows, ground_truth, corpus_rows)
    else:
        predicted_count, expected_count, matched_count = row_counts(
            spec, output, ground_truth, corpus_rows)
        output_accuracy = _row_metrics(
            predicted_count, expected_count, matched_count)
    input_document_rows = sum(
        len(corpus_rows[relation.table])
        for relation in spec.info.relations)
    unique_documents = {
        (relation.table, str(row_id))
        for relation in spec.info.relations
        for row_id in _ids(corpus_rows[relation.table])
    }
    return {
        "ground_truth_collection_id": ground_truth.collection_id,
        "ground_truth_reference_model": ground_truth.reference_model,
        "answer_accuracy": (total.as_dict() if total.evaluated else None),
        "label_accuracy": (
            label_total.as_dict() if label_total.evaluated else None),
        "span_accuracy": (
            span_total.as_dict() if span_total.evaluated else None),
        "output_accuracy": output_accuracy,
        "per_predicate": per_predicate,
        "input_document_rows": input_document_rows,
        "unique_input_documents": len(unique_documents),
    }
