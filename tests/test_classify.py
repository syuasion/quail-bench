"""Classification queries: plans, prompts, labels, and scoring."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail_b
from quail_b import predicates, prompts, rendering
from quail_b.data import (
    DATA_SEED,
    GROUND_TRUTH_ROOT,
    PUBLISHED_CORPORA,
    SOURCE_REVISIONS,
    corpus_identity,
)
from quail_b.queries import QuerySpec, pending_query_ids, queries
from quail_b.run import _write_json
from quail_b.substrait import AI_CLASSIFY_JOINED_NAME, AI_CLASSIFY_NAME
from tools.make_substrait_plans import (
    Classify,
    InList,
    Join,
    Scan,
    build_plan,
)

CLASSIFY_QUERIES = ("IMDB-11", "IMDB-12", "IMDB-13", "IMDB-14", "IMDB-15",
                    "BIO-5", "BIO-6", "FEV-11", "LEP-6", "AGENT-3", "AGENT-4",
                    "AGENT-5", "REL-AGENT-5")


def test_classification_queries_are_published():
    assert pending_query_ids() == (
        "TA-1", "TA-2", "TA-3", "REL-TA-1")
    assert tuple(query_id for query_id, spec in queries().items()
                 if spec.info.classifies) == CLASSIFY_QUERIES


def test_agent_5_classifies_each_trace_three_ways():
    info = quail_b.get_query("AGENT-5").info
    assert [operator.output for operator in info.classifies] == [
        "progress", "domain", "root_cause"]
    assert {operator.relation for operator in info.classifies} == {"t"}
    assert not info.filters and not info.in_lists and not info.joins
    assert info.select == ("t.id", "t.progress", "t.domain", "t.root_cause")


def test_every_classification_has_one_predicate():
    specs = predicates.CLASSIFY_PREDICATES + tuple(
        spec for spec in predicates.TRIPADVISOR_PREDICATES
        if spec.kind == "classify")
    by_template = {
        spec.template: spec for spec in specs}
    assert len(by_template) == len(specs)
    used = set()
    for query_id in CLASSIFY_QUERIES + pending_query_ids():
        for operator in quail_b.get_query(query_id).info.classifies:
            spec = by_template[operator.prompt]
            assert spec.labels == operator.labels
            assert predicates._descriptions(spec) == operator.descriptions
            used.add(spec.key)
    assert used == {spec.key for spec in specs}


def test_tripadvisor_department_policy_keeps_manual_and_no_action():
    operator = quail_b.get_query("TA-2").info.classifies[0]
    assert operator.labels == prompts.TA_2_LABELS
    assert operator.descriptions == prompts.TA_2_DESCRIPTIONS
    assert operator.labels == (
        "housekeeping", "maintenance", "front_desk", "billing",
        "manual_review", "no_action")
    assert operator.partner is None
    assert "multiple departments" in operator.descriptions[4]
    assert "external conditions" in operator.descriptions[5]


def test_plans_carry_label_columns_and_filters():
    info = quail_b.get_query("IMDB-13").info
    assert [operator.id for operator in info.operators] == [
        "classify-1", "in-list-1", "join-1"]
    assert info.select == ("r.id", "r.sentiment", "a.id")
    assert list(quail_b.get_query("IMDB-13").plan.relations[0].root.names) \
        == ["r", "sentiment", "a"]
    (in_list,) = info.in_lists
    assert in_list.accepted == ("negative", "mixed")
    agent = quail_b.get_query("AGENT-4").info
    assert [operator.output for operator in agent.classifies] == [
        "progress", "test_result"]
    assert agent.in_lists[0].accepted == prompts.AGENT_CHANGED_CODE
    assert agent.classifies[1].descriptions == (
        prompts.AGENT_TEST_RESULT_DESCRIPTIONS)
    assert quail_b.get_query("IMDB-12").info.classifies[0].descriptions == (
        ("",) * len(prompts.IMDB_GENRE_LABELS))


def test_joined_classification_plan():
    plan = quail_b.get_query("IMDB-15").plan
    info = quail_b.get_query("IMDB-15").info
    assert [operator.id for operator in info.operators] == [
        "classify-1", "in-list-1", "join-1", "classify-2"]
    one, pair = info.classifies
    assert (one.relation, one.partner, one.relations) == ("r", None, ("r",))
    assert (pair.relation, pair.partner) == ("r", "a")
    assert pair.relations == ("r", "a")
    assert pair.prompt == prompts.IMDB_ASPECT_SENTIMENT
    assert pair.labels == prompts.IMDB_SENTIMENT_LABELS
    assert pair.output == "aspect_sentiment"
    assert info.pairing_join(pair) is info.joins[0]
    assert info.select == ("r.id", "r.sentiment", "a.id", "r.aspect_sentiment")
    assert list(plan.relations[0].root.names) == [
        "r", "sentiment", "a", "aspect_sentiment"]
    names = {declaration.extension_function.name
             for declaration in plan.extensions}
    assert {AI_CLASSIFY_NAME, AI_CLASSIFY_JOINED_NAME} <= names
    # the ProjectRel sits directly over the JoinRel, anchor field first
    project = plan.relations[0].root.input.project.input.project
    assert project.common.hint.alias == "classify-2"
    assert project.input.join.common.hint.alias == "join-1"
    arguments = project.expressions[0].scalar_function.arguments
    fields = [argument.value.selection.direct_reference.struct_field.field
              for argument in arguments[1:3]]
    assert fields == [1, 4]    # r.body, then a.aspect after r's 3 fields


def _spec(tree, select):
    return QuerySpec.from_plan("TEST", "test", build_plan(tree, select))


def _pair_tree(documents=None):
    reviews = Scan("reviews", "r", "body")
    aspects = Scan("aspects", "a", "aspect")
    return Classify(Join(reviews, aspects, ("r", "a"), "{0} {1}"), "{0} {1}",
                    ("yes", "no"), "x", documents=documents)


def test_joined_classification_rules():
    info = _spec(_pair_tree(), ("r", "a", "r.x")).info
    assert info.classifies[0].relations == ("r", "a")
    reversed_info = _spec(_pair_tree(("a", "r")), ("r", "a", "a.x")).info
    assert reversed_info.classifies[0].relations == ("a", "r")
    with pytest.raises(ValueError, match="one-document classification"):
        _spec(InList(_pair_tree(), "x", ("yes",)), ("r", "a"))
    with pytest.raises(ValueError, match="both relations' ids"):
        _spec(_pair_tree(), ("r", "r.x"))
    # no join pairs exactly the two classified relations
    three = Classify(
        Join(Join(Scan("reviews", "r", "body"), Scan("aspects", "a", "aspect"),
                  ("r", "a"), "{0} {1}"),
             Scan("aspects", "a2", "aspect"), ("r", "a2"), "{0} {1}"),
        "{0} {1}", ("yes", "no"), "x", documents=("a", "a2"))
    with pytest.raises(ValueError, match="must follow the join"):
        _spec(three, ("r", "a", "a2", "a.x"))


@pytest.mark.parametrize(("labels", "message"), [
    (("fixed the bug", "fixed the bug quickly"), "extends label"),
    (("same", "Same "), "distinct"),
    (("only",), "at least two"),
])
def test_invalid_label_lists_are_rejected(labels, message):
    tree = Classify(Scan("reviews", "r", "body"), "{0}", labels, "x")
    with pytest.raises(ValueError, match=message):
        _spec(tree, ("r", "r.x"))


def test_invalid_label_uses_are_rejected():
    scan = Scan("reviews", "r", "body")
    tree = Classify(scan, "{0}", ("yes", "no"), "x")
    with pytest.raises(ValueError, match="not in its call"):
        _spec(InList(tree, "x", ("maybe",)), None)
    with pytest.raises(ValueError, match="never used"):
        _spec(tree, None)
    with pytest.raises(ValueError, match="relation's id"):
        _spec(tree, ("r.x",))


def test_classify_prompt_text():
    text = rendering.render_classify_prompt(
        prompts.FEV_TOPIC, "Paris is in France.", ("geography", "politics"),
        ("places and borders", ""))
    expected_text = (
        "DOCUMENT:\nParis is in France.\n\n"
        "Answer with exactly one of the categories below for the following "
        "question: Judge strictly from the claim above which topic it is "
        "about.\n\n"
        "Categories:\n- geography: places and borders\n- politics\nANSWER:")
    assert text == expected_text
    spec = predicates.PREDICATE_BY_KEY["quailb.fever.claim.topic"]
    assert predicates.render_classify_prompt(spec, "x").endswith(
        "- religion\n- other\nANSWER:")


def test_joined_classify_prompt_text():
    text = rendering.render_classify_prompt(
        prompts.IMDB_ASPECT_SENTIMENT, "The plot dragged on.",
        prompts.IMDB_ASPECT_SENTIMENT_LABELS, partner="pacing")
    expected_text = (
        "DOCUMENT:\nThe plot dragged on.\n\n"
        "(The document above is DOCUMENT {0}.)\n\n"
        "DOCUMENT {1}:\npacing\n\n"
        "Answer with exactly one of the categories below for the following "
        "question: Judge strictly from the review in DOCUMENT {0} what "
        "sentiment it expresses about the movie aspect in DOCUMENT {1}.\n\n"
        "Categories:\n- positive\n- negative\n- neutral\n- mixed\nANSWER:")
    assert text == expected_text
    spec = predicates.PREDICATE_BY_KEY["quailb.imdb.review.aspect_sentiment"]
    assert predicates.render_classify_prompt(
        spec, "The plot dragged on.", "pacing") == text
    with pytest.raises(ValueError, match="joined rows"):
        predicates.render_classify_prompt(spec, "The plot dragged on.")
    with pytest.raises(ValueError, match="one document"):
        predicates.render_classify_prompt(SENTIMENT, "x", "y")
    with pytest.raises(ValueError, match="placeholders"):
        rendering.render_classify_prompt(
            prompts.IMDB_SENTIMENT, "x", ("a", "b"), partner="y")


def test_joined_classify_label_sets_have_their_own_identity():
    spec = predicates.PREDICATE_BY_KEY["quailb.imdb.review.aspect_sentiment"]
    payload = predicates.predicate_payload(spec)
    assert payload["render"] == "classify_pair_anchor_then_partner_v1"
    assert payload["anchor_note"] == rendering.JOIN_ANCHOR_NOTE
    assert payload["partner_label"] == rendering.JOIN_DOC_LABEL
    assert (payload["right_table"], payload["right_column"]) == (
        "aspects", "aspect")
    single = predicates.predicate_payload(SENTIMENT)
    assert "anchor_note" not in single and "partner_label" not in single
    # the published one-document label sets keep their identity
    assert predicates.predicate_version(SENTIMENT)[0] == (
        "pv_5706c27900253228036d76b0fb339409")


def test_classify_label_sets_have_their_own_identity():
    spec = predicates.PREDICATE_BY_KEY["quailb.imdb.review.sentiment"]
    payload = predicates.predicate_payload(spec)
    assert payload["render"] == "classify_document_then_categories_v1"
    assert payload["labels"] == list(prompts.IMDB_SENTIMENT_LABELS)
    assert "task_instruction" not in payload
    identity = predicates.label_set_identity(spec, "c_test", "hash")
    assert identity["sources"][0]["spec"] == predicates.CLASSIFY_JUDGE_SPEC


def _collection(root, tables, label_sets):
    """Write a sf=0.1 corpus and one collection of the given label sets.

    Args:
        root: Local mirror root.
        tables: Table name to rows.
        label_sets: (predicate, rows) pairs. A classify predicate's rows
            are (left_id, label), or (left_id, right_id, label) for a
            classification of joined rows; a filter's (left_id, answer); a join's
            (left_id, right_id, answer).
    """
    corpus_id = PUBLISHED_CORPORA[0.1]
    corpus = root / GROUND_TRUTH_ROOT / "corpora" / corpus_id
    corpus.mkdir(parents=True)
    manifest = corpus_identity(tables, 0.1, DATA_SEED, SOURCE_REVISIONS)
    manifest["corpus_id"] = corpus_id
    _write_json(corpus / "manifest.json", manifest)
    for name, table in tables.items():
        pq.write_table(table, corpus / f"{name}.parquet")
    ids = {}
    for index, (predicate, rows) in enumerate(label_sets):
        key = f"predicate_{index}"
        label_id = f"ls_test_{index}"
        ids[key] = label_id
        directory = root / GROUND_TRUTH_ROOT / "label_sets" / key / label_id
        directory.mkdir(parents=True)
        _write_json(directory / "manifest.json", {
            "status": "complete", "rows": len(rows),
            "source_rows": {"test": len(rows)},
            "predicate": {"key": key, **predicate},
        })
        columns = {
            "predicate_key": [key] * len(rows),
            "label_set_id": [label_id] * len(rows),
            "left_id": [row[0] for row in rows],
        }
        if predicate["kind"] == "classify":
            columns["right_id"] = pa.array(
                [row[1] if len(row) == 3 else None for row in rows],
                pa.string())
            columns["label"] = [row[-1] for row in rows]
        elif predicate["kind"] == "join":
            columns["right_id"] = [row[1] for row in rows]
            columns["answer"] = [row[2] for row in rows]
        else:
            columns["right_id"] = pa.array([None] * len(rows), pa.string())
            columns["answer"] = [row[1] for row in rows]
        pq.write_table(pa.table(columns), directory / "labels.parquet")
    collection_id = "gt_test"
    collection = root / GROUND_TRUTH_ROOT / "collections" / collection_id
    collection.mkdir(parents=True)
    _write_json(collection / "manifest.json", {
        "status": "complete", "collection_id": collection_id,
        "corpus_id": corpus_id, "scale_factor": 0.1, "label_sets": ids,
        "summary": {"model": "test"},
    })
    _write_json(corpus / "active_collection.json",
                {"collection_id": collection_id})


def _classify_predicate(spec):
    predicate = {"kind": "classify", "template": spec.template,
                 "left_table": spec.left_table, "left_column": spec.left_column,
                 "labels": list(spec.labels)}
    if spec.right_table:
        predicate.update(right_table=spec.right_table,
                         right_column=spec.right_column)
    return predicate


SENTIMENT = predicates.PREDICATE_BY_KEY["quailb.imdb.review.sentiment"]
ASPECT_SENTIMENT = predicates.PREDICATE_BY_KEY[
    "quailb.imdb.review.aspect_sentiment"]
IMDB_TABLES = {
    "reviews": pa.table({"id": ["r0", "r1", "r2"], "body": ["a", "b", "c"]}),
    "aspects": pa.table({"id": ["a0", "a1"], "aspect": ["x", "y"]}),
}
IMDB_LABEL_SETS = [
    (_classify_predicate(SENTIMENT), [
        ("r0", "negative"), ("r1", "positive"), ("r2", "mixed")]),
    ({"kind": "join", "template": prompts.DISCUSS_ASPECT,
      "left_table": "reviews", "right_table": "aspects"}, [
        ("r0", "a0", True), ("r0", "a1", True), ("r1", "a0", True),
        ("r1", "a1", False), ("r2", "a0", False), ("r2", "a1", True)]),
]
JOINED_LABELS = [("r0", "a0", "negative"), ("r0", "a1", "mixed"),
               ("r1", "a0", "positive"), ("r2", "a1", "negative")]


def _imdb_13(root):
    _collection(root, IMDB_TABLES, IMDB_LABEL_SETS)


def _imdb_15(root, joined_labels=JOINED_LABELS):
    """A collection with a label set for the join's rows."""
    _collection(root, IMDB_TABLES, [
        *IMDB_LABEL_SETS, (_classify_predicate(ASPECT_SENTIMENT), joined_labels)])


def _imdb_13_output(rows=True, wrong_row=False):
    engine_rows = pa.table({
        "r": ["r0", "r0", "r1"],
        "sentiment": ["negative"] * 2
        + ["mixed" if wrong_row else "negative"],
        "a": ["a0", "a1", "a0"],
    })
    return quail_b.RunOutput(
        {},
        {"join-1": pa.table({
            "r": ["r0", "r0", "r1", "r1"], "a": ["a0", "a1", "a0", "a1"],
            "answer": [True, True, True, False]})},
        engine_rows if rows else None,
        runtime_s=1.0,
        classify_answers={"classify-1": pa.table({
            "r": ["r0", "r1", "r2"],
            "label": ["negative", "negative", "positive"]})})


def test_classification_run_scores_labels_and_rows(tmp_path):
    _imdb_13(tmp_path)
    benchmark = quail_b.load_benchmark(["IMDB-13"], root=tmp_path)
    expected = quail_b.scoring.expected_rows(
        benchmark.queries[0], benchmark.ground_truth, benchmark.tables)
    assert sorted(expected.to_pylist(), key=str) == sorted([
        {"a": "a0", "r": "r0", "sentiment": "negative"},
        {"a": "a1", "r": "r0", "sentiment": "negative"},
        {"a": "a1", "r": "r2", "sentiment": "mixed"},
    ], key=str)

    record = quail_b.run(
        lambda spec, tables: _imdb_13_output(), queries=["IMDB-13"],
        output_dir=tmp_path / "run", root=tmp_path)
    accuracy = record["queries"][0]["metrics"]["accuracy"]
    assert accuracy["label_accuracy"] == {
        "correct": 1, "evaluated": 3, "unlabeled": 0, "accuracy": 0.333333}
    assert accuracy["answer_accuracy"]["correct"] == 4
    assert accuracy["output_accuracy"]["predicted_rows"] == 3
    assert accuracy["output_accuracy"]["expected_rows"] == 3
    assert accuracy["output_accuracy"]["matching_rows"] == 2
    classify_item = accuracy["per_predicate"][-1]
    assert classify_item["op"] == "classify"
    assert classify_item["alias"] == "r"
    report = (tmp_path / "run" / "report.md").read_text()
    assert "| Label accuracy | Evaluated labels |" in report

    saved = json.loads((tmp_path / "run" / "run.json").read_text())
    quail_b.report(tmp_path / "run", root=tmp_path)
    rescored = json.loads((tmp_path / "run" / "run.json").read_text())
    assert rescored["queries"][0]["metrics"] == saved["queries"][0]["metrics"]

    untraced = quail_b.RunOutput(
        None, None, _imdb_13_output().rows, runtime_s=1.0)
    record = quail_b.run(
        lambda spec, tables: untraced, queries=["IMDB-13"],
        output_dir=tmp_path / "untraced", root=tmp_path)
    output = record["queries"][0]["metrics"]["accuracy"]["output_accuracy"]
    assert (output["predicted_rows"], output["matching_rows"]) == (3, 2)
    assert record["queries"][0]["metrics"]["accuracy"]["label_accuracy"] is None


def test_rows_must_follow_the_classification_answers(tmp_path):
    _imdb_13(tmp_path)
    with pytest.raises(ValueError, match="not implied"):
        quail_b.run(
            lambda spec, tables: _imdb_13_output(wrong_row=True),
            queries=["IMDB-13"], output_dir=tmp_path / "run", root=tmp_path)

    def unknown_label(spec, tables):
        output = _imdb_13_output()
        output.classify_answers["classify-1"] = pa.table({
            "r": ["r0", "r1", "r2"],
            "label": ["negative", "negative", "loves it"]})
        return output

    with pytest.raises(ValueError, match="not one of the query's labels"):
        quail_b.run(unknown_label, queries=["IMDB-13"],
                    output_dir=tmp_path / "unknown", root=tmp_path)


def test_two_label_columns_and_chained_calls(tmp_path):
    complaint = predicates.PREDICATE_BY_KEY["quailb.imdb.review.main_complaint"]
    progress = predicates.PREDICATE_BY_KEY["quailb.agent.trace.progress"]
    result = predicates.PREDICATE_BY_KEY["quailb.agent.trace.test_result"]
    topic = predicates.PREDICATE_BY_KEY["quailb.fever.claim.topic"]
    _collection(tmp_path, {
        "reviews": pa.table({"id": ["r0", "r1", "r2"],
                             "body": ["a", "b", "c"]}),
        "agent_traces": pa.table({
            "id": ["t0", "t1"], "trace": ["p", "q"],
            "trajectory_id": ["j0", "j1"], "turn_index": [3, 5],
            "token_count": [10, 20]}),
        "claims": pa.table({
            "id": ["c0", "c1", "c2"], "claim": ["u", "v", "w"],
            "label": ["SUPPORTS"] * 3, "evidence_wiki_url": ["p0", "p1", "p2"]}),
    }, [
        (_classify_predicate(SENTIMENT), [
            ("r0", "negative"), ("r1", "positive"), ("r2", "mixed")]),
        (_classify_predicate(complaint), [
            ("r0", "slow pacing"), ("r1", "no specific complaint"),
            ("r2", "poor acting")]),
        (_classify_predicate(progress), [
            ("t0", "located the relevant code"),
            ("t1", "changed the code, check fails")]),
        (_classify_predicate(result), [("t0", "bug still occurs"),
                                       ("t1", "bug still occurs")]),
        (_classify_predicate(topic), [("c0", "politics"), ("c1", "sports"),
                                      ("c2", "history")]),
    ])

    def labels(alias, ids, values):
        return pa.table({alias: ids, "label": values})

    def execute(spec, tables):
        if spec.id == "IMDB-14":
            return quail_b.RunOutput(
                {}, {}, pa.table({
                    "r": ["r0", "r2"], "sentiment": ["negative", "mixed"],
                    "complaint": ["slow pacing", "poor writing"]}),
                runtime_s=1.0, classify_answers={
                    "classify-1": labels("r", ["r0", "r1", "r2"],
                                         ["negative", "positive", "mixed"]),
                    "classify-2": labels("r", ["r0", "r2"],
                                         ["slow pacing", "poor writing"])})
        if spec.id == "AGENT-4":
            return quail_b.RunOutput(
                {}, {}, pa.table({
                    "t": ["t1"], "progress": ["changed the code, check fails"],
                    "test_result": ["bug still occurs"]}),
                runtime_s=1.0, classify_answers={
                    "classify-1": labels(
                        "t", ["t0", "t1"], ["located the relevant code",
                                            "changed the code, check fails"]),
                    "classify-2": labels("t", ["t1"], ["bug still occurs"])})
        return quail_b.RunOutput(
            {}, {}, pa.table({"c": ["c0", "c2"],
                              "topic": ["politics", "history"]}),
            runtime_s=1.0, classify_answers={"classify-1": labels(
                "c", ["c0", "c1", "c2"], ["politics", "science", "history"])})

    record = quail_b.run(execute, queries=["IMDB-14", "AGENT-4", "FEV-11"],
                         output_dir=tmp_path / "run", root=tmp_path)
    imdb, agent, fever = (
        item["metrics"]["accuracy"] for item in record["queries"])
    assert imdb["label_accuracy"] == {
        "correct": 4, "evaluated": 5, "unlabeled": 0, "accuracy": 0.8}
    assert imdb["output_accuracy"]["matching_rows"] == 1
    assert agent["label_accuracy"]["correct"] == 3
    assert agent["output_accuracy"]["exact_match"]
    assert fever["label_accuracy"]["correct"] == 2
    assert fever["output_accuracy"]["exact_match"]
    for item in record["queries"]:
        assert item["metrics"]["minimum_tokens"] is None


def _imdb_15_output(pair_answers=None, extra_row=False):
    """An IMDB-15 run: r0 and r1 are critical, and two pairs get a label."""
    rows = pa.table({
        "r": ["r0", "r1"] + (["r0"] if extra_row else []),
        "sentiment": ["negative", "negative"] + (["negative"] if extra_row
                                                 else []),
        "a": ["a0", "a0"] + (["a1"] if extra_row else []),
        "aspect_sentiment": ["negative", "positive"] + (["mixed"] if extra_row
                                                        else []),
    })
    return quail_b.RunOutput(
        {},
        {"join-1": pa.table({
            "r": ["r0", "r0", "r1", "r1"], "a": ["a0", "a1", "a0", "a1"],
            "answer": [True, True, True, False]})},
        rows, runtime_s=1.0,
        classify_answers={
            "classify-1": pa.table({
                "r": ["r0", "r1", "r2"],
                "label": ["negative", "negative", "positive"]}),
            "classify-2": pair_answers or pa.table({
                "r": ["r0", "r1"], "a": ["a0", "a0"],
                "label": ["negative", "positive"]})})


def test_joined_classification_run_scores_labels_and_rows(tmp_path):
    _imdb_15(tmp_path)
    benchmark = quail_b.load_benchmark(["IMDB-15"], root=tmp_path)
    expected = quail_b.scoring.expected_rows(
        benchmark.queries[0], benchmark.ground_truth, benchmark.tables)
    assert sorted(expected.to_pylist(), key=str) == sorted([
        {"a": "a0", "r": "r0", "sentiment": "negative",
         "aspect_sentiment": "negative"},
        {"a": "a1", "r": "r0", "sentiment": "negative",
         "aspect_sentiment": "mixed"},
        {"a": "a1", "r": "r2", "sentiment": "mixed",
         "aspect_sentiment": "negative"},
    ], key=str)

    record = quail_b.run(
        lambda spec, tables: _imdb_15_output(), queries=["IMDB-15"],
        output_dir=tmp_path / "run", root=tmp_path)
    item = record["queries"][0]
    accuracy = item["metrics"]["accuracy"]
    # r0 is the only correct sentiment; both joined-row labels agree
    assert accuracy["label_accuracy"] == {
        "correct": 3, "evaluated": 5, "unlabeled": 0, "accuracy": 0.6}
    assert accuracy["answer_accuracy"]["correct"] == 4
    # the pair (r0, a1) answered TRUE but got no label, so it is not a row
    assert accuracy["output_accuracy"]["predicted_rows"] == 2
    assert accuracy["output_accuracy"]["expected_rows"] == 3
    assert accuracy["output_accuracy"]["matching_rows"] == 1
    pair_item = accuracy["per_predicate"][-1]
    assert (pair_item["predicate_key"], pair_item["op"], pair_item["alias"]) \
        == ("predicate_2", "classify", "r")
    assert item["metrics"]["minimum_tokens"] is None
    assert item["metrics"]["evaluated_document_pairs"] == 4

    saved = json.loads((tmp_path / "run" / "run.json").read_text())
    quail_b.report(tmp_path / "run", root=tmp_path)
    rescored = json.loads((tmp_path / "run" / "run.json").read_text())
    assert rescored["queries"][0]["metrics"] == saved["queries"][0]["metrics"]

    untraced = quail_b.RunOutput(
        None, None, _imdb_15_output().rows, runtime_s=1.0)
    record = quail_b.run(
        lambda spec, tables: untraced, queries=["IMDB-15"],
        output_dir=tmp_path / "untraced", root=tmp_path)
    output = record["queries"][0]["metrics"]["accuracy"]["output_accuracy"]
    assert (output["predicted_rows"], output["matching_rows"]) == (2, 1)
    assert record["queries"][0]["metrics"]["accuracy"]["label_accuracy"] is None


def test_joined_rows_must_follow_their_labels(tmp_path):
    _imdb_15(tmp_path)
    with pytest.raises(ValueError, match="answers imply 2"):
        quail_b.run(
            lambda spec, tables: _imdb_15_output(extra_row=True),
            queries=["IMDB-15"], output_dir=tmp_path / "run", root=tmp_path)
    # the engine also classified (r1, a1), a pair the reference join does
    # not keep, so that answer has no reference label and is left out
    unlabeled = pa.table({
        "r": ["r0", "r1", "r0", "r1"], "a": ["a0", "a0", "a1", "a1"],
        "label": ["negative", "positive", "mixed", "positive"]})
    record = quail_b.run(
        lambda spec, tables: _imdb_15_output(unlabeled, extra_row=True),
        queries=["IMDB-15"], output_dir=tmp_path / "unlabeled",
        root=tmp_path)
    labels = record["queries"][0]["metrics"]["accuracy"]["label_accuracy"]
    # three sentiment answers and the three labeled pairs
    assert (labels["evaluated"], labels["unlabeled"]) == (6, 1)


def test_reference_joined_rows_need_a_label(tmp_path):
    # (r2, a1) is a reference join pair of a critical review without a label
    _imdb_15(tmp_path, JOINED_LABELS[:3])
    benchmark = quail_b.load_benchmark(["IMDB-15"], root=tmp_path)
    with pytest.raises(KeyError, match="predicate_2 and 1 pairs of join-1"):
        quail_b.scoring.expected_rows(
            benchmark.queries[0], benchmark.ground_truth, benchmark.tables)
    with pytest.raises(KeyError, match="1 pairs of join-1"):
        quail_b.run(
            lambda spec, tables: _imdb_15_output(), queries=["IMDB-15"],
            output_dir=tmp_path / "run", root=tmp_path)
