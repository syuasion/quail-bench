"""Run and rescore published-format inputs without an inference engine."""

import json
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail_b
from quail_b.data import (
    DATA_SEED,
    GROUND_TRUTH_ROOT,
    PUBLISHED_CORPORA,
    SOURCE_REVISIONS,
    corpus_identity,
)
from quail_b.run import _write_json


def _tripadvisor_output(spec, tables):
    """Fabricate consistent answers to exercise the harness, not a model."""
    from quail_b.scoring import rows_from_answers

    relations = {item.alias: tables[item.table] for item in spec.info.relations}
    filters = {
        item.id: pa.table({item.relation: relations[item.relation]["id"],
                           "answer": [True] * len(relations[item.relation])})
        for item in spec.info.filters
    }
    joins = {}
    matched = []
    for item in spec.info.joins:
        left, right = item.relations
        matched = [{left: first["id"], right: second["id"]}
                   for first in relations[left].to_pylist()
                   for second in relations[right].to_pylist()
                   if all(first[lcol] == second[rcol] for lcol, rcol in item.on)]
        joins[item.id] = pa.table({
            left: [row[left] for row in matched],
            right: [row[right] for row in matched],
            "answer": [True] * len(matched),
        })
    classifications = {}
    for item in spec.info.classifies:
        documents = (matched if item.partner is not None else
                     [{item.relation: value}
                      for value in relations[item.relation]["id"].to_pylist()])
        classifications[item.id] = pa.table({
            **{alias: [row[alias] for row in documents] for alias in item.relations},
            "label": [item.labels[0]] * len(documents),
        })
    scores = {}
    if spec.info.scores:
        item = spec.info.scores[0]
        ids = sorted(relations[item.relation]["id"].to_pylist())
        values = [1 - index / len(ids) for index in range(len(ids))]
        scores[item.id] = pa.table({item.relation: ids, "score": values})
        rows = pa.table({item.relation: ids[:spec.info.fetch.count],
                         item.output: values[:spec.info.fetch.count]})
    else:
        rows = rows_from_answers(spec, filters, joins, corpus_rows=tables,
                                 classify_answers=classifications)
    return quail_b.RunOutput(
        filters, joins, rows, runtime_s=1.0,
        measurements={"fresh_tokens": 100, "input_tokens": 100},
        classify_answers=classifications, score_answers=scores)


@pytest.mark.parametrize("as_directory", [False, True])
def test_tripadvisor_pending_run_and_portable_report(
        tmp_path, tripadvisor_corpus, as_directory):
    import shutil

    from quail_b.queries import pending_query_ids

    destination = tmp_path / "pending-run"
    ids = pending_query_ids()
    assert len(ids) == 4
    record = quail_b.run(
        _tripadvisor_output, queries=ids,
        corpus_manifest=(tripadvisor_corpus.parent if as_directory
                         else tripadvisor_corpus),
        accuracy=False, output_dir=destination, gpu_hourly_rate_usd=3.0,
        metadata={"engine": "fabricated CPU fixture"})
    assert record["status"] == "complete"
    assert record["scale_factor"] is None
    assert record["accuracy_enabled"] is False
    assert record["collection_id"] is None
    assert record["reference_model"] is None
    for item in record["queries"]:
        metrics = item["metrics"]
        assert metrics["accuracy"] is None
        assert metrics["fresh_tokens"] == 100
        assert metrics["returned_rows"] == (12 if item["id"] == "TA-3" else 8)
        assert metrics["cost_usd"] == pytest.approx(3 / 3600)
        assert (destination / item["id"] / "plan.substrait").exists()
    measurements = pq.read_table(destination / "measurements.parquet").to_pylist()
    assert len(measurements) == 4
    assert all(row["answers_correct"] is None and row["expected_rows"] is None
               and row["predicted_rows"] == (12 if row["query"] == "TA-3" else 8)
               for row in measurements)
    assert "Accuracy: unavailable" in (destination / "report.md").read_text()
    shutil.rmtree(tripadvisor_corpus.parent)
    moved = tmp_path / "moved-run"
    destination.rename(moved)
    quail_b.report(moved)
    regenerated = json.loads((moved / "run.json").read_text())
    assert regenerated["queries"] == record["queries"]
    manifest = moved / "inputs" / "manifest.json"
    changed = json.loads(manifest.read_text())
    changed["settings"]["seed"] = 999
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        quail_b.report(moved)


def test_tripadvisor_unlabelled_run_still_validates_outputs(
        tmp_path, tripadvisor_corpus):
    def invalid(spec, tables):
        output = _tripadvisor_output(spec, tables)
        output.rows = pa.table({"r": ["unknown-review"]})
        return output

    destination = tmp_path / "invalid-run"
    with pytest.raises(ValueError):
        quail_b.run(invalid, queries=["TA-1"], accuracy=False,
                    corpus_manifest=tripadvisor_corpus, output_dir=destination)
    record = json.loads((destination / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["queries"][0]["status"] == "scoring_failed"
    assert (destination / "TA-1" / "rows.parquet").exists()


def test_tripadvisor_rows_only_outputs_obey_matching_keys(
        tmp_path, tripadvisor_corpus):
    def mismatched(spec, tables):
        left = tables["tripadvisor_pair_left"].to_pylist()[0]
        right = next(row for row in tables["tripadvisor_pair_right"].to_pylist()
                     if row["pair_id"] != left["pair_id"])
        return quail_b.RunOutput(
            None, None, pa.table({"a": [left["id"]], "b": [right["id"]]}),
            runtime_s=1.0)

    with pytest.raises(ValueError, match="join equality"):
        quail_b.run(mismatched, queries=["TA-3"], accuracy=False,
                    corpus_manifest=tripadvisor_corpus,
                    output_dir=tmp_path / "mismatched-run")


@pytest.mark.parametrize("complaints", [(1, 3, 4), (1,), ()])
def test_tripadvisor_gates_limit_join_answers_to_surviving_endpoints(
        tripadvisor_corpus, complaints):
    from quail_b.data import read_tripadvisor_manifest
    from quail_b.run import _validate_output
    from quail_b.scoring import rows_from_answers

    spec = quail_b.get_query("TA-3")
    _, tables = read_tripadvisor_manifest(tripadvisor_corpus)
    endpoints = {relation.alias: tables[relation.table].to_pylist()
                 for relation in spec.info.relations}
    kept = {
        alias: {row["id"] for row in rows
                if int(row["review_id"]) % 100 in complaints}
        for alias, rows in endpoints.items()
    }
    filters = {
        operator.id: pa.table({
            operator.relation: [row["id"]
                                for row in endpoints[operator.relation]],
            "answer": [row["id"] in kept[operator.relation]
                       for row in endpoints[operator.relation]],
        }) for operator in spec.info.filters
    }
    eligible = [{"a": left["id"], "b": right["id"], "answer": True}
                for left in endpoints["a"] for right in endpoints["b"]
                if left["pair_id"] == right["pair_id"]
                and left["id"] in kept["a"] and right["id"] in kept["b"]]
    schema = pa.schema([("a", pa.string()), ("b", pa.string()),
                        ("answer", pa.bool_())])
    joins = {"join-1": pa.Table.from_pylist(eligible, schema=schema)}
    rows = rows_from_answers(spec, filters, joins, corpus_rows=tables)
    output = quail_b.RunOutput(filters, joins, rows, runtime_s=1.0)
    _validate_output(spec, output, tables)
    assert all(len(answers) == 12 for answers in filters.values())
    assert len(eligible) == len(rows) == len(complaints) * (len(complaints) - 1)
    if not eligible:
        return
    left, right = next((left, right) for left in endpoints["a"]
                       for right in endpoints["b"]
                       if left["pair_id"] == right["pair_id"]
                       and (left["id"] not in kept["a"]
                            or right["id"] not in kept["b"]))
    wrong = pa.table({"a": [left["id"]], "b": [right["id"]]})
    output.rows = pa.concat_tables([wrong, rows.slice(1)])
    with pytest.raises(ValueError, match="not implied"):
        _validate_output(spec, output, tables)


@pytest.mark.parametrize("fault", ["nan", "outside", "string", "order", "missing"])
def test_tripadvisor_score_outputs_require_valid_ranking(
        tmp_path, tripadvisor_corpus, fault):
    def invalid(spec, tables):
        output = _tripadvisor_output(spec, tables)
        if fault in ("nan", "outside", "string"):
            values = {"nan": [float("nan")] * len(output.rows),
                      "outside": [1.1] * len(output.rows),
                      "string": ["high"] * len(output.rows)}[fault]
            output.rows = output.rows.set_column(
                1, "forced_move_score", pa.array(values))
        elif fault == "order":
            output.rows = output.rows.take(pa.array([2, 1, 0]))
        else:
            output.rows = output.rows.slice(0, 2)
        return output

    with pytest.raises(ValueError):
        quail_b.run(invalid, queries=["REL-TA-1"], accuracy=False,
                    corpus_manifest=tripadvisor_corpus,
                    output_dir=tmp_path / f"bad-score-{fault}")


def test_tripadvisor_accepts_float32_returned_scores(tmp_path, tripadvisor_corpus):
    def execute(spec, tables):
        output = _tripadvisor_output(spec, tables)
        output.rows = output.rows.set_column(
            1, "forced_move_score", output.rows["forced_move_score"].cast(pa.float32()))
        return output

    record = quail_b.run(
        execute, queries=["REL-TA-1"], accuracy=False,
        corpus_manifest=tripadvisor_corpus, output_dir=tmp_path / "float32-run")
    assert record["status"] == "complete"
    quail_b.report(tmp_path / "float32-run")


@pytest.mark.parametrize("query", ["TA-2", "REL-TA-1"])
@pytest.mark.parametrize("traces,count", [("absent", 0), ("absent", 3), ("partial", 3)])
def test_tripadvisor_unary_outputs_require_all_selected_rows(
        tmp_path, tripadvisor_corpus, query, traces, count):
    def incomplete(spec, tables):
        output = _tripadvisor_output(spec, tables)
        output.rows = output.rows.slice(0, count)
        name = "classify_answers" if spec.info.classifies else "score_answers"
        answers = getattr(output, name)
        alias = spec.info.relations[0].alias
        retained = set(output.rows[alias].to_pylist())
        setattr(output, name, None if traces == "absent" else {
            key: table.filter(pa.array([value in retained
                                       for value in table[alias].to_pylist()]))
            for key, table in answers.items()})
        return output

    with pytest.raises(ValueError, match="query requires 8"):
        quail_b.run(incomplete, queries=[query], accuracy=False,
                    corpus_manifest=tripadvisor_corpus,
                    output_dir=tmp_path / "incomplete-run")


@pytest.mark.parametrize("query", ["TA-2", "REL-TA-1"])
def test_tripadvisor_complete_rows_do_not_require_answer_traces(
        tmp_path, tripadvisor_corpus, query):
    def execute(spec, tables):
        output = _tripadvisor_output(spec, tables)
        output.classify_answers = output.score_answers = None
        return output

    record = quail_b.run(
        execute, queries=[query], accuracy=False,
        corpus_manifest=tripadvisor_corpus, output_dir=tmp_path / "rows-only-run")
    assert record["status"] == "complete"
    assert record["queries"][0]["metrics"]["returned_rows"] == 8


def _inputs(root, sf):
    spec = quail_b.get_query("IMDB-4")
    tables = {
        "reviews": pa.table({"id": ["r0", "r1"], "body": ["good", "bad"]}),
        "aspects": pa.table({"id": ["a0"], "aspect": ["acting"]}),
    }
    corpus_id = PUBLISHED_CORPORA[sf]
    corpus = root / GROUND_TRUTH_ROOT / "corpora" / corpus_id
    corpus.mkdir(parents=True)
    manifest = corpus_identity(tables, sf, DATA_SEED, SOURCE_REVISIONS)
    manifest["corpus_id"] = corpus_id
    _write_json(corpus / "manifest.json", manifest)
    for name, table in tables.items():
        pq.write_table(table, corpus / f"{name}.parquet")
    collection_id = f"gt_{sf}"
    collection = root / GROUND_TRUTH_ROOT / "collections" / collection_id
    collection.mkdir(parents=True)
    label_sets = {}
    predicates = [
        ("filter", filter_spec.prompt, None)
        for filter_spec in spec.info.filters
    ] + [("join", spec.info.joins[0].prompt, "aspects")]
    for index, (kind, template, right) in enumerate(predicates):
        key = f"predicate_{index}"
        label_id = f"ls_{sf}_{index}"
        label_sets[key] = label_id
        directory = root / GROUND_TRUTH_ROOT / "label_sets" / key / label_id
        directory.mkdir(parents=True)
        _write_json(directory / "manifest.json", {
            "status": "complete", "rows": 2, "source_rows": {"test": 2},
            "predicate": {
                "key": key, "template": template, "kind": kind,
                "left_table": "reviews", "left_column": "body",
                "right_table": right, "right_column": "aspect" if right else None,
            },
        })
        pq.write_table(pa.table({
            "predicate_key": [key, key], "label_set_id": [label_id, label_id],
            "left_id": ["r0", "r1"], "right_id": ["a0" if right else None] * 2,
            "answer": [True, True],
        }), directory / "labels.parquet")
    _write_json(collection / "manifest.json", {
        "status": "complete", "collection_id": collection_id,
        "corpus_id": corpus_id, "scale_factor": sf,
        "label_sets": label_sets, "summary": {"model": "test"},
    })
    _write_json(corpus / "active_collection.json", {"collection_id": collection_id})
    return collection_id


def test_join_runs_at_all_scales_and_report_cli(tmp_path):
    calls = []

    def execute(spec, tables):
        calls.append(spec.id)
        assert set(tables) == {"reviews", "aspects"}
        return quail_b.RunOutput(
            {filter_spec.id: pa.table({
                "r": ["r0", "r1"], "answer": [True, True]})
             for filter_spec in spec.info.filters},
            {"join-1": pa.table({
                "r": ["r0", "r1"], "a": ["a0", "a0"], "answer": [True, True]})},
            pa.table({"r": ["r0", "r1"], "a": ["a0", "a0"]}),
            runtime_s=2.0)

    for sf in PUBLISHED_CORPORA:
        _inputs(tmp_path, sf)
        destination = tmp_path / f"run-{sf}"
        record = quail_b.run(
            execute, queries=["IMDB-4"], scale_factor=sf,
            output_dir=destination, root=tmp_path, gpu_count=2,
            gpu_hourly_rate_usd=3.6)
        assert record["schema_version"] == 2
        metrics = record["queries"][0]["metrics"]
        assert metrics["input_rows"] == {"r": 2, "a": 1}
        assert metrics["input_tokens"] is None
        assert metrics["input_tokens_per_second"] is None
        assert metrics["fresh_tokens"] is None
        assert metrics["minimum_tokens"] is None
        assert metrics["regret_tokens"] is None
        assert metrics["evaluated_document_pairs"] == 2
        assert metrics["document_pairs_per_second"] == 1.0
        assert metrics["cost_usd"] == 0.004
        assert metrics["cost_usd_per_million_input_tokens"] is None
        assert metrics["kv_regret_percent"] is None
        assert metrics["accuracy"]["output_accuracy"]["exact_match"]
        plan_path = destination / "IMDB-4/plan.substrait"
        assert plan_path.read_bytes() == quail_b.get_query("IMDB-4").plan_bytes
        assert record["queries"][0]["files"]["plan"] == "plan.substrait"
        before = (destination / "report.md").read_text()
        quail_b.report(destination, rescore=False)
        assert (destination / "report.md").read_text() == before
        completed = subprocess.run(
            [sys.executable, "-m", "quail_b", "report", str(destination),
             "--root", str(tmp_path)], capture_output=True, text=True, timeout=30)
        assert completed.returncode == 0, completed.stderr
        assert str(destination / "report.md") in completed.stdout
    assert calls == ["IMDB-4"] * 3


@pytest.mark.parametrize("hourly_rate", [None, 0.0, 3.6])
def test_prompt_pieces_give_the_minimum_and_the_regret(
        tmp_path, monkeypatch, hourly_rate):
    import quail_b.minimum

    def encode(texts):
        return [[byte + 1 for byte in text.encode("utf-8")] for text in texts]

    loaded = []
    monkeypatch.setattr(quail_b.minimum, "load_tokenizer",
                        lambda name: loaded.append(name) or encode)
    _inputs(tmp_path, 0.1)
    spec = quail_b.get_query("IMDB-4")
    pieces = {
        "tokenizer": "test-tokenizer", "preamble": [1, 2],
        "filters": [
            {"id": filter_spec.id, "tail": [10 + index, 20]}
            for index, filter_spec in enumerate(spec.info.filters)
        ],
        "joins": [{"id": "join-1", "anchor": "r", "frame": [30, 31],
                   "label": [40], "tail": [50, 51, 52]}],
    }

    def execute(spec, tables):
        return quail_b.RunOutput(
            {filter_spec.id: pa.table({
                "r": ["r0", "r1"], "answer": [True, True]})
             for filter_spec in spec.info.filters},
            {"join-1": pa.table({
                "r": ["r0", "r1"], "a": ["a0", "a0"], "answer": [True, True]})},
            pa.table({"r": ["r0", "r1"], "a": ["a0", "a0"]}),
            runtime_s=2.0, measurements={"fresh_tokens": 1000},
            prompt_pieces=pieces)

    destination = tmp_path / "run"
    record = quail_b.run(execute, queries=["IMDB-4"], output_dir=destination,
                         root=tmp_path, gpu_count=2,
                         gpu_hourly_rate_usd=hourly_rate)
    metrics = record["queries"][0]["metrics"]
    # the preamble once, then "good" and "bad" (4 and 3 tokens, sharing
    # nothing); per review the two filter tails and the frame, which
    # share nothing; per pair the label, "acting", and the tail
    stages = len(spec.info.filters)
    assert stages == 2
    minimum = (2 + 4 + 3) + 2 * (2 * 2 + 2) + 2 * (1 + 6 + 3)
    assert metrics["minimum_tokens"] == minimum
    assert metrics["input_tokens"] == 65
    assert metrics["input_tokens_per_second"] == 32.5
    assert metrics["regret_tokens"] == 1000 - minimum
    cost = None if hourly_rate is None else 2 / 3600 * 2 * hourly_rate
    cost_per_million = None if cost is None else cost / 65 * 1e6
    regret_percent = 100 * (1000 - minimum) / 1000
    assert metrics["cost_usd_per_million_input_tokens"] == cost_per_million
    assert metrics["kv_regret_percent"] == regret_percent
    assert loaded == ["test-tokenizer"]
    saved = json.loads((destination / "IMDB-4/prompt_pieces.json").read_text())
    assert saved == pieces

    quail_b.report(destination, root=tmp_path)
    rescored = json.loads((destination / "run.json").read_text())
    assert rescored["queries"][0]["metrics"] == metrics
    table = pq.read_table(destination / "measurements.parquet")
    assert table.to_pylist()[0] == {
        "query": "IMDB-4", "runtime_s": 2.0, "fresh_tokens": 1000,
        "input_tokens": 65, "input_tokens_per_second": 32.5,
        "minimum_tokens": minimum, "regret_tokens": 1000 - minimum,
        "regret_approximate": False,
        "evaluated_document_pairs": 2, "input_rows": 3,
        "answers_evaluated": 2 * stages + 2, "answers_correct": 2 * stages + 2,
        "labels_evaluated": None, "labels_correct": None,
        "predicted_rows": 2, "expected_rows": 2, "matching_rows": 2,
        "cost_usd": cost,
        "cost_usd_per_million_input_tokens": cost_per_million,
        "kv_regret_percent": regret_percent,
    }
    assert "| IMDB-4 | r: 2, a: 1 | 65 | 1000 |" in (
        destination / "report.md").read_text()

    report_text = (destination / "report.md").read_text()
    assert "$/million input tokens | KV regret (%)" in report_text
    assert "| 95.9 |" in report_text

    def empty(spec, tables):
        output = execute(spec, tables)
        output.filter_answers = {
            key: table.slice(0, 0) for key, table in output.filter_answers.items()}
        output.join_answers = {
            key: table.slice(0, 0) for key, table in output.join_answers.items()}
        output.rows = output.rows.slice(0, 0)
        output.measurements = {"fresh_tokens": 0}
        return output

    empty_record = quail_b.run(
        empty, queries=["IMDB-4"], output_dir=tmp_path / "empty", root=tmp_path,
        gpu_hourly_rate_usd=3.6)
    empty_metrics = empty_record["queries"][0]["metrics"]
    assert empty_metrics["input_tokens"] == 0
    assert empty_metrics["cost_usd_per_million_input_tokens"] is None
    assert empty_metrics["kv_regret_percent"] is None

    def broken(spec, tables):
        output = execute(spec, tables)
        output.prompt_pieces = {"tokenizer": "test-tokenizer", "joins": []}
        return output

    with pytest.raises(ValueError, match="missing filter operators"):
        quail_b.run(broken, queries=["IMDB-4"], output_dir=tmp_path / "broken",
                    root=tmp_path)

    def without_fresh(spec, tables):
        output = execute(spec, tables)
        output.measurements = {}
        return output

    with pytest.raises(ValueError, match="fresh_tokens"):
        quail_b.run(without_fresh, queries=["IMDB-4"],
                    output_dir=tmp_path / "no-fresh", root=tmp_path)

    def too_few(spec, tables):
        output = execute(spec, tables)
        output.measurements = {"fresh_tokens": 1}
        return output

    with pytest.raises(ValueError, match="below the minimum"):
        quail_b.run(too_few, queries=["IMDB-4"],
                    output_dir=tmp_path / "too-few", root=tmp_path)


def test_saved_answers_survive_scoring_failure_and_can_move(tmp_path):
    import shutil

    collection = _inputs(tmp_path, 0.1)

    def execute(spec, tables):
        return quail_b.RunOutput(
            None, None, pa.table({"r": ["r0"], "a": ["a0"]}), runtime_s=2.0)

    destination = tmp_path / "run"
    quail_b.run(execute, queries=["IMDB-4"], output_dir=destination, root=tmp_path)
    record = json.loads((destination / "run.json").read_text())
    assert record["queries"][0]["metrics"]["document_pairs_per_second"] is None
    shutil.copytree(destination, tmp_path / "copied")
    quail_b.report(tmp_path / "copied", root=tmp_path)
    record["queries"][0]["files"]["rows"] = "../../outside.parquet"
    _write_json(destination / "run.json", record)
    with pytest.raises(ValueError, match="outside the query directory"):
        quail_b.report(destination, root=tmp_path)

    def missing_label(spec, tables):
        table = tmp_path / GROUND_TRUTH_ROOT / "label_sets/predicate_2/ls_0.1_2"
        labels = pq.read_table(table / "labels.parquet")
        pq.write_table(labels.slice(0, 1), table / "labels.parquet")
        return quail_b.RunOutput(
            None,
            {"join-1": pa.table({
                "r": ["r1"], "a": ["a0"], "answer": [True]})},
            pa.table({"r": ["r1"], "a": ["a0"]}), runtime_s=2.0)

    quail_b.run(
        missing_label, queries=["IMDB-4"], output_dir=tmp_path / "pinned",
        root=tmp_path, collection_id=collection)
    with pytest.raises(ValueError, match="loaded 1 rows"):
        quail_b.report(tmp_path / "pinned", root=tmp_path)
    assert (tmp_path / "pinned/IMDB-4/rows.parquet").exists()


def test_score_answers_are_saved_read_back_and_checked(tmp_path):
    from quail_b.run import _read_output, _save_output, _validate_output

    spec = quail_b.get_query("REL-AGENT-4")
    scores = pa.table({"t": ["tr0", "tr1"], "score": [0.9, 0.1]})
    rows = pa.table({"t": ["tr0"], "recovered_score": [0.9]})
    output = quail_b.RunOutput({}, {}, rows, runtime_s=1.0,
                               score_answers={"score-1": scores})
    paths = _save_output(tmp_path / "q", output, spec)
    assert paths["scores"] == [{"key": "score-1", "path": "scores-0.parquet"}]
    loaded = _read_output(tmp_path / "q", {"files": paths})
    assert loaded.score_answers["score-1"].equals(scores)
    tables = {"agent_traces": pa.table({"id": ["tr0", "tr1"]})}
    with pytest.raises(ValueError, match="unknown score operator"):
        _validate_output(spec, quail_b.RunOutput(
            {}, {}, rows, runtime_s=1.0,
            score_answers={"score-9": scores}), tables)
    with pytest.raises(ValueError, match="non-null floats"):
        _validate_output(spec, quail_b.RunOutput(
            {}, {}, rows, runtime_s=1.0, score_answers={"score-1": pa.table(
                {"t": ["tr0"], "score": [1]})}), tables)
