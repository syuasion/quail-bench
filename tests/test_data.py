"""CPU checks for the QUAIL-B document sets."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quail_b.data import (
    AGENT_TRACE_DOCUMENTS,
    AGENT_TRACE_MAX_TOKENS,
    AGENT_TRACE_TURN_INTERVAL,
    ASPECTS,
    CORPUS_COLUMNS,
    ISSUE_PATCH_MAX_CHARS,
    ISSUE_RUN_ISSUES,
    ISSUE_RUN_MAX_TOKENS,
    ISSUE_RUN_SOURCE_RUNS,
    ISSUE_RUNS_PER_ISSUE,
    LEPARD_POSITIVE_PAIRS,
    MESSAGE_MAX_CHARS,
    SALES_DEALS,
    SCENARIOS,
    SETS,
    SUPPORT_TASKS,
    SUPPORT_TRACES_PER_TASK,
    SUPPORT_TRIALS_PER_MODEL,
    WRENCH_MAX_TOKENS,
    WRENCH_OUTPUT_CHARS,
    WRENCH_TASKS,
    _agent_snapshot_boundaries,
    _agent_trace_rows,
    _issue_run_candidates,
    _issue_run_rows,
    _lepard_documents,
    _message_rows,
    _n_agent_documents,
    _n_issue_run_issues,
    _n_lepard_pairs,
    _n_sales_deals,
    _n_support_tasks,
    _n_wrench_tasks,
    _request,
    _sales_rows,
    _sample_lepard_pairs,
    _select_agent_snapshots,
    _support_rows,
    _support_source_tasks,
    _transcript,
    _wrench_rows,
    _wrench_tasks,
    _tripadvisor_date,
    _tripadvisor_manifest_identity,
    _tripadvisor_table_identity,
    build_tripadvisor,
    read_tripadvisor_manifest,
    tripadvisor_pair_tables,
)


def test_set_table_matches_design():
    assert SETS == {
        "reviews": 50_000,
        "reports": 5_000,
        "claims": 5_000,
        "agent_traces": AGENT_TRACE_DOCUMENTS,
        "policies": 1_000_000,
        "contracts": 449,
    }
    assert LEPARD_POSITIVE_PAIRS == 5_000
    assert _n_lepard_pairs(0.1) == 500
    assert len(ASPECTS) == 12
    assert len(SCENARIOS) == 100
    assert AGENT_TRACE_TURN_INTERVAL == 5
    assert AGENT_TRACE_MAX_TOKENS == 24_000
    assert _n_agent_documents(1.0) == 17_718
    assert _n_agent_documents(0.1) == 1_772


def test_agent_snapshots_include_every_fifth_turn_and_following_tool():
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "issue"},
    ]
    for turn in range(1, 12):
        messages.append({"role": "assistant", "content": f"action {turn}"})
        messages.append({"role": "tool", "content": f"result {turn}"})

    text, boundaries = _agent_snapshot_boundaries(messages)
    snapshots = [(turn, text[:end]) for turn, end in boundaries]

    assert [turn for turn, _snapshot in snapshots] == [5, 10]
    assert snapshots[0][1].endswith("[TOOL]\nresult 5")
    assert snapshots[1][1].startswith(snapshots[0][1])
    assert snapshots[1][1].endswith("[TOOL]\nresult 10")


class _WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


def test_agent_trace_rows_have_unique_ids_and_cumulative_text():
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "issue"},
    ]
    for turn in range(1, 11):
        messages.append({"role": "assistant", "content": f"action {turn}"})
        messages.append({"role": "tool", "content": f"result {turn}"})

    rows = _agent_trace_rows(7, messages, _WordTokenizer())

    assert [row["id"] for row in rows] == ["at0007-t005", "at0007-t010"]
    assert [row["turn_index"] for row in rows] == [5, 10]
    assert rows[1]["trace"].startswith(rows[0]["trace"])
    assert rows[0]["token_count"] < rows[1]["token_count"]


def test_agent_trace_rows_require_a_user_issue():
    messages = [
        {"role": "system", "content": "Solve the issue."},
        {"role": "assistant", "content": "action"},
    ]

    assert _agent_trace_rows(7, messages, _WordTokenizer()) == []


def test_lepard_samples_pairs_before_deduplicating_documents():
    context_a = "A" * 60
    context_b = "B" * 60
    context_c = "C" * 60
    rows = [
        ("d1", context_a, "p1"),
        ("d1", context_a, "p1"),
        ("d1", context_a, "p2"),
        ("d2", context_b, "p1"),
        ("d3", context_c, "missing"),
    ]
    passages = {"p1": "shared passage", "p2": "shared passage"}

    pairs = _sample_lepard_pairs(rows, passages, 10)
    contexts, passage_rows = _lepard_documents(pairs)

    assert len(pairs) == 3
    assert {row["destination_context"]: row["cited_passage_ids"]
            for row in contexts} == {
        context_a: ["p1", "p2"],
        context_b: ["p1"],
    }
    assert passage_rows == [
        {"id": "lp0", "passage_text": "shared passage",
         "passage_ids": ["p1", "p2"]},
    ]


def test_lepard_pair_sample_is_stable_and_nested():
    rows = [(f"d{i}", f"context {i} " + "x" * 50, f"p{i}")
            for i in range(20)]
    passages = {f"p{i}": f"passage {i}" for i in range(20)}

    small = _sample_lepard_pairs(rows, passages, 5)
    large = _sample_lepard_pairs(reversed(rows), passages, 10)

    assert small == large[:5]


def test_full_scale_takes_every_eligible_snapshot():
    import pytest

    snapshots = [[{"id": "a"}, {"id": "b"}], [{"id": "c"}]]

    assert _select_agent_snapshots(iter(snapshots), 2) == [
        {"id": "a"}, {"id": "b"}]
    assert _select_agent_snapshots(iter(snapshots), 5, full=True) == [
        {"id": "a"}, {"id": "b"}, {"id": "c"}]
    with pytest.raises(ValueError, match="expected 5"):
        _select_agent_snapshots(iter(snapshots), 5)


def _conversation():
    return [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "I want to return my order."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "find_order", "arguments": '{"id": 1}'}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "x" * 3000},
        {"role": "assistant", "content": "Returns are closed."},
        {"role": "user", "content": "That is not your policy."},
    ]


def test_message_rows_link_each_message_to_the_ones_before_it():
    rows = _message_rows("sp0001", _conversation())

    assert [row["id"] for row in rows] == [f"sp0001/{i}" for i in range(6)]
    assert rows[2]["content"] == '[tool call] find_order({"id": 1})'
    assert rows[3]["tool_call_id"] == "call-1"
    assert rows[3]["content"].endswith("\n[cut 1000 characters]")
    assert len(rows[3]["content"]) == MESSAGE_MAX_CHARS + len(
        "\n[cut 1000 characters]")
    assert rows[5] == {
        "id": "sp0001/5", "trace_id": "sp0001", "turn_index": 5,
        "role": "user", "content": "That is not your policy.",
        "tool_call_id": None, "prev_id": "sp0001/4",
        "prev_user_id": "sp0001/1", "prev_assistant_id": "sp0001/4"}
    assert rows[0]["prev_id"] is None


def test_transcript_leaves_out_the_system_prompt_and_cuts_tool_output():
    text = _transcript(_conversation(), tool_chars=5)

    assert text.startswith("[USER]\nI want to return my order.\n\n[ASSISTANT]")
    assert "policy" not in text.split("[USER]")[0]
    assert "[TOOL]\nxxxxx\n[cut 2995 characters]" in text
    assert _request(_conversation()) == "I want to return my order."


def test_support_rows_take_whole_tasks_in_order():
    runs = [{"domain": "retail", "model": model, "trial": trial,
             "reward": float(trial % 2), "messages": _conversation()}
            for model in ("sonnet-35-new", "gpt-4o") for trial in range(4)]
    tasks = [("retail-3", runs), ("airline-1", runs)]

    traces, messages = _support_rows(tasks, 2)

    assert len(traces) == 16 and len(messages) == 16 * 6
    assert [trace["id"] for trace in traces[:2]] == ["sp0000", "sp0001"]
    assert [(trace["model"], trace["trial"]) for trace in traces[:5]] == [
        ("gpt-4o", 0), ("gpt-4o", 1), ("gpt-4o", 2), ("gpt-4o", 3),
        ("sonnet-35-new", 0)]
    assert traces[8]["task_id"] == "airline-1"
    assert traces[1]["reward"] == 1.0
    assert traces[0]["message_count"] == 6
    assert messages[6]["trace_id"] == "sp0001"
    with pytest.raises(ValueError, match="expected 3"):
        _support_rows(tasks, 3)
    with pytest.raises(ValueError, match="has 7 runs"):
        _support_rows([("retail-3", runs[:7])], 1)


def test_issue_run_rows_keep_fitting_runs_of_candidate_issues():
    def run(issue, turns, resolved):
        messages = [{"role": "system", "content": "agent"},
                    {"role": "user", "content": f"issue {issue}"}]
        for turn in range(turns):
            messages.append({"role": "assistant", "content": f"step {turn}"})
            messages.append({"role": "tool", "content": "out " * 200})
        return {"instance_id": issue, "repo": "r", "resolved": resolved,
                "trajectory": messages,
                "model_patch": "+" * (ISSUE_PATCH_MAX_CHARS + 10)
                if resolved else None}

    class _WordTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    rows = [run("a", 2, 1) for _ in range(ISSUE_RUN_SOURCE_RUNS)]
    rows[1] = run("a", ISSUE_RUN_MAX_TOKENS, 0)    # too long, skipped
    rows[2] = run("a", 2, 0)                       # no patch
    rows += [run("b", ISSUE_RUN_MAX_TOKENS, 0)
             for _ in range(ISSUE_RUN_SOURCE_RUNS)]
    rows += [run("c", 1, 1) for _ in range(ISSUE_RUN_SOURCE_RUNS)]
    ids = [row["instance_id"] for row in rows] + ["d"] * 3
    batches = [pa.Table.from_pylist(rows[:5]), pa.Table.from_pylist(rows[5:])]

    assert set(_issue_run_candidates(ids, 2)) == {"a", "b", "c"}
    assert len(_issue_run_candidates(ids, 1)) == 2
    runs, messages = _issue_run_rows(batches, ["b", "a", "c"], 2,
                                     _WordTokenizer())

    assert [row["instance_id"] for row in runs] == (
        ["a"] * ISSUE_RUNS_PER_ISSUE + ["c"] * ISSUE_RUNS_PER_ISSUE)
    assert [row["id"] for row in runs[:2]] == ["ir00000", "ir00001"]
    assert runs[0]["request"] == "issue a"
    assert runs[0]["resolved"] == 1 and runs[0]["message_count"] == 6
    assert runs[0]["token_count"] == len(runs[0]["transcript"].split())
    assert "[cut" in runs[0]["transcript"]
    assert "characters cut" in runs[0]["patch"]
    assert runs[1]["resolved"] == 0 and runs[1]["patch"] == ""
    assert len(messages) == (6 + 4) * ISSUE_RUNS_PER_ISSUE
    assert messages[0]["trace_id"] == "ir00000"
    with pytest.raises(ValueError, match="expected 3"):
        _issue_run_rows(batches, ["b", "a", "c"], 3, _WordTokenizer())


def test_trace_set_sizes_scale_with_the_scale_factor():
    assert SUPPORT_TASKS == 165 and SUPPORT_TRACES_PER_TASK == 8
    assert _n_support_tasks(0.1) == 16
    assert _n_support_tasks(1.0) == 165
    assert ISSUE_RUN_ISSUES == 400 and ISSUE_RUNS_PER_ISSUE == 8
    assert _n_issue_run_issues(0.1) == 40
    assert _n_issue_run_issues(0.5) == 200
    assert set(CORPUS_COLUMNS) >= {
        "support_traces", "support_messages", "issue_runs", "issue_messages"}
    assert CORPUS_COLUMNS["support_messages"] == CORPUS_COLUMNS["issue_messages"]


def _wrench_trajectory(commands, output="ok"):
    steps = [{"step_id": 1, "source": "user", "message": "harness rules"}]
    for number, command in enumerate(commands, start=2):
        steps.append({
            "step_id": number, "source": "agent",
            "message": f"Analysis: step {number}",
            "tool_calls": [{"function_name": "bash_command",
                            "arguments": {"keystrokes": command + "\n"}}],
            "observation": {"results": [
                {"content": "New Terminal Output:\n" + output}]},
        })
    return steps


def test_wrench_rows_render_runs_and_steps_of_chosen_tasks():
    class _WordTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    index = [
        {"task_id": "t2", "model": "gpt-5.4", "trajectory_path": "a",
         "instruction": "Fix the server. ", "mode": "hack"},
        {"task_id": "t1", "model": "gpt-5.4", "trajectory_path": "b",
         "instruction": "Sort the file.", "mode": "baseline"},
        {"task_id": "t2", "model": "gemini-3.1-pro", "trajectory_path": "c",
         "instruction": "Fix the server.", "mode": "baseline"},
        {"task_id": "t2", "model": "claude-opus-4.6", "trajectory_path": "d",
         "instruction": "Fix the server.", "mode": "hack"},
    ]
    trajectories = {
        "a": _wrench_trajectory(["cat check.sh", "echo PASS > out"],
                                "x" * (WRENCH_OUTPUT_CHARS + 30)),
        "b": _wrench_trajectory(["sort f"]),
        "c": _wrench_trajectory(["systemctl restart app"]),
        "d": _wrench_trajectory(["word " * WRENCH_MAX_TOKENS]),
    }

    runs, steps = _wrench_rows(index, ["t2", "t1"], trajectories.get,
                               _WordTokenizer())

    assert [(run["id"], run["task_id"], run["mode"]) for run in runs] == [
        ("wr00000", "t2", "hack"), ("wr00001", "t2", "baseline"),
        ("wr00002", "t1", "baseline")]
    first = runs[0]["transcript"]
    assert first.startswith("[TASK]\nFix the server.\n\n[AGENT]\nAnalysis")
    assert "harness rules" not in first
    assert "[COMMANDS]\ncat check.sh" in first
    assert "characters cut ...]" in first
    assert runs[0]["step_count"] == 2
    assert runs[0]["token_count"] == len(first.split())
    assert [step["id"] for step in steps[:3]] == [
        "wr00000/1", "wr00000/2", "wr00001/1"]
    assert steps[1]["text"].startswith("[TASK]\nFix the server.\n\n[STEP]\n")
    assert "echo PASS > out" in steps[1]["text"]
    assert "cat check.sh" not in steps[1]["text"]
    assert steps[1]["run_id"] == "wr00000" and steps[1]["step_index"] == 2


def test_wrench_tasks_are_seeded_and_scale():
    index = [{"task_id": f"t{i}"} for i in range(50)] * 2
    assert _wrench_tasks(index, 5) == _wrench_tasks(index, 10)[:5]
    assert len(set(_wrench_tasks(index, 50))) == 50
    assert WRENCH_TASKS == 331
    assert _n_wrench_tasks(0.1) == 33 and _n_wrench_tasks(1.0) == 331


def test_sales_rows_order_calls_within_deals(monkeypatch):
    monkeypatch.setattr("quail_b.data.SALES_DEALS", {"b2b": 2, "b2c": 1})
    calls = {
        "b2b": [
            ("o1", "Closed", 900, "2024-03-05T10:00", "v2", "second call"),
            ("o1", "Closed", 900, "2024-02-19T10:00", "v1", "first call"),
            ("o2", "Discovery", 50.5, "2023-01-01T09:00", "v3", "only call"),
        ],
        "b2c": [("o9", "Quote", 7, "2023-05-05T08:00", "v9", "car call")],
    }

    rows = _sales_rows(calls, 1.0)

    assert [row["domain"] for row in rows] == ["b2b"] * 3 + ["b2c"]
    assert [row["id"] for row in rows] == [f"sc{i:05d}" for i in range(4)]
    o1 = [row for row in rows if row["deal_stage"] == "Closed"]
    assert [row["transcript"] for row in o1] == ["first call", "second call"]
    assert [row["call_index"] for row in o1] == [1, 2]
    assert o1[0]["prev_call_id"] is None
    assert o1[1]["prev_call_id"] == o1[0]["id"]
    assert len({row["deal_id"] for row in rows}) == 3
    assert isinstance(rows[-1]["deal_amount"], float)
    assert rows[-1]["prev_call_id"] is None
    with pytest.raises(ValueError, match="b2c has 2 deals"):
        _sales_rows({**calls, "b2c": calls["b2c"] + [
            ("o8", "Quote", 1, "2023-06-01T08:00", "v8", "x")]}, 1.0)


def test_sales_deals_scale_per_company():
    assert SALES_DEALS == {"b2b": 1_170, "b2c": 2_290}
    assert _n_sales_deals("b2b", 0.1) == 117
    assert _n_sales_deals("b2c", 1.0) == 2_290


def test_support_source_keeps_the_first_trials_of_each_model(
        monkeypatch, tmp_path):
    def download(url):
        trials = 8 if "sonnet" in url else 4
        path = tmp_path / url.rsplit("/", 1)[1]
        path.write_text(json.dumps([
            {"task_id": task, "trial": trial, "reward": 1.0, "traj": []}
            for task in range(3) for trial in range(trials)]))
        return path

    monkeypatch.setattr("quail_b.data._download", download)
    tasks = _support_source_tasks()

    assert len(tasks) == 6
    for _, runs in tasks:
        assert len(runs) == SUPPORT_TRACES_PER_TASK
        assert max(run["trial"] for run in runs) == SUPPORT_TRIALS_PER_MODEL - 1
def _tripadvisor_rows(hotels=6, reviews=24):
    return [{
        "id": hotel * 1000 + day, "offering_id": hotel,
        "title": "Hotel complaint", "text": f"The shower failed on day {day}.",
        "date": f"January {day}, 2012", "date_stayed": "January 2012",
        "author": {"id": "AUTHOR_METADATA", "username": "PRIVATE_NAME"},
        "ratings": {"overall": 2, "cleanliness": 3},
    } for hotel in range(1, hotels + 1) for day in range(1, reviews + 1)]


def test_tripadvisor_default_sampling_and_correct_hotel_id(tmp_path):
    built = build_tripadvisor(tmp_path, rows=_tripadvisor_rows())
    manifest, tables = read_tripadvisor_manifest(tmp_path)
    assert built == manifest
    assert manifest["counts"]["reviews"] == 100
    assert manifest["counts"]["same_hotel_pairs"] == 500
    assert len(tables["tripadvisor_earlier"]) == len(tables["tripadvisor_later"]) == 50
    for table in tables.values():
        assert table.column_names == ["id", "hotel_id", "body"]
        for row in table.to_pylist():
            assert int(row["id"]) // 1000 == int(row["hotel_id"])
            assert "AUTHOR_METADATA" not in row["body"]
            assert "PRIVATE_NAME" not in row["body"]
            assert "January 2012" not in row["body"]
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["reviews"][0]["source_review"]["ratings"]["overall"] == 2
    for hotel in metadata["hotels"]:
        cohort = [row for row in metadata["reviews"]
                  if row["hotel_id"] == hotel["hotel_id"]]
        assert max(row["post_date"] for row in cohort if row["cohort"] == "earlier") < (
            min(row["post_date"] for row in cohort if row["cohort"] == "later"))


def test_tripadvisor_default_identity_is_unchanged(tmp_path):
    baseline = build_tripadvisor(tmp_path / "default", rows=_tripadvisor_rows())
    explicit = build_tripadvisor(tmp_path / "explicit", rows=_tripadvisor_rows(),
                                pair_mode="earlier_later")
    assert baseline == explicit
    assert "pair_mode" not in baseline["settings"]
    assert baseline["corpus_full_hash"] == (
        "fecb65495f75091c11b78b1c15b36555abe7f38d1e06780dbf8cd70fee09fc37")


def test_tripadvisor_all_unique_pairs_preserves_sample_and_full_bodies(tmp_path):
    base = build_tripadvisor(tmp_path / "base", rows=_tripadvisor_rows())
    manifest = build_tripadvisor(tmp_path / "pairs", rows=_tripadvisor_rows(),
                                pair_mode="all_unique_pairs")
    _, base_tables = read_tripadvisor_manifest(tmp_path / "base")
    read, tables = read_tripadvisor_manifest(tmp_path / "pairs")
    assert read == manifest
    assert manifest["settings"]["pair_mode"] == "all_unique_pairs"
    assert manifest["counts"]["same_hotel_pairs"] == 950
    assert manifest["counts"]["earlier_later_pairs"] == 500
    assert manifest["corpus_id"] != base["corpus_id"]
    assert all(table.equals(tables[name]) for name, table in base_tables.items())
    reviews = {row["id"]: row for row in tables["tripadvisor_reviews"].to_pylist()}
    left = tables["tripadvisor_pair_left"].to_pylist()
    right = tables["tripadvisor_pair_right"].to_pylist()
    assert len(left) == len(right) == 950
    pairs = set()
    for first, second in zip(left, right, strict=True):
        assert first["review_id"] < second["review_id"]
        assert first["hotel_id"] == second["hotel_id"]
        assert first["pair_id"] == second["pair_id"]
        assert json.loads(first["pair_id"]) == [first["review_id"], second["review_id"]]
        for endpoint in (first, second):
            assert endpoint["body"] == reviews[endpoint["review_id"]]["body"]
            assert endpoint["hotel_id"] == reviews[endpoint["review_id"]]["hotel_id"]
        pairs.add((first["review_id"], second["review_id"]))
    assert len(pairs) == 950
    assert not pairs & {(second, first) for first, second in pairs}


def test_tripadvisor_pair_tables_are_stable_and_reject_duplicate_ids():
    reviews = pa.table({"id": ["3", "1", "2", "4"],
                        "hotel_id": ["10", "10", "10", "20"],
                        "body": ["C", "A", "B", "D"]})
    expected = tripadvisor_pair_tables(reviews)
    reordered = tripadvisor_pair_tables(reviews.take([3, 2, 1, 0]))
    assert all(table.equals(reordered[name]) for name, table in expected.items())
    assert len(expected["tripadvisor_pair_left"]) == 3
    with pytest.raises(ValueError, match="duplicate TripAdvisor"):
        tripadvisor_pair_tables(pa.concat_tables([reviews, reviews.slice(0, 1)]))


@pytest.mark.parametrize("fault", ["body", "pair_id", "review_id", "order"])
def test_tripadvisor_pair_reader_rejects_rehashed_endpoint_changes(tmp_path, fault):
    manifest = build_tripadvisor(tmp_path, rows=_tripadvisor_rows(),
                                pair_mode="all_unique_pairs")
    name = "tripadvisor_pair_left"
    path = tmp_path / f"{name}.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    if fault == "body":
        rows[0]["body"] += " changed"
    elif fault == "pair_id":
        rows[0]["pair_id"] = "[]"
    elif fault == "review_id":
        rows[0]["review_id"] = rows[-1]["review_id"]
    else:
        rows[0], rows[1] = rows[1], rows[0]
    changed = pa.Table.from_pylist(rows, schema=table.schema)
    pq.write_table(changed, path)
    manifest["tables"][name] = _tripadvisor_table_identity(changed)
    manifest = _tripadvisor_manifest_identity({
        key: value for key, value in manifest.items()
        if key not in ("corpus_id", "corpus_full_hash")})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unique source-review pairs"):
        read_tripadvisor_manifest(tmp_path)


def test_tripadvisor_nested_stable_sampling_and_explicit_truncation(tmp_path):
    rows = _tripadvisor_rows()
    small = build_tripadvisor(tmp_path / "a", rows=rows, hotel_limit=2)
    reordered = build_tripadvisor(tmp_path / "b", rows=reversed(rows), hotel_limit=2)
    large = build_tripadvisor(tmp_path / "c", rows=rows, hotel_limit=5)
    assert small == reordered
    _, first = read_tripadvisor_manifest(tmp_path / "a")
    _, second = read_tripadvisor_manifest(tmp_path / "c")
    for name in first:
        by_id = {row["id"]: row for row in second[name].to_pylist()}
        assert all(by_id[row["id"]] == row for row in first[name].to_pylist())
    clipped = build_tripadvisor(tmp_path / "d", rows=rows, max_chars=10)
    _, tables = read_tripadvisor_manifest(tmp_path / "d")
    assert clipped["counts"]["truncated_reviews"] == 100
    assert all(len(body) == 10
               for body in tables["tripadvisor_reviews"]["body"].to_pylist())
    assert clipped["corpus_id"] != large["corpus_id"]


def test_tripadvisor_english_dates_ignore_host_locale(tmp_path):
    import locale

    initial = locale.setlocale(locale.LC_TIME)
    baseline = build_tripadvisor(tmp_path / "baseline", rows=_tripadvisor_rows())
    try:
        for candidate in ("fr_FR.UTF-8", "fr_FR.utf8", "French_France.1252"):
            try:
                locale.setlocale(locale.LC_TIME, candidate)
                break
            except locale.Error:
                continue
        else:
            pytest.skip("French LC_TIME locale is unavailable")
        _tripadvisor_date.cache_clear()
        rebuilt = build_tripadvisor(tmp_path / "french", rows=_tripadvisor_rows())
        assert rebuilt == baseline
        assert read_tripadvisor_manifest(tmp_path / "french")[0] == baseline
        assert _tripadvisor_date("December 31, 2012") == "2012-12-31"
    finally:
        locale.setlocale(locale.LC_TIME, initial)
        _tripadvisor_date.cache_clear()


@pytest.mark.parametrize("pin", ["", False, 0])
@pytest.mark.parametrize("mode", ["rows", "zip"])
def test_tripadvisor_rejects_explicit_invalid_hash_pins(
        tmp_path, monkeypatch, pin, mode):
    import hashlib
    from zipfile import ZipFile

    settings = {"rows": _tripadvisor_rows()}
    if mode == "zip":
        archive = tmp_path / "review.txt.zip"
        with ZipFile(archive, "w") as stream:
            stream.writestr("review.txt", "\n".join(
                json.dumps(row) for row in settings["rows"]))
        monkeypatch.setattr("quail_b.data.TRIPADVISOR_SOURCE_SHA256",
                            hashlib.sha256(archive.read_bytes()).hexdigest())
        settings = {"archive_path": archive}
    with pytest.raises(ValueError, match="SHA256"):
        build_tripadvisor(tmp_path / "invalid-pin", source_sha256=pin, **settings)


@pytest.mark.parametrize("settings", [
    {"hotel_limit": 0}, {"hotel_limit": True}, {"reviews_per_hotel": 1},
    {"reviews_per_hotel": 2.0}, {"max_chars": 0}, {"seed": False},
])
def test_tripadvisor_rejects_invalid_settings(tmp_path, settings):
    with pytest.raises(ValueError):
        build_tripadvisor(tmp_path, rows=_tripadvisor_rows(), **settings)


@pytest.mark.parametrize("identity", [None, "", "01", "hotel-a", -1, True, 1.0])
def test_tripadvisor_rejects_malformed_identities(tmp_path, identity):
    rows = _tripadvisor_rows()
    rows[0]["offering_id"] = identity
    with pytest.raises(ValueError):
        build_tripadvisor(tmp_path, rows=rows)


def test_tripadvisor_rejects_duplicate_reviews_across_hotels(tmp_path):
    rows = _tripadvisor_rows()
    rows[-1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        build_tripadvisor(tmp_path, rows=rows)


def test_tripadvisor_rejects_insufficient_or_single_date_cohorts(tmp_path):
    rows = _tripadvisor_rows(hotels=1)
    with pytest.raises(ValueError, match="not enough"):
        build_tripadvisor(tmp_path / "few", rows=rows)
    for row in rows:
        row["date"] = "January 1, 2012"
    with pytest.raises(ValueError, match="not enough"):
        build_tripadvisor(tmp_path / "ties", rows=rows, hotel_limit=1)


def test_tripadvisor_excludes_bad_dates_and_empty_text_without_language_claim(tmp_path):
    rows = _tripadvisor_rows()
    rows[0]["text"] = "La chambre était bruyante."
    rows[1]["date"] = "not a date"
    rows[2]["text"] = ""
    built = build_tripadvisor(tmp_path, rows=rows)
    assert built["counts"]["excluded_text_or_date"] == 2
    assert "language" not in built["settings"]


def test_tripadvisor_local_zip_pin_and_cli(tmp_path):
    import hashlib
    import subprocess
    import sys
    from zipfile import ZipFile

    archive = tmp_path / "review.txt.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("review.txt", "\n".join(json.dumps(row)
                                              for row in _tripadvisor_rows()))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="SHA256"):
        build_tripadvisor(tmp_path / "bad", archive_path=archive,
                          source_sha256="0" * 64)
    assert not (tmp_path / "bad" / "manifest.json").exists()
    built = build_tripadvisor(tmp_path / "good", archive_path=archive,
                             source_sha256=digest)
    assert built["source"]["sha256"] == digest
    assert built["source"]["mode"] == "local_zip"
    result = subprocess.run([
        sys.executable, "-m", "quail_b", "prepare-tripadvisor", str(tmp_path / "cli"),
        "--archive-path", str(archive), "--source-sha256", digest,
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == built


@pytest.mark.parametrize("file", [
    "tripadvisor_reviews.parquet", "tripadvisor_earlier.parquet",
    "tripadvisor_later.parquet", "metadata.json", "manifest.json",
])
def test_tripadvisor_reader_rejects_changed_files(tmp_path, file):
    build_tripadvisor(tmp_path, rows=_tripadvisor_rows())
    path = tmp_path / file
    if file.endswith("parquet"):
        table = pq.read_table(path)
        body = table["body"].to_pylist()
        body[0] += " changed"
        pq.write_table(table.set_column(2, "body", pa.array(body)), path)
    else:
        value = json.loads(path.read_text())
        value["tampered"] = True
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        read_tripadvisor_manifest(tmp_path)


@pytest.mark.parametrize("fault", [
    "overlap", "source_id", "source_hotel", "source_body", "date", "cohort", "counts",
])
def test_tripadvisor_validates_provenance_after_rehash(tmp_path, fault):
    manifest = build_tripadvisor(tmp_path, rows=_tripadvisor_rows())
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    row = metadata["reviews"][0]
    if fault == "overlap":
        name = "tripadvisor_later"
        table = pq.read_table(tmp_path / f"{name}.parquet")
        early = pq.read_table(tmp_path / "tripadvisor_earlier.parquet").slice(0, 1)
        table = pa.concat_tables([early, table.slice(1)])
        pq.write_table(table, tmp_path / f"{name}.parquet")
        manifest["tables"][name] = _tripadvisor_table_identity(table)
    elif fault == "counts":
        manifest["counts"]["same_hotel_pairs"] += 1
    else:
        if fault == "source_id":
            row["source_review"]["id"] += 100_000
        elif fault == "source_hotel":
            row["source_review"]["offering_id"] += 100
        elif fault == "source_body":
            row["source_review"]["text"] += " changed"
        elif fault == "date":
            row["post_date"] = "2012-01-31"
        else:
            row["cohort"] = "later"
        metadata_path.write_text(json.dumps(metadata))
        from quail_b.data import _full_hash

        manifest["metadata"]["full_hash"] = _full_hash(metadata)
    manifest = _tripadvisor_manifest_identity({
        key: value for key, value in manifest.items()
        if key not in ("corpus_id", "corpus_full_hash")})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_tripadvisor_manifest(tmp_path)


@pytest.mark.parametrize("section, value", [
    ("source", None), ("settings", []), ("settings", {"hotel_limit": 5}),
    ("counts", {}), ("tables", None), ("metadata", {"file": "metadata.json"}),
])
def test_tripadvisor_rejects_malformed_manifest_sections(tmp_path, section, value):
    manifest = build_tripadvisor(tmp_path, rows=_tripadvisor_rows())
    manifest[section] = value
    manifest = _tripadvisor_manifest_identity({
        key: value for key, value in manifest.items()
        if key not in ("corpus_id", "corpus_full_hash")})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="malformed TripAdvisor manifest section"):
        read_tripadvisor_manifest(tmp_path)


@pytest.mark.parametrize("fault", ["list", "reviews", "missing_id", "hotels"])
def test_tripadvisor_rejects_malformed_provenance_metadata(tmp_path, fault):
    from quail_b.data import _full_hash

    manifest = build_tripadvisor(tmp_path, rows=_tripadvisor_rows())
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if fault == "list":
        metadata = []
    elif fault == "reviews":
        metadata["reviews"] = None
    elif fault == "missing_id":
        del metadata["reviews"][0]["id"]
    else:
        metadata["hotels"] = [None]
    metadata_path.write_text(json.dumps(metadata))
    manifest["metadata"]["full_hash"] = _full_hash(metadata)
    manifest = _tripadvisor_manifest_identity({
        key: value for key, value in manifest.items()
        if key not in ("corpus_id", "corpus_full_hash")})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_tripadvisor_manifest(tmp_path)
