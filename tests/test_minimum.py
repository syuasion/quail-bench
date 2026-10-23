"""CPU checks for the prefix trie and the minimum input tokens of a run."""

import random

import numpy as np
import pyarrow as pa
import pytest

from quail_b.minimum import (
    DocumentTokens,
    _PartnerTrie,
    input_tokens,
    minimum_input_tokens,
    prefix_trie_size,
    token_metrics,
    validate_prompt_pieces,
)
from quail_b.queries import QuerySpec
from quail_b.scoring import RunOutput
from tools.make_substrait_plans import (
    Classify,
    Extract,
    Filter,
    Join,
    Scan,
    Score,
    Struct,
    build_plan,
)


def _spec(query_id, description, tree, select=None):
    return QuerySpec.from_plan(
        query_id, description, build_plan(tree, select=select))


def _encode(texts):
    return [[byte + 1 for byte in text.encode("utf-8")] for text in texts]


def _ids(text):
    return _encode([text])[0]


def _lcp(left, right):
    n = 0
    while n < min(len(left), len(right)) and left[n] == right[n]:
        n += 1
    return n


PRE = _ids("DOCUMENT:\n")
QUESTION = _ids("\n\nuseful?\nANSWER:")
FRAME = _ids("\n\nDoes it mention the aspect?")
LABEL = _ids("\n\nASPECT:\n")
TAIL = _ids("\nANSWER:")


def test_prefix_trie_size_counts_shared_prefixes_once():
    assert prefix_trie_size([[1, 2, 3], [1, 2, 4], [9]]) == 5
    assert prefix_trie_size([[1, 2], [1, 2]]) == 2
    assert prefix_trie_size([(), (7,)]) == 1
    assert prefix_trie_size([]) == 0


def test_minimum_input_tokens_counts_each_document_prefix_once_and_pairs_apart():
    spec = _spec(
        "TEST-1",
        "one filter then one join",
        Join(
            Filter(Scan("docs", "d", "body"), "useful {0}"),
            Scan("aspects", "a", "name"),
            ("d", "a"),
            "{0} mentions {1}",
        ),
    )
    corpus = {
        "docs": pa.table({"id": ["a", "b", "c"],
                          "body": ["same start one", "same start two", "other"]}),
        "aspects": pa.table({"id": ["x", "y"], "name": ["aa", "ab"]}),
    }
    pieces = validate_prompt_pieces(spec, {
        "tokenizer": "test", "preamble": PRE,
        "filters": [{"id": "filter-1", "tail": QUESTION}],
        "joins": [{"id": "join-1", "anchor": "d", "frame": FRAME,
                   "label": LABEL, "tail": TAIL}],
    })
    filter_answers = {"filter-1": pa.table({
        "d": ["a", "b", "c"], "answer": [True, True, False]})}
    join_answers = {"join-1": pa.table({
        "d": ["a", "a", "b", "b"], "a": ["x", "y", "x", "y"],
        "answer": [True, False, True, True]})}
    minimum = minimum_input_tokens(
        spec, pieces, filter_answers, join_answers,
        DocumentTokens(corpus, _encode))

    # the three documents share the preamble, and the first two share
    # "same start " (11 tokens) beyond it; every document gets the
    # question once, the two anchors get the frame and label once,
    # sharing the lead they have in common with the question; each
    # anchor's partners "aa" and "ab" share their first token
    pre = len(PRE)
    documents = 3 * pre + 14 + 14 + 5 - (pre + pre + 11)
    anchored = prefix_trie_size([QUESTION, FRAME + LABEL])
    pairs = 2 * (2 + len(TAIL)) - 1
    assert minimum == documents + len(QUESTION) + 2 * anchored + 2 * pairs
    full_inputs = (
        3 * (len(PRE) + len(QUESTION)) + 14 + 14 + 5
        + 4 * (len(PRE) + 14 + len(FRAME) + len(LABEL) + 2 + len(TAIL)))
    assert input_tokens(spec, pieces, filter_answers, join_answers,
                        DocumentTokens(corpus, _encode)) == full_inputs


def test_minimum_input_tokens_counts_a_document_once_across_uses():
    second = Filter(Scan("docs", "d2", "body"), "short {0}")
    second = Filter(second, "clear {0}")
    spec = _spec(
        "TEST-2",
        "a self join with two filter stages on one side",
        Join(Scan("docs", "d1", "body"), second, ("d1", "d2"), "{0} before {1}"),
    )
    corpus = {"docs": pa.table({"id": ["a", "b"], "body": ["alpha", "beta"]})}
    first, second = _ids("\n\nshort?\nANSWER:"), _ids("\n\nclear?\nANSWER:")
    pieces = validate_prompt_pieces(spec, {
        "tokenizer": "test", "preamble": PRE,
        "filters": [{"id": "filter-1", "tail": first},
                    {"id": "filter-2", "tail": second}],
        "joins": [{"id": "join-1", "anchor": "d1", "frame": FRAME,
                   "label": LABEL, "tail": TAIL}],
    })
    # both rows pass the first stage, only "alpha" reaches the second;
    # the join anchors on d1, the same two documents
    filter_answers = {
        "filter-1": pa.table({"d2": ["a", "b"], "answer": [True, True]}),
        "filter-2": pa.table({"d2": ["a"], "answer": [True]}),
    }
    join_answers = {"join-1": pa.table({
        "d1": ["a", "a", "b", "b"], "d2": ["a", "b", "a", "b"],
        "answer": [True, True, False, True]})}
    minimum = minimum_input_tokens(
        spec, pieces, filter_answers, join_answers,
        DocumentTokens(corpus, _encode))

    documents = 2 * len(PRE) + 5 + 4 - len(PRE)
    alpha = prefix_trie_size([first, second, FRAME + LABEL])
    beta = prefix_trie_size([first, FRAME + LABEL])
    pairs = 9 + 2 * len(TAIL)
    assert minimum == documents + alpha + beta + 2 * pairs
    full_inputs = (
        2 * (len(PRE) + len(first)) + 9
        + len(PRE) + len(second) + 5
        + 4 * (len(PRE) + len(FRAME) + len(LABEL) + len(TAIL)) + 36)
    assert input_tokens(spec, pieces, filter_answers, join_answers,
                        DocumentTokens(corpus, _encode)) == full_inputs


def test_minimum_input_tokens_of_a_join_is_the_trie_of_its_requests():
    spec = _spec(
        "TEST-5",
        "one join",
        Join(Scan("docs", "d", "body"), Scan("notes", "n", "text"),
             ("d", "n"), "{0} cites {1}"),
    )
    corpus = {
        "docs": pa.table({"id": ["r", "g"], "body": ["red apple", "green pear"]}),
        "notes": pa.table({"id": ["1", "2", "3", "4"],
                           "text": ["the cat sat", "the cat ran", "a dog",
                                    "the cat sat"]}),
    }
    pieces = validate_prompt_pieces(spec, {
        "tokenizer": "test", "preamble": PRE,
        "joins": [{"id": "join-1", "anchor": "d", "frame": FRAME,
                   "label": LABEL, "tail": TAIL}],
    })
    pairs = [("r", "1"), ("r", "2"), ("r", "3"), ("r", "4"),
             ("g", "1"), ("g", "3")]
    join_answers = {"join-1": pa.table({
        "d": [anchor for anchor, _ in pairs],
        "n": [partner for _, partner in pairs],
        "answer": [True] * len(pairs)})}
    minimum = minimum_input_tokens(
        spec, pieces, {}, join_answers, DocumentTokens(corpus, _encode))

    # with unlimited KV every distinct prefix of the requests is computed
    # once: the label once per anchor, and the partners of one anchor
    # sharing "the cat " and the whole of a repeated text
    bodies = {"r": "red apple", "g": "green pear"}
    texts = {"1": "the cat sat", "2": "the cat ran", "3": "a dog",
             "4": "the cat sat"}
    requests = [PRE + _ids(bodies[anchor]) + FRAME + LABEL
                + _ids(texts[partner]) + TAIL for anchor, partner in pairs]
    assert minimum == prefix_trie_size(requests)


def test_minimum_input_tokens_ask_a_repeated_text_once():
    spec = _spec("TEST-6", "one filter",
                 Filter(Scan("docs", "d", "body"), "useful {0}"))
    corpus = {"docs": pa.table({"id": ["a", "b", "c"],
                                "body": ["same text", "same text", "other"]})}
    pieces = validate_prompt_pieces(spec, {
        "tokenizer": "test", "preamble": PRE,
        "filters": [{"id": "filter-1", "tail": QUESTION}],
    })
    filter_answers = {"filter-1": pa.table({
        "d": ["a", "b", "c"], "answer": [True, True, False]})}
    documents = DocumentTokens(corpus, _encode)
    texts = [PRE + _ids(text) for text in ("same text", "other")]

    # rows a and b hold one text, so its question is asked once
    assert minimum_input_tokens(
        spec, pieces, filter_answers, {}, documents) == (
            prefix_trie_size([text + QUESTION for text in texts]))


def test_minimum_input_tokens_share_repeated_anchors():
    spec = _spec(
        "TEST-7",
        "one join",
        Join(Scan("docs", "d", "body"), Scan("notes", "n", "text"),
             ("d", "n"), "{0} cites {1}"),
    )
    corpus = {
        "docs": pa.table({"id": ["r", "s"], "body": ["red apple", "red apple"]}),
        "notes": pa.table({"id": ["1", "2"], "text": ["the cat", "a dog"]}),
    }
    pieces = validate_prompt_pieces(spec, {
        "tokenizer": "test", "preamble": PRE,
        "joins": [{"id": "join-1", "anchor": "d", "frame": FRAME,
                   "label": LABEL, "tail": TAIL}],
    })
    join_answers = {"join-1": pa.table({
        "d": ["r", "s"], "n": ["1", "2"], "answer": [True, False]})}
    documents = DocumentTokens(corpus, _encode)
    partners = [_ids(text) + TAIL for text in ("the cat", "a dog")]

    # r and s hold one text, so its frame and label come once and each
    # row's partner follows them
    assert minimum_input_tokens(spec, pieces, {}, join_answers, documents) == (
        prefix_trie_size([PRE + _ids("red apple") + FRAME + LABEL + partner
                          for partner in partners]))


def test_minimum_input_tokens_read_a_repeated_text_once():
    spec = _spec("TEST-9", "one extraction",
                 Extract(Scan("docs", "d", "body"), "When does it end?", "ends"),
                 ("d", Struct("answers", ("d.ends",))))
    corpus = {"docs": pa.table({"id": ["a", "b"], "body": ["ends May", "ends May"]})}
    tail = _ids("\n\nWhen does it end?\nANSWER:")
    pieces = validate_prompt_pieces(spec, {
        "tokenizer": "test", "preamble": PRE,
        "extracts": [{"id": "extract-1", "tail": tail}]})
    answers = {"extract-1": pa.table({"d": ["a", "b"], "ends": ["May", "May"]})}
    documents = DocumentTokens(corpus, _encode)

    # rows a and b hold one contract, so its question is read once
    assert minimum_input_tokens(spec, pieces, {}, {}, documents,
                                extract_answers=answers) == (
        len(PRE + _ids("ends May")) + len(tail))


def test_token_metrics_reject_fresh_tokens_below_one_ask_of_a_repeated_text(
        monkeypatch):
    monkeypatch.setattr("quail_b.minimum.load_tokenizer", lambda _: _encode)
    spec = _filter_spec()
    corpus = {"docs": pa.table({"id": ["a", "b"], "body": ["alpha", "alpha"]})}
    pieces = {"tokenizer": "test", "preamble": PRE,
              "filters": [{"id": "filter-1", "tail": QUESTION}]}
    answers = {"filter-1": pa.table({"d": ["a", "b"], "answer": [True, True]})}

    def metrics(fresh):
        return token_metrics(spec, RunOutput(
            answers, {}, pa.table({"d": ["a", "b"]}), runtime_s=1.0,
            measurements={"fresh_tokens": fresh}, prompt_pieces=pieces), corpus)

    once = len(PRE) + 5 + len(QUESTION)
    assert metrics(once)["minimum_tokens"] == once
    assert metrics(once)["regret_tokens"] == 0
    assert metrics(once + len(QUESTION))["regret_tokens"] == len(QUESTION)
    with pytest.raises(ValueError, match="below the minimum"):
        metrics(once - 1)


def test_partner_trie_size_matches_the_trie_of_any_subset():
    rng = random.Random(0)
    table = ("notes", "text")
    texts = {str(i): [rng.choice((1, 2, 3)) for _ in range(rng.randint(0, 6))]
             for i in range(60)}
    documents = {(*table, i): np.asarray(t, dtype=np.uint32)
                 for i, t in texts.items()}
    keys = [(table, i) for i in texts]
    trie = _PartnerTrie(documents, keys, TAIL)
    for _ in range(200):
        members = rng.sample(keys, rng.randint(1, len(keys)))
        assert trie.size(frozenset(members)) == prefix_trie_size(
            [texts[i] + TAIL for _, i in members])


def _filter_spec():
    return _spec(
        "TEST-0",
        "one filter",
        Filter(Scan("docs", "d", "body"), "useful {0}"),
    )


def _filter_output(**kwargs):
    return RunOutput(
        {"filter-1": pa.table({"d": ["a"], "answer": [True]})},
        {}, pa.table({"d": ["a"]}), runtime_s=1.0, **kwargs)


def test_token_metrics_record_fresh_tokens_without_prompt_pieces():
    spec = _filter_spec()
    output = _filter_output(measurements={
        "input_tokens": 50,
        "fresh_tokens": 40,
    })
    assert token_metrics(spec, output, {}) == {
        "input_tokens": 50,
        "fresh_tokens": 40, "minimum_tokens": None, "regret_tokens": None,
        "regret_approximate": False,
    }
    assert token_metrics(spec, _filter_output(), {}) == {
        "input_tokens": None,
        "fresh_tokens": None, "minimum_tokens": None, "regret_tokens": None,
        "regret_approximate": False,
    }


def test_token_metrics_require_fresh_tokens_with_prompt_pieces():
    spec = _filter_spec()
    pieces = {
        "tokenizer": "test", "preamble": PRE,
        "filters": [{"id": "filter-1", "tail": QUESTION}],
    }
    output = _filter_output(prompt_pieces=pieces)
    with pytest.raises(ValueError, match="fresh_tokens"):
        token_metrics(spec, output, {})
    output.measurements = {"fresh_tokens": True}
    with pytest.raises(ValueError, match="nonnegative"):
        token_metrics(spec, output, {})
    output.prompt_pieces = None
    output.measurements = {"fresh_tokens": -1}
    with pytest.raises(ValueError, match="nonnegative"):
        token_metrics(spec, output, {})
    output.measurements = {"input_tokens": True}
    with pytest.raises(ValueError, match="input_tokens"):
        token_metrics(spec, output, {})


def test_token_metrics_need_answers_with_prompt_pieces():
    spec = _filter_spec()
    pieces = {
        "tokenizer": "test", "preamble": PRE,
        "filters": [{"id": "filter-1", "tail": QUESTION}],
    }
    output = RunOutput(
        None, None, pa.table({"d": ["a"]}), runtime_s=1.0,
        measurements={"fresh_tokens": 10}, prompt_pieces=pieces)
    with pytest.raises(ValueError, match="filter and join answers"):
        token_metrics(spec, output, {})


@pytest.mark.parametrize("missing", [False, True])
def test_input_tokens_distinguish_empty_and_missing_stages(missing):
    spec = _filter_spec()
    pieces = validate_prompt_pieces(spec, {
        "tokenizer": "test", "preamble": PRE,
        "filters": [{"id": "filter-1", "tail": QUESTION}],
    })
    answers = {} if missing else {"filter-1": pa.table({
        "d": pa.array([], type=pa.string()),
        "answer": pa.array([], type=pa.bool_()),
    })}
    assert input_tokens(spec, pieces, answers, {}, DocumentTokens({}, _encode)) == (
        None if missing else 0)


def test_input_tokens_do_not_count_recomputed_kv(monkeypatch):
    monkeypatch.setattr("quail_b.minimum.load_tokenizer", lambda _: _encode)
    spec = _filter_spec()
    corpus = {"docs": pa.table({"id": ["a"], "body": ["alpha"]})}
    pieces = {
        "tokenizer": "test", "preamble": PRE,
        "filters": [{"id": "filter-1", "tail": QUESTION}],
    }
    counts = [token_metrics(spec, _filter_output(
        prompt_pieces=pieces, measurements={"fresh_tokens": fresh}), corpus)
        for fresh in (100, 200)]
    assert counts[0]["input_tokens"] == counts[1]["input_tokens"]
    assert counts[1]["regret_tokens"] - counts[0]["regret_tokens"] == 100
    reported = counts[0]["input_tokens"] // 2
    scaled = token_metrics(spec, _filter_output(prompt_pieces=pieces, measurements={
        "fresh_tokens": 0, "input_tokens": reported}), corpus)
    assert scaled == {
        "input_tokens": reported, "fresh_tokens": 0,
        "minimum_tokens": round(
            counts[0]["minimum_tokens"] * reported / counts[0]["input_tokens"]),
        "regret_tokens": 0, "regret_approximate": True,
    }


CLASSIFY_TAIL = _ids("\n\nwhich tone?\n- A: calm\n- B: angry\nANSWER:")


def test_minimum_input_tokens_count_a_classification_as_a_document_suffix():
    spec = _spec(
        "TEST-3",
        "one filter then one classification of the survivors",
        Classify(Filter(Scan("docs", "d", "body"), "useful {0}"),
                 "tone of {0}", ("calm", "angry"), "tone"),
        ("d", "d.tone"),
    )
    corpus = {"docs": pa.table({"id": ["a", "b"], "body": ["alpha", "beta"]})}
    pieces = validate_prompt_pieces(spec, {
        "tokenizer": "test", "preamble": PRE,
        "filters": [{"id": "filter-1", "tail": QUESTION}],
        "classifies": [{"id": "classify-1", "tail": CLASSIFY_TAIL}],
    })
    filter_answers = {"filter-1": pa.table({
        "d": ["a", "b"], "answer": [True, False]})}
    classify_answers = {"classify-1": pa.table({
        "d": ["a"], "label": ["calm"]})}
    tokens = DocumentTokens(corpus, _encode)
    minimum = minimum_input_tokens(
        spec, pieces, filter_answers, {}, tokens, classify_answers)

    # both documents share the preamble; "alpha" gets the filter question
    # and the classification tail, sharing the lead they have in common
    documents = 2 * len(PRE) + 5 + 4 - len(PRE)
    alpha = len(QUESTION) + len(CLASSIFY_TAIL) - _lcp(QUESTION, CLASSIFY_TAIL)
    assert minimum == documents + alpha + len(QUESTION)
    assert input_tokens(spec, pieces, filter_answers, {}, tokens,
                        classify_answers) == (
        2 * (len(PRE) + len(QUESTION)) + 9 + len(PRE) + len(CLASSIFY_TAIL) + 5)
    assert input_tokens(spec, pieces, filter_answers, {}, tokens) is None


def test_minimum_input_tokens_count_joined_rows_as_pairs_after_the_anchor():
    spec = _spec(
        "TEST-4",
        "one join then one classification of the joined rows",
        Classify(Join(Scan("docs", "d", "body"), Scan("aspects", "a", "name"),
                      ("d", "a"), "{0} mentions {1}"),
                 "how {0} treats {1}", ("well", "badly"), "treatment"),
        ("d", "a", "d.treatment"),
    )
    corpus = {
        "docs": pa.table({"id": ["a"], "body": ["alpha"]}),
        "aspects": pa.table({"id": ["x", "y"], "name": ["aa", "ab"]}),
    }
    classify_frame = _ids("\n\nHow does it treat the aspect?")
    pieces = validate_prompt_pieces(spec, {
        "tokenizer": "test", "preamble": PRE,
        "joins": [{"id": "join-1", "anchor": "d", "frame": FRAME,
                   "label": LABEL, "tail": TAIL}],
        "classifies": [{"id": "classify-1", "anchor": "d",
                        "frame": classify_frame, "label": LABEL,
                        "tail": CLASSIFY_TAIL}],
    })
    join_answers = {"join-1": pa.table({
        "d": ["a", "a"], "a": ["x", "y"], "answer": [True, False]})}
    classify_answers = {"classify-1": pa.table({
        "d": ["a"], "a": ["x"], "label": ["well"]})}
    tokens = DocumentTokens(corpus, _encode)
    minimum = minimum_input_tokens(
        spec, pieces, {}, join_answers, tokens, classify_answers)

    # the anchor is computed once with both frames and their label after
    # it; the join's partners "aa" and "ab" share their first token, and
    # the classified pair gets its own partner and tail
    anchored = prefix_trie_size([FRAME + LABEL, classify_frame + LABEL])
    joined = 2 * (2 + len(TAIL)) - 1
    classified = 2 + len(CLASSIFY_TAIL)
    assert minimum == len(PRE) + 5 + anchored + joined + classified


def test_prompt_pieces_name_every_classification_in_its_form():
    spec = _spec(
        "TEST-5",
        "one classification",
        Classify(Scan("docs", "d", "body"), "tone of {0}",
                 ("calm", "angry"), "tone"),
        ("d", "d.tone"),
    )
    with pytest.raises(ValueError, match="missing classify operators"):
        validate_prompt_pieces(spec, {"tokenizer": "test"})
    with pytest.raises(ValueError, match="unknown classify operator"):
        validate_prompt_pieces(spec, {"tokenizer": "test", "classifies": [
            {"id": "classify-1", "tail": [1]}, {"id": "classify-2"}]})
    checked = validate_prompt_pieces(spec, {"tokenizer": "test", "classifies": [
        {"id": "classify-1", "anchor": "d", "frame": [1], "tail": [2]}]})
    assert checked["classifies"] == [{"id": "classify-1", "tail": [2]}]
    joined = _spec(
        "TEST-6",
        "one classification of joined rows",
        Classify(Join(Scan("docs", "d", "body"), Scan("aspects", "a", "name"),
                      ("d", "a"), "{0} mentions {1}"),
                 "how {0} treats {1}", ("well", "badly"), "treatment"),
        ("d", "a", "d.treatment"),
    )
    join = {"id": "join-1", "anchor": "d"}
    with pytest.raises(ValueError, match="anchors on an alias"):
        validate_prompt_pieces(joined, {"tokenizer": "test", "joins": [join],
                                        "classifies": [{"id": "classify-1"}]})


def test_token_metrics_count_label_reading_as_regret(monkeypatch):
    monkeypatch.setattr("quail_b.minimum.load_tokenizer", lambda _: _encode)
    spec = _spec(
        "TEST-7",
        "one classification",
        Classify(Scan("docs", "d", "body"), "tone of {0}",
                 ("calm", "angry"), "tone"),
        ("d", "d.tone"),
    )
    corpus = {"docs": pa.table({"id": ["a"], "body": ["alpha"]})}
    pieces = {"tokenizer": "test", "preamble": PRE,
              "classifies": [{"id": "classify-1", "tail": CLASSIFY_TAIL}]}
    answers = {"classify-1": pa.table({"d": ["a"], "label": ["calm"]})}
    prompt = len(PRE) + 5 + len(CLASSIFY_TAIL)

    def metrics(measurements, classify_answers=answers):
        return token_metrics(spec, RunOutput(
            {}, {}, pa.table({"d": ["a"]}), runtime_s=1.0,
            measurements=measurements, prompt_pieces=pieces,
            classify_answers=classify_answers), corpus)

    # three label tokens fed after the cue are beyond the minimum
    assert metrics({"fresh_tokens": prompt + 3}) == {
        "input_tokens": prompt, "fresh_tokens": prompt + 3,
        "minimum_tokens": prompt, "regret_tokens": 3,
        "regret_approximate": False}
    with pytest.raises(ValueError, match="classify answers"):
        metrics({"fresh_tokens": prompt}, classify_answers=None)


def test_score_pieces_and_answers_count_a_score_as_a_document_suffix(
        monkeypatch):
    monkeypatch.setattr("quail_b.minimum.load_tokenizer", lambda _: _encode)
    spec = _spec(
        "TEST-7",
        "one score",
        Score(Scan("docs", "d", "body"), "useful {0}", "useful_score"),
        ("d", "d.useful_score"),
    )
    with pytest.raises(ValueError, match="missing score operators"):
        validate_prompt_pieces(spec, {"tokenizer": "test"})
    with pytest.raises(ValueError, match="unknown score operator"):
        validate_prompt_pieces(spec, {"tokenizer": "test", "scores": [
            {"id": "score-1", "tail": [1]}, {"id": "score-2"}]})
    pieces = validate_prompt_pieces(spec, {
        "tokenizer": "test", "preamble": PRE,
        "scores": [{"id": "score-1", "tail": QUESTION}]})
    assert pieces["scores"] == [{"id": "score-1", "tail": QUESTION}]
    corpus = {"docs": pa.table({
        "id": ["a", "b"], "body": ["same start one", "same start two"]})}
    scores = {"score-1": pa.table({"d": ["a", "b"], "score": [0.9, 0.1]})}
    documents = DocumentTokens(corpus, _encode)
    # the two documents share the preamble and "same start " (11
    # tokens); each gets the question once
    assert minimum_input_tokens(spec, pieces, {}, {}, documents,
                                score_answers=scores) == (
        len(PRE) + 11 + 3 + 3 + 2 * len(QUESTION))
    assert input_tokens(spec, pieces, {}, {}, documents,
                        score_answers=scores) == (
        2 * (len(PRE) + len(QUESTION)) + 14 + 14)
    assert input_tokens(spec, pieces, {}, {}, documents) is None
    rows = pa.table({"d": ["a"], "useful_score": [0.9]})
    with pytest.raises(ValueError, match="score answers"):
        token_metrics(spec, RunOutput(
            {}, {}, rows, runtime_s=1.0, measurements={"fresh_tokens": 100},
            prompt_pieces=pieces), corpus)
    metrics = token_metrics(spec, RunOutput(
        {}, {}, rows, runtime_s=1.0, measurements={"fresh_tokens": 100},
        prompt_pieces=pieces, score_answers=scores), corpus)
    assert metrics["input_tokens"] == 2 * (len(PRE) + len(QUESTION)) + 28
    assert metrics["minimum_tokens"] == len(PRE) + 17 + 2 * len(QUESTION)
    assert metrics["regret_tokens"] == 100 - metrics["minimum_tokens"]

