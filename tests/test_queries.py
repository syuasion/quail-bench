"""CPU checks for the QUAIL-B query catalog."""

import json
from importlib.resources import files

import pytest
from substrait import plan_pb2

from quail_b import prompts
from quail_b.prompts import (
    CARDIOVASCULAR_REACTION,
    NEUROLOGICAL_REACTION,
    REACTION,
    SERIOUS_ADVERSE_EVENT,
)
from quail_b.queries import (
    FILTER_SELECTIVITY_ESTIMATES,
    IN_LIST_SELECTIVITY_ESTIMATES,
    JOIN_SELECTIVITY_ESTIMATES,
    PRIVACY_QUERIES,
    QUERIES,
    QUERY_FAMILY_WORKLOADS,
    QUERY_ORDER,
    QuerySpec,
    get_query,
    queries,
    query_family_name,
    split_query_families,
    split_query_ids,
)
from quail_b.run import _query_hash
from quail_b.substrait import (
    AI_JOIN_NAME,
    AND_NAME,
    EQUAL_NAME,
    inspect_plan,
)
from tools.make_substrait_plans import write_plans

TRIPADVISOR_QUERIES = ("TA-1", "TA-2", "TA-3", "REL-TA-1")


def test_catalog_has_the_85_default_queries_and_two_privacy_queries():
    assert len(QUERIES) == 85
    assert QUERY_ORDER == (
        *(f"IMDB-{i}" for i in range(1, 16)),
        *(f"BIO-{i}" for i in range(1, 7)),
        *(f"FEV-{i}" for i in range(1, 12)),
        *(f"LEP-{i}" for i in range(1, 7)),
        *(f"AGENT-{i}" for i in range(1, 6)),
        *(f"REL-AGENT-{i}" for i in range(1, 8)),
        *(f"SUPPORT-{i}" for i in range(1, 7)),
        *(f"RUNS-{i}" for i in range(1, 5)),
        *(f"WRENCH-{i}" for i in range(1, 6)),
        *(f"SALES-{i}" for i in range(1, 6)),
        *(f"CUAD-{i}" for i in range(1, 16)),
    )
    assert [spec.id for spec in PRIVACY_QUERIES] == ["PRIV-1", "PRIV-2"]
    assert list(queries(include_privacy=True)) == [*QUERY_ORDER, "PRIV-1", "PRIV-2"]
    for spec in QUERIES:
        info = inspect_plan(spec.plan)
        assert all(
            filter_spec.prompt in FILTER_SELECTIVITY_ESTIMATES
            for filter_spec in info.filters
        ), spec.id
        assert all(
            join.prompt in JOIN_SELECTIVITY_ESTIMATES
            for join in info.joins
        ), spec.id
        assert all(
            (info.classify_output(in_list.output).prompt,
             frozenset(in_list.accepted)) in IN_LIST_SELECTIVITY_ESTIMATES
            for in_list in info.in_lists
        ), spec.id
        if not info.classifies and not info.relational:
            answers = {extract.output for extract in info.extracts}
            assert all(name.endswith(".id") or name.split(".")[-1] in answers
                       for name in info.select), spec.id
    for spec in PRIVACY_QUERIES:
        info = inspect_plan(spec.plan)
        assert any(
            filter_spec.prompt not in FILTER_SELECTIVITY_ESTIMATES
            for filter_spec in info.filters
        ), spec.id


def test_query_spec_contains_only_identity_and_plan_bytes():
    assert tuple(QuerySpec.__dataclass_fields__) == (
        "id",
        "description",
        "plan_bytes",
    )


@pytest.mark.parametrize("query_id", ["BIO-1", "BIO-3"])
def test_biodex_filters_match_the_replacement_reference(query_id):
    info = inspect_plan(queries()[query_id].plan)
    assert [(op.id, op.relation, op.prompt) for op in info.filters] == [
        ("filter-1", "r", SERIOUS_ADVERSE_EVENT),
    ]
    assert len(info.joins) == (1 if query_id == "BIO-3" else 0)


def test_bio_4_filters_each_input_and_reuses_the_report_in_both_joins():
    info = inspect_plan(queries()["BIO-4"].plan)
    assert [(op.relation, op.prompt) for op in info.filters] == [
        ("r", SERIOUS_ADVERSE_EVENT),
        ("n", NEUROLOGICAL_REACTION),
        ("c", CARDIOVASCULAR_REACTION),
    ]
    assert [(op.relations, op.prompt) for op in info.joins] == [
        (("r", "n"), REACTION),
        (("r", "c"), REACTION),
    ]
    assert [(op.alias, op.table) for op in info.relations] == [
        ("r", "reports"), ("n", "terms"), ("c", "terms"),
    ]
    assert info.select == ("r.id", "n.id", "c.id")
    assert FILTER_SELECTIVITY_ESTIMATES[NEUROLOGICAL_REACTION] == 505 / 1127
    assert FILTER_SELECTIVITY_ESTIMATES[CARDIOVASCULAR_REACTION] == 394 / 1127


def test_filters_are_substrait_relations_over_their_input():
    spec = queries()["IMDB-4"]

    plan = spec.plan
    assert isinstance(plan, plan_pb2.Plan)
    info = inspect_plan(plan)
    assert [
        (relation.alias, relation.table, relation.text_column)
        for relation in info.relations
    ] == [
        ("r", "reviews", "body"),
        ("a", "aspects", "aspect"),
    ]
    assert [
        (operator.id, operator.relation) for operator in info.filters
    ] == [
        ("filter-1", "r"),
        ("filter-2", "r"),
    ]
    assert [operator.id for operator in info.joins] == ["join-1"]

    project = plan.relations[0].root.input.project
    join = project.input.join
    assert join.common.hint.alias == "join-1"
    assert join.left.filter.common.hint.alias == "filter-2"
    assert join.left.filter.input.filter.common.hint.alias == "filter-1"
    assert join.left.filter.input.filter.input.read.common.hint.alias == "r"
    assert join.right.read.common.hint.alias == "a"
    assert not plan.expected_type_urls


def test_fev_10_combines_ai_and_ordinary_join_conditions():
    spec = queries()["FEV-10"]
    plan = spec.plan
    names = {
        declaration.extension_function.function_anchor:
        declaration.extension_function.name
        for declaration in plan.extensions
    }
    assert set(names.values()) == {
        AI_JOIN_NAME,
        AND_NAME,
        EQUAL_NAME,
        "ai_filter:str_str",
    }
    condition = plan.relations[0].root.input.project.input.join.expression
    assert names[condition.scalar_function.function_reference] == AND_NAME
    assert {
        names[argument.value.scalar_function.function_reference]
        for argument in condition.scalar_function.arguments
    } == {AI_JOIN_NAME, EQUAL_NAME}
    join = plan.relations[0].root.input.project.input.join
    assert tuple(join.left.filter.input.read.base_schema.names) == (
        "id",
        "claim",
        "evidence_wiki_url",
    )
    assert tuple(join.right.filter.input.read.base_schema.names) == (
        "id",
        "text",
    )


def test_repeated_tables_and_join_order_round_trip_through_substrait():
    spec = queries()["IMDB-9"]
    details = inspect_plan(spec.plan)

    assert [(relation.alias, relation.table) for relation in details.relations] == [
        ("r1", "reviews"),
        ("a1", "aspects"),
        ("r2", "reviews"),
        ("a2", "aspects"),
    ]
    assert [join.relations for join in details.joins] == [
        ("r1", "a1"),
        ("r2", "a1"),
        ("r2", "a2"),
    ]


def test_definition_hash_uses_query_semantics_not_protobuf_bytes():
    original = queries()["IMDB-4"]
    copy = QuerySpec.from_plan(original.id, original.description, original.plan)

    assert copy.plan_bytes == original.plan_bytes
    assert _query_hash(copy) == _query_hash(original)
    different_wire = original.plan
    different_wire.version.producer = "another-producer"
    same_query = QuerySpec.from_plan(
        original.id,
        original.description,
        different_wire,
    )
    assert same_query.plan_bytes != original.plan_bytes
    assert _query_hash(same_query) == _query_hash(original)
    changed_description = QuerySpec.from_plan(
        original.id,
        "changed",
        original.plan,
    )
    assert _query_hash(changed_description) == _query_hash(original)
    changed_prompt = original.plan
    first_filter = (
        changed_prompt.relations[0]
        .root.input.project.input.join.left.filter.input.filter
    )
    first_filter.condition.scalar_function.arguments[0].value.literal.string = (
        "changed"
    )
    changed_query = QuerySpec.from_plan(
        original.id,
        original.description,
        changed_prompt,
    )
    assert _query_hash(changed_query) != _query_hash(original)


def test_ai_extension_definition_is_packaged():
    extension = files("quail_b").joinpath("substrait_extensions.yaml").read_text()

    assert "urn: extension:org.fsdatalab.quail_b:functions_ai" in extension
    assert "name: ai_filter" in extension
    assert "name: ai_join" in extension
    assert "name: ai_classify" in extension
    assert "name: ai_score" in extension
    assert "name: ai_extract" in extension
    # a filter, a score, an extraction, a one-document classification,
    # and a pair: prompt, two documents, labels, descriptions
    assert extension.count("- name: document") == 6


def test_substrait_plans_are_packaged():
    package = files("quail_b")
    catalog = package.joinpath("plans", "catalog.json")
    entries = json.loads(catalog.read_text())

    assert len(entries) == 104
    assert tuple(entry["id"] for entry in entries
                 if entry.get("draft"))[-4:] == TRIPADVISOR_QUERIES
    assert all(
        package.joinpath("plans", f"{entry['id']}.json").is_file()
        for entry in entries
    )


def test_checked_in_plans_equal_the_generator_output(tmp_path):
    write_plans(tmp_path)
    package = files("quail_b").joinpath("plans")

    generated = sorted(path.name for path in tmp_path.iterdir())
    assert generated == sorted(path.name for path in package.iterdir())
    for name in generated:
        assert (tmp_path / name).read_text() == package.joinpath(name).read_text(), (
            f"{name} differs; run tools/make_substrait_plans.py"
        )


def test_parallel_query_split_matches_stock_vllm():
    assert split_query_ids(QUERY_ORDER, 4) == (
        QUERY_ORDER[0:22],
        QUERY_ORDER[22:43],
        QUERY_ORDER[43:64],
        QUERY_ORDER[64:85],
    )


def test_query_family_split_matches_benchmark_catalog():
    assert QUERY_FAMILY_WORKLOADS == {
        "IMDB": "imdb",
        "BIO": "biodex",
        "FEV": "fever",
        "LEP": "lepard",
        "AGENT": "agent",
        "REL": "relational",
        "SUPPORT": "support",
        "RUNS": "runs",
        "WRENCH": "wrench",
        "SALES": "sales",
        "CUAD": "cuad",
        "SEC": "sec",
        "TA": "tripadvisor",
    }
    assert split_query_families(QUERY_ORDER) == (
        QUERY_ORDER[0:15],
        QUERY_ORDER[15:21],
        QUERY_ORDER[21:32],
        QUERY_ORDER[32:38],
        QUERY_ORDER[38:43],
        QUERY_ORDER[43:50],
        QUERY_ORDER[50:56],
        QUERY_ORDER[56:60],
        QUERY_ORDER[60:65],
        QUERY_ORDER[65:70],
        QUERY_ORDER[70:85],
    )
    assert query_family_name(QUERY_ORDER[0:15]) == "imdb"
    assert query_family_name(QUERY_ORDER[43:50]) == "relational"


def test_tripadvisor_plans_are_pending_and_reuse_existing_operators():
    expected = TRIPADVISOR_QUERIES
    assert all(query_id not in queries() for query_id in expected)
    assert tuple(queries(include_draft=True))[-4:] == expected
    assert split_query_families(expected) == (("REL-TA-1",), expected[:-1])
    assert query_family_name(expected[:-1]) == "tripadvisor"
    assert query_family_name(expected[-1:]) == "relational"
    assert [item.prompt for item in get_query("TA-1").info.filters] == [
        prompts.TA_1_1, prompts.TA_1_2]
    assert get_query("TA-2").info.select == ("r.id", "r.department")


def test_tripadvisor_join_pairs_each_review_once_with_later_reviews_of_its_hotel():
    info = get_query("TA-3").info
    assert [(item.alias, item.table, item.text_column) for item in info.relations] == [
        ("r1", "tripadvisor_reviews", "body"),
        ("r2", "tripadvisor_reviews", "body"),
    ]
    (join,) = info.joins
    assert join.relations == ("r1", "r2")
    assert join.on == (("hotel_id", "hotel_id"),)
    assert join.comparisons == (("id", "<", "id"),)
    assert join.prompt == prompts.TA_3


def test_tripadvisor_join_filters_each_endpoint_for_an_explicit_complaint():
    spec = get_query("TA-3")
    expected_prompt = (
        "Does DOCUMENT {0} explicitly report a concrete hotel problem or complaint? "
        "Answer TRUE if it does, otherwise FALSE."
    )
    assert prompts.TA_3_1 == expected_prompt
    assert [(item.id, item.relation, item.prompt) for item in spec.info.filters] == [
        ("filter-1", "r1", expected_prompt),
        ("filter-2", "r2", expected_prompt),
    ]
    original_plan = spec.plan
    join = original_plan.relations[0].root.input.project.input.join
    join.left.CopyFrom(join.left.filter.input)
    join.right.CopyFrom(join.right.filter.input)
    original = QuerySpec.from_plan(spec.id, spec.description, original_plan)
    assert not original.info.filters
    assert original.info.joins == spec.info.joins
    assert _query_hash(original) != _query_hash(spec)


def test_tripadvisor_join_requires_a_matching_concrete_problem_in_both_reviews():
    spec = get_query("TA-3")
    (join,) = spec.info.joins
    assert (spec.id, join.id, spec.info.select) == (
        "TA-3", "join-1", ("r1.id", "r2.id"))
    expected_prompt = (
        "Do BOTH hotel reviews in DOCUMENT {0} and DOCUMENT {1} explicitly report "
        "the SAME specific hotel problem? Answer TRUE only if each review describes "
        "a concrete hotel failure or complaint and the problems match. Answer FALSE "
        "if either review reports no problem, contains only praise, or describes a "
        "different problem. Shared amenities, positive experiences, broad topics, "
        "and general sentiment do not count as a match."
    )
    assert join.prompt == expected_prompt


def test_tripadvisor_join_revision_does_not_reuse_the_old_definition_hash():
    revised = get_query("TA-3")
    old_plan = revised.plan
    old_prompt = (
        old_plan.relations[0].root.input.project.input.join.expression
        .scalar_function.arguments[0].value.scalar_function.arguments[0]
        .value.literal
    )
    old_prompt.string = (
        "Do the hotel reviews in DOCUMENT {0} and DOCUMENT {1} report the "
        "same specific hotel problem, rather than just the same broad topic?"
    )
    original = QuerySpec.from_plan(revised.id, revised.description, old_plan)
    assert original.info.joins[0].id == revised.info.joins[0].id
    assert original.info.relations == revised.info.relations
    assert original.info.joins[0].on == revised.info.joins[0].on
    assert _query_hash(original) != _query_hash(revised)


def test_tripadvisor_score_is_unary_boolean_with_stable_top_ten():
    info = get_query("REL-TA-1").info
    assert [(item.alias, item.table) for item in info.relations] == [
        ("r", "tripadvisor_reviews")]
    (score,) = info.scores
    assert (score.relation, score.prompt, score.output) == (
        "r", prompts.REL_TA_1, "forced_move_score")
    assert info.select == ("r.id", "r.forced_move_score")
    assert info.sort.keys == (("r.forced_move_score", True), ("r.id", False))
    assert (info.fetch.count, info.fetch.offset) == (10, 0)


def test_query_family_rejects_mixed_or_unknown_queries():
    with pytest.raises(ValueError, match="expected one query family"):
        query_family_name(("IMDB-1", "BIO-1"))
    with pytest.raises(ValueError, match="unknown query family"):
        split_query_families(("OTHER-1",))


@pytest.mark.parametrize("query_id, expected", [
    ("IMDB-2", "f3b93b898b0d631fb451046b072920cb81f12d5aabb8dc1b853f913030b4f45e"),
    ("LEP-5", "6ca4bd71ae0304b98448d295d35213853c2d52242f8cc5bf23a817b0b9889a88"),
    # a one-document classification with an IN-list filter
    ("IMDB-14", "80c0d0927d64080e0d03936da550ce7c96ac0fc88bba22f8e0b2b793c35b9de9"),
])
def test_raw_query_hash_matches_before_chat(query_id, expected, monkeypatch):
    import importlib

    run_module = importlib.import_module("quail_b.run")
    spec = get_query(query_id)
    assert run_module._query_hash(spec) == expected
    monkeypatch.setattr(run_module, "PROMPT_FORMAT", "qwen3-chat-nonthinking-v1")
    assert run_module._query_hash(spec) != expected


def test_a_join_comparison_written_right_side_first_is_read_left_side_first():
    import pyarrow as pa

    from quail_b.scoring import _apply_conditions

    spec = get_query("TA-3")
    plan = plan_pb2.Plan()
    plan.CopyFrom(spec.plan)
    names = {declaration.extension_function.function_anchor:
             declaration.extension_function.name
             for declaration in plan.extensions}
    join = plan.relations[0].root.input.project.input.join
    (less,) = [argument.value.scalar_function
               for argument in join.expression.scalar_function.arguments
               if names[argument.value.scalar_function.function_reference]
               == "lt:any_any"]
    first, second = (argument.value for argument in less.arguments)
    swapped = (second.SerializeToString(), first.SerializeToString())
    less.arguments[0].value.ParseFromString(swapped[0])
    less.arguments[1].value.ParseFromString(swapped[1])
    reversed_spec = QuerySpec.from_plan(spec.id, spec.description, plan)
    (reversed_join,) = reversed_spec.info.joins
    assert reversed_join.comparisons == (("id", ">", "id"),)

    reviews = pa.table({"id": ["1", "2", "3"], "hotel_id": ["10", "10", "20"],
                        "body": ["a", "b", "c"]})
    pairs = pa.table({"r1": ["1", "2", "1", "1"], "r2": ["2", "1", "1", "3"]})
    corpus = {"tripadvisor_reviews": reviews}
    (join_spec,) = spec.info.joins
    kept = _apply_conditions(pairs, join_spec, spec, corpus)
    assert kept.to_pydict() == {"r1": ["1"], "r2": ["2"]}
    kept = _apply_conditions(pairs, reversed_join, reversed_spec, corpus)
    assert kept.to_pydict() == {"r1": ["2"], "r2": ["1"]}
