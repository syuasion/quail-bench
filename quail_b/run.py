"""Run engine callbacks and save reproducible benchmark results."""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from quail_b._files import download_cache
from quail_b.benchmark import load_benchmark
from quail_b.data import read_tripadvisor_manifest
from quail_b.minimum import token_metrics
from quail_b.relational import apply_tail
from quail_b.rendering import PROMPT_FORMAT
from quail_b.scoring import (
    RunOutput,
    _apply_conditions,
    answer_labels,
    corpus_ids,
    encode_ids,
    evaluate,
    implied_row_count,
    implied_rows_mask,
    output_columns,
    scores_from_answers,
    selected_extracts,
)
from quail_b.substrait import (
    Classify,
    ColumnTest,
    Extract,
    Filter,
    InList,
    Join,
    Score,
    output_name,
)

RUN_SCHEMA_VERSION = 2


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def _query_hash(spec):
    plan = spec.plan
    urns = {
        extension.extension_urn_anchor: extension.urn
        for extension in plan.extension_urns
    }
    functions = []
    for declaration in plan.extensions:
        if not declaration.HasField("extension_function"):
            continue
        function = declaration.extension_function
        functions.append(
            (
                urns[function.extension_urn_reference],
                function.name,
            )
        )
    functions.sort()
    operators = []
    for operator in spec.info.operators:
        if isinstance(operator, Filter):
            operators.append({
                "kind": "filter",
                "id": operator.id,
                "relation": operator.relation,
                "prompt": operator.prompt,
            })
        elif isinstance(operator, Classify):
            operators.append({
                "kind": "classify",
                "id": operator.id,
                "relation": operator.relation,
                "prompt": operator.prompt,
                "labels": operator.labels,
                "descriptions": operator.descriptions,
                "output": operator.output,
                # only a classification of joined rows names a partner, so the
                # hashes of the one-document queries stay as published
                **({"partner": operator.partner}
                   if operator.partner is not None else {}),
            })
        elif isinstance(operator, Extract):
            operators.append({
                "kind": "extract",
                "id": operator.id,
                "relation": operator.relation,
                "prompt": operator.prompt,
                "output": operator.output,
            })
        elif isinstance(operator, InList):
            operators.append({
                "kind": "in_list",
                "id": operator.id,
                "relation": operator.relation,
                "output": operator.output,
                "accepted": operator.accepted,
            })
        elif isinstance(operator, ColumnTest):
            operators.append({
                "kind": "column_test",
                "id": operator.id,
                "relation": operator.relation,
                "column": operator.column,
                "comparison": operator.comparison,
                "value": operator.value,
            })
        elif isinstance(operator, Score):
            operators.append({
                "kind": "score",
                "id": operator.id,
                "relation": operator.relation,
                "prompt": operator.prompt,
                "output": operator.output,
            })
        else:
            assert isinstance(operator, Join)
            operators.append({
                "kind": "join",
                "id": operator.id,
                "relations": operator.relations,
                "prompt": operator.prompt,
                "on": operator.on,
            })
    definition = {
        "substrait_version": [
            plan.version.major_number,
            plan.version.minor_number,
            plan.version.patch_number,
        ],
        "functions": functions,
        "relations": [
            {
                "alias": relation.alias,
                "table": relation.table,
                "text_column": relation.text_column,
            }
            for relation in spec.info.relations
        ],
        "operators": operators,
        "select": spec.info.select,
    }
    if spec.info.tail:
        # only a relational query has a tail, so the published hashes
        # of the other queries stay as they are
        definition["tail"] = [
            {"kind": type(step).__name__.lstrip("_").lower(),
             **{key: value for key, value in vars(step).items()}}
            for step in spec.info.tail]
    if PROMPT_FORMAT != "raw-v1":
        definition["prompt_format"] = PROMPT_FORMAT
    encoded = json.dumps(
        definition,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _save_output(directory, output, spec):
    directory.mkdir()
    (directory / "plan.substrait").write_bytes(spec.plan_bytes)
    pq.write_table(output.rows, directory / "rows.parquet", compression="zstd")
    paths = {
        "plan": "plan.substrait",
        "rows": "rows.parquet",
        "filters": None,
        "joins": None,
    }
    if output.classify_answers is not None:
        paths["classifications"] = None
    if output.score_answers is not None:
        paths["scores"] = None
    if output.extract_answers is not None:
        paths["extracts"] = None
    for kind, answers in (
            ("filters", output.filter_answers), ("joins", output.join_answers),
            ("classifications", output.classify_answers),
            ("scores", output.score_answers),
            ("extracts", output.extract_answers)):
        if answers is None:
            continue
        paths[kind] = []
        for index, (key, table) in enumerate(answers.items()):
            name = f"{kind}-{index}.parquet"
            pq.write_table(table, directory / name, compression="zstd")
            paths[kind].append({"key": key, "path": name})
    if output.prompt_pieces is not None:
        _write_json(directory / "prompt_pieces.json", output.prompt_pieces)
        paths["prompt_pieces"] = "prompt_pieces.json"
    return paths


def _read_output(directory, record, rows=True):
    """Read a saved output; rows=False skips the rows table, which can be huge."""
    def file(name):
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("answer file is outside the query directory")
        return path

    paths = record["files"]
    filters = paths["filters"]
    joins = paths["joins"]
    classifications = paths.get("classifications")
    scores = paths.get("scores")
    extracts = paths.get("extracts")
    pieces = paths.get("prompt_pieces")
    return RunOutput(
        None if filters is None else {
            item["key"]: pq.read_table(file(item["path"])) for item in filters},
        None if joins is None else {
            item["key"]: pq.read_table(file(item["path"])) for item in joins},
        pq.read_table(file(paths["rows"])) if rows else None,
        record.get("runtime_s"),
        record.get("measurements", {}),
        None if pieces is None else json.loads(file(pieces).read_text()),
        None if classifications is None else {
            item["key"]: pq.read_table(file(item["path"]))
            for item in classifications},
        None if scores is None else {
            item["key"]: pq.read_table(file(item["path"]))
            for item in scores},
        None if extracts is None else {
            item["key"]: pq.read_table(file(item["path"]))
            for item in extracts})


ROW_SAMPLE = 100_000    # rows of a traced result checked one by one


def _sample_rows(table, count):
    """Up to count rows of a table, evenly spaced.

    Taken chunk by chunk: a take over the whole table concatenates each
    column first, and a string column of hundreds of millions of rows
    overflows Arrow's 32-bit string offsets.
    """
    if table.num_rows <= count:
        return table
    step = table.num_rows // count
    last = step * count
    pieces = []
    start = 0
    for batch in table.to_batches():
        end = start + batch.num_rows
        first = -(-start // step) * step
        indices = range(first, min(end, last), step)
        if indices:
            pieces.append(batch.take(pa.array(
                [index - start for index in indices], pa.int64())))
        start = end
    return pa.Table.from_batches(pieces, schema=table.schema)


def _validate_output(spec, output, tables):
    if (isinstance(output.runtime_s, bool)
            or not isinstance(output.runtime_s, (int, float))
            or not math.isfinite(output.runtime_s) or output.runtime_s < 0):
        raise ValueError("runtime_s must be a finite nonnegative number")
    references = corpus_ids(spec, tables)

    def validate_ids(table, aliases):
        # a join can return hundreds of millions of rows: one Arrow pass
        # maps each id to its corpus position, and the checks run on
        # those small integers
        if any(table[alias].null_count for alias in aliases):
            raise ValueError("document IDs cannot be null")
        codes = encode_ids(table, aliases, references)
        for alias in aliases:
            if codes[alias].null_count:
                raise ValueError(f"unknown document ID for alias {alias}")
        if codes.num_rows and (
                codes.group_by(aliases).aggregate([]).num_rows != codes.num_rows):
            raise ValueError("duplicate document IDs in an answer table")

    selected = [name.split(".")[0] for name in spec.info.select
                if name.endswith(".id")]
    columns = output_columns(spec)
    label_names = {
        operator.output: operator for operator in spec.info.classifies}

    def validate_labels(table, names):
        for name in names:
            column = table[name]
            if column.null_count or not pa.types.is_string(column.type):
                raise ValueError("labels must be non-null strings")
            allowed = pa.array(label_names[name].labels, pa.string())
            if not pc.all(pc.is_in(column, value_set=allowed)).as_py():
                raise ValueError(
                    f"a {name} label is not one of the query's labels")

    returned = [name for name in columns if name in label_names]
    structs = {struct.name: struct for struct in spec.info.structs}
    returned_answers = selected_extracts(spec)

    def validate_answers(table):
        for name, struct in structs.items():
            column = table[name]
            fields = [output_name(field) for field in struct.fields]
            if (not pa.types.is_struct(column.type)
                    or [item.name for item in column.type] != fields
                    or not all(pa.types.is_string(item.type)
                               for item in column.type)):
                raise ValueError(
                    f"{name} must be a struct of string answers named {fields}")

    def rows_follow_extracts(rows):
        # a returned answer is the one the engine's answer table holds
        for operator in returned_answers:
            table = (output.extract_answers or {}).get(operator.id)
            if table is None:
                continue
            alias = operator.relation
            struct = spec.info.struct_of(operator).name
            known = pa.table({
                alias: table[alias].cast(pa.string()),
                "known": table["answer"].cast(pa.string())})
            joined = pa.table({
                alias: rows[alias].cast(pa.string()),
                "returned": pc.struct_field(
                    rows[struct], operator.output).cast(pa.string()),
            }).join(known, keys=[alias], join_type="left outer")
            same = pc.or_kleene(
                pc.equal(joined["returned"], joined["known"]),
                pc.and_(pc.is_null(joined["returned"]),
                        pc.is_null(joined["known"])))
            if joined.num_rows and not pc.all(same.fill_null(False)).as_py():
                raise ValueError(
                    f"a returned {struct}.{operator.output} answer differs "
                    "from the engine's answer table")

    answers = scores_from_answers(spec, output, tables)
    if spec.info.relational:
        if output.rows is None:
            raise ValueError("a relational query's run must include its rows")
        if set(output.rows.column_names) != set(columns):
            raise ValueError("output columns must match the query's selected "
                             "fields")
        fetch = spec.info.fetch
        if fetch is not None and output.rows.num_rows > fetch.count:
            raise ValueError(
                f"the engine returned {output.rows.num_rows:,} rows for a "
                f"fetch of {fetch.count:,}")
        validate_ids(output.rows, selected)
        validate_answers(output.rows)
        rows_follow_extracts(output.rows)
    elif output.rows is None:
        # a saved run rescored from its answers: its rows were checked
        # when they were saved
        if answers is None:
            raise ValueError("a run without answers must include its rows")
    elif set(output.rows.column_names) != set(columns):
        raise ValueError("output columns must match the query's selected "
                         "aliases and label columns")
    elif answers is None:
        validate_ids(output.rows, selected)
        if output.rows.num_rows:
            validate_labels(output.rows, returned)
        validate_answers(output.rows)
        rows_follow_extracts(output.rows)
    else:
        # a traced run's rows are implied by its answers: the count must
        # agree, and a sample of the rows must all be implied; the
        # answer tables below get the full checks
        survivors, relations = answers
        implied = implied_row_count(spec, survivors, relations)
        if output.rows.num_rows != implied:
            raise ValueError(
                f"the engine returned {output.rows.num_rows:,} rows but its "
                f"answers imply {implied:,}")
        sample = _sample_rows(output.rows, ROW_SAMPLE)
        validate_ids(sample, selected)
        mask = implied_rows_mask(
            pa.table({name: sample[name].cast(pa.string())
                      for name in columns if name not in structs}),
            survivors, relations, spec,
            answer_labels(spec, output.classify_answers))
        if not (pc.all(mask).as_py() if sample.num_rows else True):
            raise ValueError("a returned row is not implied by the answers")
        validate_answers(sample)
        rows_follow_extracts(sample)
    if output.rows is not None:
        for join in spec.info.joins:
            if not (join.on or join.comparisons) or not set(
                    join.relations).issubset(
                    output.rows.column_names):
                continue
            for batch in output.rows.select(join.relations).to_batches(
                    max_chunksize=ROW_SAMPLE):
                rows = pa.Table.from_batches([batch])
                if _apply_conditions(rows, join, spec, tables).num_rows != len(rows):
                    raise ValueError("returned rows violate the join conditions")
    filters = {
        filter_spec.id: filter_spec for filter_spec in spec.info.filters
    }
    for operator_id, table in (output.filter_answers or {}).items():
        if operator_id not in filters:
            raise ValueError(f"unknown filter operator {operator_id!r}")
        alias = filters[operator_id].relation
        validate_ids(table, [alias])
        if table["answer"].null_count or str(table["answer"].type) != "bool":
            raise ValueError("predicate answers must be non-null booleans")
    joins = {join.id: join for join in spec.info.joins}
    for operator_id, table in (output.join_answers or {}).items():
        if operator_id not in joins:
            raise ValueError(f"unknown join operator {operator_id!r}")
        validate_ids(table, joins[operator_id].relations)
        if table["answer"].null_count or str(table["answer"].type) != "bool":
            raise ValueError("predicate answers must be non-null booleans")
    classifies = {operator.id: operator for operator in spec.info.classifies}
    for operator_id, table in (output.classify_answers or {}).items():
        if operator_id not in classifies:
            raise ValueError(f"unknown classify operator {operator_id!r}")
        operator = classifies[operator_id]
        validate_ids(table, list(operator.relations))
        validate_labels(
            table.rename_columns([
                operator.output if name == "label" else name
                for name in table.column_names]),
            [operator.output])
    scores = {operator.id: operator for operator in spec.info.scores}
    for operator_id, table in (output.score_answers or {}).items():
        if operator_id not in scores:
            raise ValueError(f"unknown score operator {operator_id!r}")
        validate_ids(table, [scores[operator_id].relation])
        if table["score"].null_count or not pa.types.is_floating(
                table["score"].type):
            raise ValueError("scores must be non-null floats")
    extracts = {operator.id: operator for operator in spec.info.extracts}
    for operator_id, table in (output.extract_answers or {}).items():
        if operator_id not in extracts:
            raise ValueError(f"unknown extract operator {operator_id!r}")
        validate_ids(table, [extracts[operator_id].relation])
        if not pa.types.is_string(table["answer"].type):
            raise ValueError("extracted answers must be strings or null")
    info = spec.info
    if (output.rows is not None and len(info.relations) == 1
            and not info.filters and not info.joins and not info.in_lists
            and not info.column_tests and info.aggregate is None
            and info.having is None):
        # Unary classification and scoring preserve input rows. This count
        # is knowable even when an adapter omits its answer traces.
        expected = len(tables[info.relations[0].table])
        if info.fetch is not None:
            expected = min(max(expected - info.fetch.offset, 0), info.fetch.count)
        if output.rows.num_rows != expected:
            raise ValueError(
                f"the engine returned {output.rows.num_rows:,} rows but the "
                f"query requires {expected:,}")


def _score(spec, output, suite, gpu_count, gpu_hourly_rate_usd, tokens=None):
    """Score one query; `tokens` keeps document tokens across a run's queries."""
    _validate_output(spec, output, suite.tables)
    if suite.ground_truth is None:
        _validate_unlabelled_scores(spec, output, suite.tables)
    accuracy = (evaluate(spec, output, suite.ground_truth, suite.tables)
                if suite.ground_truth is not None else None)
    seconds = output.runtime_s
    inputs = {
        relation.alias: len(suite.tables[relation.table])
        for relation in spec.info.relations
    }
    metrics = {
        "runtime_s": seconds, "input_rows": inputs,
        "accuracy": accuracy, "cost_usd": None,
        **token_metrics(spec, output, suite.tables, tokens),
        "evaluated_document_pairs": None,
    }
    if accuracy is None:
        metrics["returned_rows"] = output.rows.num_rows
    metrics["input_tokens_per_second"] = (
        metrics["input_tokens"] / seconds
        if metrics["input_tokens"] is not None and seconds else None)
    if gpu_hourly_rate_usd is not None:
        metrics["cost_usd"] = seconds / 3600 * gpu_count * gpu_hourly_rate_usd
    metrics["cost_usd_per_million_input_tokens"] = (
        metrics["cost_usd"] / metrics["input_tokens"] * 1e6
        if metrics["cost_usd"] is not None and metrics["input_tokens"] else None)
    metrics["kv_regret_percent"] = (
        100 * metrics["regret_tokens"] / metrics["fresh_tokens"]
        if metrics["regret_tokens"] is not None and metrics["fresh_tokens"] else None)
    if spec.info.joins:
        pairs = output.measurements.get("evaluated_document_pairs")
        if output.join_answers is not None:
            if set(output.join_answers) == {
                join.id for join in spec.info.joins
            }:
                pairs = sum(len(table) for table in output.join_answers.values())
        if pairs is not None and (
                isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 0):
            raise ValueError("evaluated_document_pairs must be a nonnegative integer")
        metrics["evaluated_document_pairs"] = pairs
        metrics["document_pairs_per_second"] = (
            pairs / seconds if pairs is not None and seconds else None)
    else:
        metrics["documents_per_second"] = sum(inputs.values()) / seconds if (
            seconds) else None
    return metrics


def _validate_unlabelled_scores(spec, output, tables):
    """Check score domains and observable ranking without reference answers."""
    def validate(column):
        if column.null_count or not pa.types.is_floating(column.type):
            raise ValueError("scores must be non-null floats")
        valid = pc.and_(pc.is_finite(column), pc.and_(
            pc.greater_equal(column, 0), pc.less_equal(column, 1)))
        if len(column) and not pc.all(valid).as_py():
            raise ValueError("scores must be finite and between zero and one")

    for operator in spec.info.scores:
        if operator.output in output.rows.column_names:
            validate(output.rows[operator.output])
        answer = (output.score_answers or {}).get(operator.id)
        if answer is not None:
            validate(answer["score"])
            alias = operator.relation
            if {alias, operator.output}.issubset(output.rows.column_names):
                positions = pc.index_in(output.rows[alias], value_set=answer[alias])
                expected = pc.take(answer["score"], positions)
                expected = expected.cast(output.rows[operator.output].type)
                equal = pc.fill_null(pc.equal(
                    expected, output.rows[operator.output]), False)
                if len(equal) and not pc.all(equal).as_py():
                    raise ValueError("returned scores do not match saved answers")
    sort = spec.info.sort
    if sort is not None:
        keys = [(output_name(name), "descending" if descending else "ascending")
                for name, descending in sort.keys]
        if all(name in output.rows.column_names for name, _ in keys):
            if not output.rows.equals(output.rows.sort_by(keys)):
                raise ValueError("returned rows do not follow the query's sort order")
    info = spec.info
    if (len(info.scores) != 1 or info.filters or info.joins or info.classifies
            or info.column_tests or info.aggregate is not None):
        return
    operator = info.scores[0]
    answer = (output.score_answers or {}).get(operator.id)
    input_count = len(tables[info.relation(operator.relation).table])
    if answer is None or len(answer) != input_count:
        return
    fields = answer.select([operator.relation, "score"]).rename_columns(
        [f"{operator.relation}.id", f"{operator.relation}.{operator.output}"])
    expected = apply_tail(fields.to_pandas(), spec)[list(info.select)]
    expected.columns = [output_name(name) for name in info.select]
    expected_rows = pa.Table.from_pandas(expected, preserve_index=False)
    returned_rows = output.rows.select(expected.columns)
    expected_rows = expected_rows.cast(returned_rows.schema)
    if not expected_rows.equals(returned_rows):
        raise ValueError("returned rows do not match the score answers and query limit")


def run(run_query, *, queries=None, scale_factor=0.1, output_dir,
        metadata=None, gpu_count=1, gpu_hourly_rate_usd=None,
        collection_id=None, cache_dir=None, data_dir=None, root=None,
        accuracy=True, corpus_manifest=None):
    """Run selected queries, save their answers, and generate a report.

    Args:
        run_query: Function(query, tables) returning a RunOutput.
        queries: Query IDs, or None for all published queries.
        scale_factor: Published scale factor: 0.1, 0.5, or 1.0.
        output_dir: New directory for this run.
        metadata: Engine, model, configuration, warmup, and cache settings.
        gpu_count: Number of GPUs executing each query.
        gpu_hourly_rate_usd: Price per GPU hour, or None to omit cost.
        collection_id: Label collection ID, or None to resolve the active one.
        cache_dir: Download cache directory, or None for the default.
        data_dir: Optional directory of input Parquet files to validate.
        root: Optional published-data mirror for offline use.
        accuracy: Whether to load labels and compute reference agreement.
        corpus_manifest: Local TripAdvisor manifest. Its inputs are copied
            into the run directory for portable report regeneration.

    Returns:
        The saved run record. Query failures raise after saving their status.
    """
    from quail_b import __version__
    from quail_b.reporting import _write_report

    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError(f"run directory already exists: {directory}")
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 1:
        raise ValueError("gpu_count must be a positive integer")
    if gpu_hourly_rate_usd is not None and (
            isinstance(gpu_hourly_rate_usd, bool)
            or not math.isfinite(gpu_hourly_rate_usd) or gpu_hourly_rate_usd < 0):
        raise ValueError("GPU price must be finite and nonnegative")
    json.dumps(metadata or {}, allow_nan=False)
    with download_cache(cache_dir):
        suite = load_benchmark(
            queries, scale_factor=scale_factor, collection_id=collection_id,
            data_dir=data_dir, root=root, accuracy=accuracy,
            corpus_manifest=corpus_manifest)
    truth = suite.ground_truth
    record = {
        "schema_version": RUN_SCHEMA_VERSION, "quail_b_version": __version__,
        "scale_factor": suite.scale_factor, "corpus_id": suite.corpus_id,
        "accuracy_enabled": accuracy,
        "collection_id": truth.collection_id if truth is not None else None,
        "reference_model": truth.reference_model if truth is not None else None,
        "metadata": metadata or {}, "gpu_count": gpu_count,
        "gpu_hourly_rate_usd": gpu_hourly_rate_usd,
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "queries": [
            {"id": spec.id, "definition_hash": _query_hash(spec),
             "status": "pending"} for spec in suite.queries],
    }
    directory.mkdir(parents=True, exist_ok=False)
    if corpus_manifest is not None:
        _save_local_inputs(directory, corpus_manifest, suite.corpus_id)
        record["corpus_manifest"] = "inputs/manifest.json"
    path = directory / "run.json"
    _write_json(path, record)
    tokens = {}
    try:
        for spec, item in zip(suite.queries, record["queries"]):
            item["status"] = "running"
            _write_json(path, record)
            print(f"[quail-b] {spec.id}: {spec.description}", flush=True)
            tables = {
                relation.table: suite.tables[relation.table]
                for relation in spec.info.relations
            }
            try:
                output = run_query(spec, tables)
                if not isinstance(output, RunOutput):
                    raise TypeError("run_query must return a RunOutput")
                item.update(
                    files=_save_output(directory / spec.id, output, spec),
                    status="saved",
                )
                json.dumps({"runtime_s": output.runtime_s,
                            "measurements": output.measurements}, allow_nan=False)
                item.update(runtime_s=output.runtime_s,
                            measurements=output.measurements)
                _write_json(path, record)
                item["metrics"] = _score(
                    spec, output, suite, gpu_count, gpu_hourly_rate_usd, tokens)
                item["status"] = "complete"
            except Exception as error:
                item.update(
                    status="scoring_failed" if item["status"] == "saved" else "failed",
                    error=f"{type(error).__name__}: {error}")
                raise
            finally:
                _write_json(path, record)
        record["status"] = "complete"
    except BaseException:
        record["status"] = "failed"
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(path, record)
        _write_report(directory, record)
    return record


def _save_local_inputs(directory, manifest_path, corpus_id):
    """Save a checked corpus snapshot without unrelated source-directory files."""
    manifest, tables = read_tripadvisor_manifest(manifest_path)
    if manifest["corpus_id"] != corpus_id:
        raise ValueError("TripAdvisor corpus changed during run preparation")
    inputs = directory / "inputs"
    inputs.mkdir()
    for name, table in tables.items():
        pq.write_table(table, inputs / f"{name}.parquet", compression="zstd")
    source_path = Path(manifest_path).resolve()
    source = source_path if source_path.is_dir() else source_path.parent
    metadata = json.loads((source / "metadata.json").read_text())
    _write_json(inputs / "metadata.json", metadata)
    _write_json(inputs / "manifest.json", manifest)
    read_tripadvisor_manifest(inputs / "manifest.json")
