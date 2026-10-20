# QUAIL-B

QUAIL-B is a benchmark for AI functions in SQL, or AI-SQL. It is actively being
developed. **The queries use AI-powered filters, joins, and classification;
we will expand to AI-powered extract, map, and groupby.**

For example, query IMDB-4 finds the movie aspects that each review discusses,
for reviews that praise the movie and discuss its ending:

```sql
SELECT r.id AS r, a.id AS a
FROM reviews AS r
JOIN aspects AS a
  ON AI.IF(('Does this review discuss this movie aspect? Review: ', r.body,
            ' Aspect: ', a.aspect))
WHERE AI.IF(('This review mentions a positive aspect of the movie: ', r.body))
  AND AI.IF(('This review discusses the ending of the movie: ', r.body));
```

The query is written with BigQuery's
[`AI.IF`](https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-if)
function, and its prompts are shortened. QUAIL-B publishes each query as a
Substrait plan with the exact prompt text.

The benchmark contains 50 such queries over five document collections: movie
reviews, adverse drug reaction reports, claims and evidence for fact
verification, legal citations, and software agent trajectories. Each collection
comes at three scale factors, with reference answers for every filter, join,
and classification.

To benchmark your engine, you write an adapter: a Python function that receives
one query and its input tables, runs the query on your engine, and returns the
result rows. QUAIL-B

- supplies each query as a [Substrait](https://substrait.io/) plan, with its
  prompts and [PyArrow](https://arrow.apache.org/docs/python/) input tables,
- validates your results and scores them against the reference answers, and
- writes a report of runtime, accuracy, and, if your adapter records them,
  token and KV metrics.

## Contents

- [Getting started](#getting-started)
- [Running the benchmark](#running-the-benchmark)
- [Queries](#queries)
- [Scale factors](#scale-factors)
- [Metrics](#metrics)
- [Results](#results)
- [Development](#development)

The [reference](docs/reference.md) specifies the full adapter contract,
including the optional predicate answers and token data, and the exact rules
for each metric.

## Getting started

QUAIL-B requires Python 3.12:

```sh
uv add "quail-b @ git+https://github.com/fsdatalab/quail-bench.git"
```

Write an adapter and run IMDB-4 at the smallest scale factor:

```python
import pyarrow as pa
import quail_b


def run_query(query: quail_b.QuerySpec, tables: dict[str, pa.Table]):
    # query.plan is the Substrait plan; tables maps table names to data
    rows, runtime_s = my_engine.execute(query.plan, tables)
    return quail_b.RunOutput(
        filter_answers=None,  # optional: the answer for each document
        join_answers=None,    # optional: the answer for each pair
        rows=rows,
        runtime_s=runtime_s,
    )


quail_b.run(
    run_query,
    queries=["IMDB-4"],
    scale_factor=0.1,
    output_dir="results/imdb_4",
    metadata={"engine": "my_engine", "model": "Qwen/Qwen3-4B-FP8"},
)
```

`my_engine.execute` represents a call to your own engine code, which you
write. Typically it translates the Substrait plan into your engine's AI SQL
dialect, such as
[BigQuery AI SQL](https://cloud.google.com/bigquery/docs/generative-ai-overview),
and runs it. It returns the result rows and the query execution time, measured
once all model and GPU work has finished. Engine startup and model loading
stay outside the timer.

`tables` is keyed by the physical table names in the plan, such as `reviews`
and `aspects`. `rows` holds document IDs, with one column for each alias in the
`SELECT` list. IMDB-4 selects `r.id` and `a.id`, so its rows look like:

```python
pa.table({"r": ["rv17", "rv17", "rv42"], "a": ["as0", "as6", "as6"]})
```

When the run finishes, `results/imdb_4/report.md` lists the query's runtime and
the precision and recall of its rows against the reference result.

## Running the benchmark

### TripAdvisor extension

This fork adds four optional TripAdvisor queries through the existing plan
and adapter framework. Their reference labels are pending. The default
50-query workload stays unchanged.

Prepare five hotels with 20 reviews each from the local original archive:

```sh
uv run --frozen quail-b prepare-tripadvisor ../tripadvisor/inputs-allpairs-20/5/corpus \
  --archive-path ../tripadvisor/source/review.txt.zip \
  --hotel-limit 5 --reviews-per-hotel 20 --seed 0 --pair-mode all_unique_pairs
```

The archive comes from the [original hotel-review corpus](
https://www.cs.cmu.edu/~jiweil/html/hotel-review.html). Its pinned SHA256 is
`a43302098170a6c0c086970077dc8a2fce466d431b3c9d9d80ef9565bb057d8d`.
Preparation reads the ZIP locally. It does not download data or run a model.
Use `--max-chars` to truncate text explicitly. Full text is retained by default.

The output has three base Parquet tables, `manifest.json`, and `metadata.json`.
Base tables have string columns `id`, `hotel_id`, and `body`. The builder maps
raw review `id` to `id` and raw `offering_id` to `hotel_id`. Each body contains
only the review title and text. Ratings, authors, and dates stay in metadata.
The original source contains multiple languages; preparation does not filter
for English.

Hotels follow a stable seeded order. Increasing `--hotel-limit` keeps the
smaller sample as a subset. Each selected hotel has a strict calendar-date
boundary: ten sampled reviews precede ten sampled later reviews by default.
The cohorts are disjoint, with no date shared across their boundary. This
produces 100 reviews and 500 earlier/later candidate pairs in the historical
mode. Current `TA-3` uses `--pair-mode all_unique_pairs`: five hotels with twenty
reviews yield 950 unique unordered pairs. Two derived endpoint tables retain
`review_id`, body, hotel, a unique execution `id`, and shared `pair_id`.
Each endpoint first passes the same `TA_3_1` complaint filter. The AI join
uses `pair_id` equality and the unchanged `TA_3` prompt. No self-pairs,
reverse duplicates, or cross-hotel pairs are evaluated. Loading reconstructs
and validates endpoints.

The manifest records source hashes, settings, table hashes, provenance,
excluded records, and truncation counts. Loading checks IDs, source text,
cohort membership, dates, and content hashes. A supplied local archive can use
an explicit `--source-sha256` pin. The default five-hotel sample is a setup
check, not a representative estimate of hotel performance.

| Query | Task | Existing plan pattern |
| --- | --- | --- |
| `TA-1` | Concrete hotel problem, then unsuccessful staff help | `IMDB-6`: two filters |
| `TA-2` | Route each review to one of six handling categories | `IMDB-11`: unary classification |
| `TA-3` | Filter complaints on both sides, then match the same concrete problem | `IMDB-2`: AI join; `FEV-10`: equality restriction |
| `REL-TA-1` | Rank explicit forced moves; stable top 10 | `REL-AGENT-4`: score, sort, fetch |

The source builder lives alongside the other builders in `quail_b/data.py`.
The existing `tools/make_substrait_plans.py` generates these plans with the
same operator classes and IDs. Loading returns the existing `Benchmark`;
execution, output validation, and reports use `quail_b.run` and `quail_b.report`.
Local manifest support supplies unpublished inputs and preserves their
identity. There is no separate TripAdvisor parser, execution engine, or scorer.

`TA-2` uses a fixed policy: `housekeeping`, `maintenance`, `front_desk`,
`billing`, `manual_review`, and `no_action`. Multiple departments or unclear
responsibility go to `manual_review`. `REL-TA-1` asks whether a hotel problem
actually forced a room change, early departure, or relocation. Its score is
the model's Boolean answer probability, not calibrated severity or risk.
All judgments use reported review text; they do not verify events or show
whether hotel management later fixed a recurring problem.

`TA-3` requires a concrete hotel failure or complaint in both reviews. Reviews
with no problem, only praise, or different problems must return FALSE. Shared
amenities, positive experiences, broad topics, and general sentiment do not
establish a match. This prompt was tightened on 2026-10-06. Runs with the earlier
prompt retain a different query definition and definition hash. Keep their
original plans and outputs; do not compare them as runs of the revised prompt.
The unique-pair revision also changes the plan and corpus identity. Preserve
historical earlier/later results. Native token minimum and regret describe
execution-ID grouping in the prepared endpoint encoding. Original-review KV
analysis must collapse copies by `review_id` and remain a separate report.
The filtered revision adds `filter-1` on `a` and `filter-2` on `b` before
`join-1`. For one hotel with 100 reviews, the 4,950 endpoint rows on each
side require 9,900 unary answers. This plan does not filter each original
review only once. The added gates change the query definition hash; preserve
the prior unfiltered plans and outputs as a separate workload.

Supply an engine through the existing callback contract:

```python
quail_b.run(
    run_query,
    queries=["TA-1", "TA-2", "TA-3", "REL-TA-1"],
    corpus_manifest="../tripadvisor/inputs-allpairs-20/5/corpus/manifest.json",
    accuracy=False,
    output_dir="../tripadvisor/outputs/example",
)
```

The local manifest replaces published scale-factor selection. Output
validation and available token accounting remain active. Reference agreement
is unavailable until independent labels exist. The run saves a validated
input snapshot under `inputs/`, so reports can be regenerated after moving
the run directory or removing the original prepared inputs:

```sh
uv run --frozen quail-b report ../tripadvisor/outputs/example
```

CPU tests use small source fixtures and fabricated adapter outputs. They
check preparation, plans, validation, and reports. They do not measure model
accuracy or GPU performance. Engine-specific experiment drivers stay in the
adjacent `tripadvisor/` directory.

A full call to `quail_b.run` looks like:

```python
quail_b.run(
    run_query,
    queries=None,                        # None runs all 50 queries
    scale_factor=0.1,                    # 0.1, 0.5, or 1.0
    output_dir="results/vllm_qwen3_4b",  # must be a new directory
    metadata={"engine": "vllm", "model": "Qwen/Qwen3-4B-FP8"},
    gpu_count=1,
    gpu_hourly_rate_usd=3.9492,
)
```

QUAIL-B calls `run_query` once per query, in the order given. After each query
it saves the output, scores it, and updates `run.json`. At the end it writes
`report.md` and `measurements.parquet`, and returns the run record.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `run_query` | required | Your adapter |
| `queries` | `None` | Query IDs to run; `None` runs all 50 |
| `scale_factor` | `0.1` | Published scale factor: `0.1`, `0.5`, or `1.0` |
| `output_dir` | required | New directory for this run's results |
| `metadata` | `None` | JSON object saved with the run: engine, model, settings |
| `gpu_count` | `1` | GPUs used by each query, for cost |
| `gpu_hourly_rate_usd` | `None` | Price per GPU hour; `None` leaves cost unreported |
| `collection_id` | `None` | Reference collection; `None` uses the published one |
| `cache_dir` | `None` | Download cache; `None` uses `~/.cache/quail-b` |
| `data_dir` | `None` | Local input Parquet files, used in place of the download |
| `root` | `None` | Local mirror of the published data, for offline runs |

Record everything that affects performance in `metadata`: engine version,
model, batch sizes, cache settings, and warmup. The report shows only QUAIL-B's
measurements, so `metadata` is how you tell two runs apart later.

### Data and caching

The first run downloads the input tables and reference answers for the selected
queries from the public `s3://quail-bench` bucket. Later runs read them from
`~/.cache/quail-b`. Set `QUAIL_B_CACHE_DIR` or pass `cache_dir` to use another
location. QUAIL-B checks every loaded table against the published corpus
identity, so local files that differ from the published data fail the run.

Each reference answer takes about 25 bytes in memory. Loading every answer at
scale factor 0.1, about 1.2 million answers, takes about 2 seconds from the cache
and peaks at 0.62 GiB, input tables included. At scale factor 1.0, about 52 million
answers, budget about 3 GiB.

### Inspecting queries and tables

You can load any query or table directly while writing an adapter:

```python
query = quail_b.get_query("IMDB-4")
print(query.description)  # F1 -> F4 -> J1, 2 filters then 1 join
plan = query.plan         # a substrait.plan_pb2.Plan

reviews = quail_b.load_table("reviews", scale_factor=0.1)
print(reviews.num_rows)   # 5000
```

## Queries

The queries use three AI functions, declared as Substrait extensions in
[`quail_b/substrait_extensions.yaml`](quail_b/substrait_extensions.yaml):

```text
ai_filter(prompt, document) -> boolean
ai_join(prompt, left_document, right_document) -> boolean
ai_classify(prompt, document, labels, descriptions) -> string
ai_classify(prompt, anchor, partner, labels, descriptions) -> string
```

The plans combine them with scans, equality conditions, conjunction,
projection, and IN-list filters on label columns, such as
`r.sentiment IN ('negative', 'mixed')`. The first form of `ai_classify` labels
one document. The second labels the rows of a join: each row is the two
documents, one from each table, that the join kept, such as a review and one
aspect it discusses. The anchor is the document placed first in the prompt,
and the partner follows it. [Classification queries](#classification-queries)
lists the queries that use it.

| Dataset | Queries | Tables | Task |
| --- | --- | --- | --- |
| IMDB | IMDB-1 to IMDB-15 | `reviews`, `aspects` | Review aspects and sentiment |
| BioDEX | BIO-1 to BIO-6 | `reports`, `terms` | Adverse drug reactions |
| FEVER | FEV-1 to FEV-11 | `claims`, `evidence` | Fact verification |
| LePaRD | LEP-1 to LEP-6 | `citation_contexts`, `citation_passages` | Legal citations |
| SWE-Next | AGENT-1 to AGENT-5 | `agent_traces` | Software agent trajectories |
| SWE-Next | REL-AGENT-1 to REL-AGENT-7 | `agent_traces` | Relational operators over the trajectories |

Within each dataset, the first queries have a single filter or join. Later
queries chain filters, filter both join inputs, scan one table under two
aliases, and chain three joins. The plans are stored as Substrait
ProtoJSON in [`quail_b/plans/`](quail_b/plans/), with their order and
descriptions in [`catalog.json`](quail_b/plans/catalog.json).

### Example: IMDB-4

IMDB-4, the query at the top of this page, has this operator tree:

```text
Project [r.id, a.id]
└── AI Join J1                   join-1
    ├── AI Selection F4          filter-2
    │   └── AI Selection F1      filter-1
    │       └── Scan reviews AS r
    └── Scan aspects AS a
```

`F1`, `F4`, and `J1` name the three prompts in the SQL above: positive aspect,
ending, and review discusses aspect. The plan stores them as string literals.
`filter-1`, `filter-2`, and `join-1` are operator IDs. The prompt text
comes from [`quail_b/prompts.py`](quail_b/prompts.py) and
[`quail_b/rendering.py`](quail_b/rendering.py). Every prompt starts with a
document, followed by a question that begins "Evaluate TRUE or FALSE for the
following question:", so an engine can reuse a document's KV across questions.

### Developing an adapter

The 50 queries have several different shapes: how many filters, joins,
classifications, and relational operators they have, and how those operators
are arranged in the plan. The table below lists one query for each distinct
shape, from simplest to most complex. Test your adapter on these queries first,
then run it on all 50.

| Query | Shape | What it tests |
| --- | --- | --- |
| IMDB-1 | One filter | Scans, prompt rendering, and result IDs |
| IMDB-2 | One join | Pair evaluation and two output columns |
| IMDB-4 | Two filters, then one join | Operator order |
| FEV-5 | Filters on both join inputs | Filters on each side of a join |
| IMDB-8 | Two joins sharing one input | Two aliases of one table |
| FEV-8 | Chain of three joins | Multiple joins |
| FEV-10 | Filtered join with equality | Equality and AI conditions together |
| BIO-4 | Three filters, two joins | Filters on two aliases of one table |
| FEV-11 | Classification returned and filtered | A label column, and an IN-list filter on it |
| AGENT-4 | Two classifications in a chain | Label descriptions, and a label filter feeding a classification |
| IMDB-15 | Classification of joined rows | A label of two documents, returned per pair |
| REL-AGENT-1 | Two column tests, then a filter | Column tests before the model |
| REL-AGENT-2 | Filter, sort, offset, limit | Ordered and paged rows |
| REL-AGENT-4 | Score, sort, limit | `ai_score` and top-k |
| REL-AGENT-5 | Classify, group by a label, having | `COUNT`, `COUNT(DISTINCT)`, `HAVING`, label keys |
| REL-AGENT-6 | Filter, group by a column, having, limit | `MIN`, `MAX`, and a limit after an aggregate |

### Classification queries

Twelve queries return or filter on a label chosen from a fixed list. Each asks
a question an analyst would ask of that collection, with a standard label list
where one exists, such as IMDb's genres or MedDRA's system organ classes.
Label length follows from the list, from one token for sentiment to eleven
for an organ class under the Qwen3 tokenizer. IMDB-14, LEP-6, and AGENT-4
have labels that start with the same words, as category names often do.

IMDB-15 is the one query that classifies joined rows. It joins each negative
or mixed review with the movie aspects the review discusses. It then labels
each (review, aspect) row with the review's sentiment toward that aspect. For
example, the row of a review and the aspect "the acting" gets one of
positive, negative, neutral, or mixed.

| Query | Question | Labels |
| --- | --- | --- |
| IMDB-11 | What is the sentiment of each review? | 4, one token each |
| IMDB-12 | Among reviews that praise something (F1), what genre is each movie? | IMDb's 21 genres |
| IMDB-13 | Which aspects do negative or mixed reviews discuss? | Sentiment, as a filter before J1, returned per pair |
| IMDB-14 | What do negative or mixed reviews complain about most? | Sentiment, then 8 complaints; three start with "poor" and two with "too"; both returned |
| IMDB-15 | What sentiment does each negative or mixed review express toward each aspect it discusses? | Sentiment, as a filter before J1, then 4 sentiments per pair; both returned |
| BIO-5 | Which system organ class does each reaction term belong to? | MedDRA's 26 classes, 1 to 11 tokens |
| BIO-6 | Which cardiac or vascular reactions do serious reports describe? | Organ class, as a filter on one join input |
| FEV-11 | Which claims are about politics or history, and which topic? | 11 topics; one call returned and filtered |
| LEP-6 | Which passages do constitutional or criminal law excerpts cite? | 15 areas of law, as a filter before the join |
| AGENT-3 | How far have agents that recovered from a failed approach gotten? | 5 progress stages with descriptions, after a filter |
| AGENT-4 | For agents that changed the code, what did the latest test or reproduction run show? | Progress, then 4 test results with descriptions |
| AGENT-5 | For every trace: how far has the agent gotten, what is the project, and what kind of defect is the bug? | Progress, 27 PyPI topics, and 8 ODC defect types; three questions of one trace |

A classification answer is the label with the largest sum of label-token log
probabilities; the [reference](docs/reference.md#classification) defines the
prompt, the answer tables, and label accuracy.

### Relational queries

The seven `REL-AGENT` queries add relational operators over the agent traces:
a column test (a `WHERE` condition on a stored column, such as
`turn_index >= 10`, applied before any model call), `ORDER BY` with `OFFSET`
and `LIMIT`, `DISTINCT`, `GROUP BY` with `COUNT`, `COUNT(DISTINCT)`, `SUM`,
`AVG`, `MIN`, `MAX`, and `HAVING`. Two of them use a fourth AI function,
`ai_score(prompt, document) -> fp64`. It returns the model's belief, from 0
to 1, that the document answers a filter prompt TRUE.

| Query | Question | Operators |
| --- | --- | --- |
| REL-AGENT-1 | Recovered snapshots at turn 10 or later, of at most 6,000 tokens | Two column tests, `ai_filter` |
| REL-AGENT-2 | Second page of ten recovered snapshots, shortest first | `ai_filter`, `ORDER BY`, `OFFSET`, `LIMIT` |
| REL-AGENT-3 | Trajectories with a plausible fix | `ai_filter`, `DISTINCT` |
| REL-AGENT-4 | The 20 snapshots with the highest recovery score | `ai_score`, `ORDER BY`, `LIMIT` |
| REL-AGENT-5 | Snapshots and trajectories per test outcome, at least 50 snapshots | Two `ai_classify`, `GROUP BY`, `COUNT`, `COUNT(DISTINCT)`, `HAVING`, `ORDER BY` |
| REL-AGENT-6 | Trajectories with at least two fixes, earliest first | `ai_filter`, `GROUP BY`, `MIN`, `MAX`, `HAVING`, `ORDER BY`, `LIMIT` |
| REL-AGENT-7 | Ten trajectories of at least five snapshots with the highest mean fix score | `ai_score`, `GROUP BY`, `AVG`, `HAVING`, `ORDER BY`, `LIMIT` |

## Scale factors

Scale factors 0.1, 0.5, and 1.0 sample 10%, 50%, and 100% of each dataset's
target size. Each corpus is sampled with a fixed seed from pinned upstream
revisions, listed in [`quail_b/data.py`](quail_b/data.py). The queries are the
same at every scale factor. Use 0.1 while developing an adapter.

| Dataset | Table | 0.1 | 0.5 | 1.0 |
| --- | --- | ---: | ---: | ---: |
| IMDB | `reviews` | 5,000 | 25,000 | 50,000 |
| IMDB | `aspects` | 12 | 12 | 12 |
| BioDEX | `reports` | 500 | 2,500 | 5,000 |
| BioDEX | `terms` | 1,127 | 2,934 | 4,144 |
| FEVER | `claims` | 500 | 2,500 | 5,000 |
| FEVER | `evidence` | 287 | 1,037 | 1,478 |
| LePaRD | `citation_contexts` | 500 | 2,496 | 4,972 |
| LePaRD | `citation_passages` | 433 | 1,756 | 2,991 |
| SWE-Next | `agent_traces` | 1,772 | 8,859 | 17,711 |

### Reference answers

QUAIL-B scores every run against reference answers: one TRUE or FALSE label
for each document or document pair each filter or join can be asked about,
and one label for each document a classification can be asked about. A
classification of joined rows has one label for each row the reference join
keeps.

**Accuracy is not a focus of this benchmark.** Most labels are the answers of
one arbitrary model, `Qwen/Qwen3-32B-FP8`, so it is not really meaningful to
measure accuracy against them. We provide these fake labels anyway.

Filter and join labels are Qwen3 32B's answers through Quail. Classification
labels are Qwen3 32B's highest-scoring label through stock vLLM 0.26.0, as
`quail_b.predicates.CLASSIFY_JUDGE_SPEC` records.

Two datasets have real labels for join operations. First, the join that asks
whether a FEVER passage supports a claim uses the claim annotations from
[FEVER](https://huggingface.co/datasets/fever/fever) where they exist. Second,
the LePaRD citation join uses the citation links from
[LePaRD](https://huggingface.co/datasets/rmahari/LePaRD).

The input tables and labels live in the public S3 bucket `s3://quail-bench`,
under `ground_truth/quailb/schema_v1/`:

| Path | Contents |
| --- | --- |
| `corpora/<corpus_id>/` | Input tables of one scale factor, as Parquet |
| `label_sets/<dataset>/<predicate>/<label_set_id>/` | Labels of one predicate |
| `collections/<collection_id>/` | Which label set each predicate uses |

The IDs are content hashes. A corpus ID names the exact input tables, and a
collection ID names one complete set of labels for that corpus, so any change
to the data or labels produces new IDs. These are the published IDs:

| Scale factor | Corpus ID | Collection ID |
| --- | --- | --- |
| 0.1 | `c_1aa2c4f0d0b6c816fd37aa5748c33341` | `gt_72abc9af3feaea668e493ece67e980a0` |
| 0.5 | `c_6773c85b3754908434661c1dadfad0fa` | `gt_397dae1857e5850c84b61dccabb3431c` |
| 1.0 | `c_81a95887a650aaa1a343e0d688b81bef` | `gt_a72baf70eae95e6b1681a11091eaafca` |

`quail_b.run` loads the matching collection for you and records both IDs in
`run.json`. Compare results only across runs with the same IDs.

To look at the labels or score answers yourself, load them with the query's
input tables:

```python
from quail_b.scoring import agreement, expected_rows

benchmark = quail_b.load_benchmark(["IMDB-4"], scale_factor=0.1)
labels = benchmark.ground_truth           # the collection for these queries
query = benchmark.queries[0]

expected = expected_rows(query, labels, benchmark.tables)  # reference result
for key, predicate in labels.predicates.items():
    print(key, predicate.table.num_rows)  # left_id, right_id, answer

# compare your answers for F4 (filter-2) with its labels
ending = labels.predicates["quailb.imdb.review.discusses_ending"]
counts = agreement(your_filter_table, ["r"], ending)
print(counts.correct, counts.evaluated)
```

## Metrics

Every run reports query time and output quality. The other metrics appear when
the adapter returns the data they need; the
[reference](docs/reference.md#metric-requirements) lists what each one
requires. A metric that lacks its data shows as `unavailable`; QUAIL-B never
reports it as zero.

| Metric | Definition |
| --- | --- |
| Query time | `runtime_s`, in seconds |
| Output precision and recall | Returned rows compared with the reference result |
| Predicate-level accuracy | Share of filter and join answers that match the labels |
| Label accuracy | Share of classification answers that match the labels |
| Document throughput | Input documents per second, for queries with zero joins |
| Join throughput | Evaluated document pairs per second |
| GPU cost | `runtime_s / 3600 * gpu_count * gpu_hourly_rate_usd` |
| Input tokens | Full length of every evaluated prompt, including KV hits |
| Fresh tokens | Positions processed by model forward passes |
| Minimum tokens | Positions needed with an unlimited prefix KV cache |
| KV regret | Fresh tokens above the minimum, as a percentage of fresh tokens |
| Input token throughput | Input tokens per second |
| Cost per million input tokens | GPU cost divided by input tokens, times one million |

## Results

Each run writes to its `output_dir`:

```text
results/vllm_qwen3_4b/
├── report.md             # summary table of every metric
├── run.json              # configuration, data IDs, status, and metrics
├── measurements.parquet  # one row of metrics per query
└── IMDB-4/
    ├── plan.substrait    # the exact plan that ran
    ├── rows.parquet      # the result rows
    ├── filters-0.parquet # optional filter answers
    ├── joins-0.parquet   # optional join answers
    ├── classifications-0.parquet  # optional classification answers
    ├── scores-0.parquet  # optional score answers
    └── prompt_pieces.json
```

The run stops at the first adapter or scoring error and records it in
`run.json`. Outputs are saved before scoring, so you can rescore a run from its
saved files:

```sh
quail-b report results/vllm_qwen3_4b
```

## Development

To work on QUAIL-B itself:

```sh
git clone https://github.com/fsdatalab/quail-bench.git
cd quail-bench
uv sync
uv run pytest -q
```
