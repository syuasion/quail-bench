"""Load the QUAIL-B query catalog from Substrait plans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache, cached_property
from importlib.resources import files

from google.protobuf import json_format
from google.protobuf.message import DecodeError
from substrait import plan_pb2

from quail_b.prompts import (
    AGENT_CHANGED_CODE,
    AGENT_IMPLEMENTED_FIX,
    AGENT_PROGRESS,
    AGENT_RECOVERED,
    ASPECT_SENTIMENT,
    BIO_ORGAN_CLASS,
    CARDIOVASCULAR_REACTION,
    CUAD_HAS_CLAUSE,
    DISCUSS_ASPECT,
    F1,
    F4,
    F5,
    F11,
    F12,
    F13,
    FEV_TOPIC,
    IMDB_SENTIMENT,
    LEP1,
    LEP2,
    LEP3,
    LEP4,
    LEP5,
    LEP_AREA,
    LEPJOIN,
    LEPS1,
    NEUROLOGICAL_REACTION,
    REACTION,
    REFUTE,
    RUNS_DIFFERENT_APPROACH,
    RUNS_REPRODUCED,
    RUNS_TEST_STEP,
    SALES_CHANGED_OFFER,
    SALES_COMMITS,
    SALES_COMPETITOR,
    SALES_DISCOUNT,
    SERIOUS_ADVERSE_EVENT,
    SUPPORT,
    SUPPORT_DIFFERENT_APPROACH,
    SUPPORT_FRUSTRATED,
    SUPPORT_PUSHBACK,
    WRENCH_EXPLOITED,
    WRENCH_STEP_EXPLOIT,
    cuad_filter,
)
from quail_b.substrait import PlanInfo, inspect_plan

# Fixed planner inputs from the sf=0.1 Qwen3 32B fp8 labels.
# They apply at every scale factor so query planning does not read answers.
SELECTIVITY_ESTIMATE_COLLECTION = "gt_77bb8b128743a79aedddaa24c808c3f8"
SELECTIVITY_ESTIMATE_CORPUS = "c_1aa2c4f0d0b6c816fd37aa5748c33341"
SELECTIVITY_ESTIMATE_SCALE_FACTOR = 0.1
FILTER_SELECTIVITY_ESTIMATES = {
    F1: 4004 / 5000,
    F4: 1218 / 5000,
    F5: 2853 / 5000,
    # Same reports, saved label set ls_5627a6d5416349ee3e418c41594bc3a3.
    SERIOUS_ADVERSE_EVENT: 319 / 500,
    # BIO-4 labels: gt_cd3ebdb784f64b9e028e50ea73cdedd0.
    NEUROLOGICAL_REACTION: 505 / 1127,
    CARDIOVASCULAR_REACTION: 394 / 1127,
    AGENT_RECOVERED: 570 / 1772,
    AGENT_IMPLEMENTED_FIX: 537 / 1772,
    F11: 296 / 500,
    F12: 69 / 500,
    F13: 159 / 287,
    LEP1: 14 / 500,
    LEP2: 229 / 500,
    LEP3: 51 / 500,
    LEP4: 31 / 500,
    LEP5: 14 / 500,
    LEPS1: 351 / 433,
    # CUAD: the lawyers' Yes answers over the 449 contracts within the
    # token cap (quail_b.data.CUAD_CONTRACTS), the same at every scale
    cuad_filter("Termination For Convenience"): 147 / 449,
    cuad_filter("Non-Compete"): 90 / 449,
    cuad_filter("Exclusivity"): 144 / 449,
    cuad_filter("Audit Rights"): 164 / 449,
    cuad_filter("Cap On Liability"): 220 / 449,
    cuad_filter("Insurance"): 118 / 449,
    cuad_filter("License Grant"): 209 / 449,
    cuad_filter("Non-Transferable License"): 108 / 449,
    cuad_filter("Change Of Control"): 88 / 449,
    # agent traces and sales calls: gt_92964efd9d1b9c8e0cc64ebd30d73e0b
    SUPPORT_FRUSTRATED: 76 / 3442,
    RUNS_REPRODUCED: 312 / 320,
    RUNS_TEST_STEP: 10708 / 36341,
    WRENCH_EXPLOITED: 324 / 629,
    WRENCH_STEP_EXPLOIT: 1339 / 3454,
    SALES_COMPETITOR: 344 / 985,
    SALES_DISCOUNT: 104 / 985,
    SALES_COMMITS: 33 / 985,
}
JOIN_SELECTIVITY_ESTIMATES = {
    DISCUSS_ASPECT: 17683 / 60000,
    ASPECT_SENTIMENT: 9635 / 60000,
    REACTION: 19144 / 563500,
    SUPPORT: 311 / 143500,
    REFUTE: 477 / 143500,
    LEPJOIN: 500 / 216500,
    CUAD_HAS_CLAUSE: 2857 / 14817,
    # gt_92964efd9d1b9c8e0cc64ebd30d73e0b, over the candidate pairs
    SUPPORT_PUSHBACK: 161 / 3058,
    SUPPORT_DIFFERENT_APPROACH: 358 / 1024,
    RUNS_DIFFERENT_APPROACH: 8 / 117,
    SALES_CHANGED_OFFER: 56 / 639,
}
# IN-list filters, keyed by (classification prompt, accepted labels), from
# the sf=0.1 reference labels in gt_6d7ca88a74a30b665bfb67dcde76daff.
IN_LIST_SELECTIVITY_ESTIMATES = {
    (IMDB_SENTIMENT, frozenset({"negative", "mixed"})): 2923 / 5000,
    (BIO_ORGAN_CLASS, frozenset({"cardiac disorders", "vascular disorders"})):
        109 / 1127,
    (FEV_TOPIC, frozenset({"politics", "history"})): 46 / 500,
    (LEP_AREA, frozenset({"constitutional law", "criminal law"})): 159 / 500,
    # from gt_72abc9af3feaea668e493ece67e980a0
    (AGENT_PROGRESS, frozenset(AGENT_CHANGED_CODE)): 639 / 1772,
}


@dataclass(frozen=True)
class QuerySpec:
    """One benchmark query and its serialized Substrait plan."""

    id: str
    description: str
    plan_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.plan_bytes, bytes):
            raise TypeError("plan_bytes must be bytes")
        plan = plan_pb2.Plan()
        try:
            plan.ParseFromString(self.plan_bytes)
        except DecodeError as error:
            raise ValueError(f"{self.id}: invalid Substrait plan bytes") from error
        inspect_plan(plan)
        object.__setattr__(
            self,
            "plan_bytes",
            plan.SerializeToString(deterministic=True),
        )

    @classmethod
    def from_plan(
        cls,
        query_id: str,
        description: str,
        plan: plan_pb2.Plan,
    ) -> "QuerySpec":
        """Create a query from a Substrait plan."""
        return cls(
            query_id,
            description,
            plan.SerializeToString(deterministic=True),
        )

    @property
    def plan(self) -> plan_pb2.Plan:
        """Return a parsed copy of the Substrait plan."""
        return plan_pb2.Plan.FromString(self.plan_bytes)

    @cached_property
    def info(self) -> PlanInfo:
        """Return the benchmark reading of the plan: its relations and steps."""
        return inspect_plan(self.plan)


@cache
def _load_queries() -> tuple[tuple[QuerySpec, ...], ...]:
    """Return (regular, privacy, draft) query definitions."""
    root = files("quail_b").joinpath("plans")
    catalog = json.loads(root.joinpath("catalog.json").read_text())
    regular = []
    privacy = []
    draft = []
    seen = set()
    for item in catalog:
        query_id = item["id"]
        if query_id in seen:
            raise ValueError(f"duplicate query ID {query_id!r}")
        seen.add(query_id)
        plan = json_format.Parse(
            root.joinpath(f"{query_id}.json").read_text(),
            plan_pb2.Plan(),
        )
        query = QuerySpec.from_plan(query_id, item["description"], plan)
        if item.get("draft", False):
            draft.append(query)
        elif item["privacy"]:
            privacy.append(query)
        else:
            regular.append(query)
    return tuple(regular), tuple(privacy), tuple(draft)


def __getattr__(name: str):
    # QUERIES, PRIVACY_QUERIES, and QUERY_ORDER are read from the plan
    # files on first use, so importing the package does not parse them.
    if name not in ("QUERIES", "PRIVACY_QUERIES", "QUERY_ORDER"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    regular, privacy, _draft = _load_queries()
    globals().update(
        QUERIES=regular,
        PRIVACY_QUERIES=privacy,
        QUERY_ORDER=tuple(spec.id for spec in regular),
    )
    return globals()[name]


# the relational queries read the agent traces and their labels, as
# their own family so a runner keeps their results apart
QUERY_FAMILY_WORKLOADS = {
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


def queries(include_privacy: bool = False, *,
            include_draft: bool = False) -> dict[str, QuerySpec]:
    """Return the benchmark queries by id, in benchmark order.

    Args:
        include_privacy: Also return the privacy policy queries.
        include_draft: Also return queries without published reference labels.
    """
    regular, privacy, draft = _load_queries()
    specs = (regular + (privacy if include_privacy else ())
             + (draft if include_draft else ()))
    return {spec.id: spec for spec in specs}


def split_query_ids(ids, containers):
    """Split query IDs into the same equal chunks as stock vLLM."""
    count, extra = divmod(len(ids), containers)
    chunks = []
    start = 0
    for index in range(containers):
        size = count + (1 if index < extra else 0)
        chunks.append(tuple(ids[start:start + size]))
        start += size
    return tuple(chunk for chunk in chunks if chunk)


def split_query_families(ids):
    """Return one ordered query group for each QUAIL-B family."""
    groups = tuple(
        tuple(
            query_id
            for query_id in ids
            if query_id.split("-", 1)[0] == prefix
        )
        for prefix in QUERY_FAMILY_WORKLOADS
    )
    assigned = {query_id for group in groups for query_id in group}
    unknown = [query_id for query_id in ids if query_id not in assigned]
    if unknown:
        raise ValueError(f"unknown query family for {unknown}")
    return tuple(group for group in groups if group)


def query_family_name(ids):
    """Return the name for one query family."""
    prefixes = {query_id.split("-", 1)[0] for query_id in ids}
    if len(prefixes) != 1:
        raise ValueError(
            f"expected one query family, found {sorted(prefixes)}"
        )
    prefix = prefixes.pop()
    try:
        return QUERY_FAMILY_WORKLOADS[prefix]
    except KeyError as error:
        raise ValueError(f"unknown query family {prefix!r}") from error


def get_query(query_id: str) -> QuerySpec:
    """Return one benchmark query definition by id."""
    return queries(include_draft=True)[query_id]
