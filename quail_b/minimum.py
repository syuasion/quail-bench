"""The fewest input tokens a run's requests need with unlimited KV.

Every request an engine made is a token sequence: a document prefix
followed by a filter or classification question, or an anchor prefix
and its frame followed by a partner label, the partner document, and
the answer cue (a join, or a classification of joined rows).
With unlimited KV every distinct prefix across the document
sequences is computed once: each document's text once, and the
questions and frames after one document once each, sharing the lead
they have in common (an engine that rewinds KV to where two questions
diverge computes that lead once). The partner label after a frame is
the same for every pair of one anchor, so it is computed once per
anchor. Each pair's partner document and answer cue follow it; the
pairs of one anchor share the tokens their partner documents begin
with, as a prefix cache over whole requests does. What an engine computed
beyond the minimum is its regret, whatever the cause: an evicted
anchor computed again, a set scanned twice under two aliases, or a
prompt prefix the documents share computed once per document.

Rows with the same text hold equal prefixes, so a prefix cache over
whole requests computes their questions once. The minimum counts them
as one document, whatever their ids or tables.

The minimum does not depend on how an engine reads a label. A
classification's tail is the reference prompt's: the question, the
labels by name, and the answer cue. Reading the label takes no position
after the cue. An engine that lists the labels under letters, or feeds
label tokens after the cue to score them, computes more than the
minimum, and the difference is regret.

The engine reports the prompt pieces it used as token ids (see
`validate_prompt_pieces`); the documents are tokenized here with the
tokenizer the pieces name, after the run, so nothing is tracked while
the query runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from quail_b.data import _ids


@dataclass
class _Document:
    alias: str
    row_id: str
    suffixes: set = field(default_factory=set)
    groups: dict = field(default_factory=dict)


def _tokens(sequence) -> np.ndarray:
    return np.asarray(sequence, dtype=np.uint32)


def prefix_trie_size(sequences) -> int:
    """Return the distinct prefix positions across token sequences.

    Sorted, each sequence sits next to the one it shares the longest
    prefix with, so the trie holds the total length minus the shared
    prefix of every neighbouring pair. Sequences compare as big endian
    bytes, whose order is the token order.
    """
    keys = sorted(_tokens(sequence).astype(">u4").tobytes()
                  for sequence in sequences)
    total = sum(len(key) for key in keys) // 4
    shared = 0
    for earlier, later in zip(keys, keys[1:]):
        length = min(len(earlier), len(later))
        differs = (np.frombuffer(earlier, np.uint8, length)
                   != np.frombuffer(later, np.uint8, length))
        first = int(differs.argmax()) if differs.any() else length
        shared += first // 4
    return total - shared


def _token_list(value, name):
    if not isinstance(value, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value):
        raise ValueError(f"prompt pieces: {name} must be a list of token ids")
    return [int(item) for item in value]


def validate_prompt_pieces(spec, pieces) -> dict:
    """Check and normalize the prompt pieces an engine reports.

    Args:
        spec: The query.
        pieces: A dict with `tokenizer` (a HuggingFace tokenizer name,
            the one that tokenized the documents), `preamble` (token
            ids before every document), `filters` (a list of
            `{"id", "tail"}`: the operator ID and ids after the
            document of that filter) and `joins` (a list of
            `{"id", "anchor", "frame", "label", "tail"}`: the
            anchor alias, the ids after the anchor document, the ids
            before the partner document, and the ids after it).
            A query with classifications also needs `classifies`: a
            list of `{"id", "tail"}` for a classification of one
            document, and of `{"id", "anchor", "frame", "label",
            "tail"}`, read as for a join, for a classification of
            joined rows. A classification's pieces are those of the
            reference prompt (`render_classify_prompt`), whatever prompt
            the engine sent; its tail ends with the answer cue. A query
            with scores also needs `scores`: a list of `{"id", "tail"}`,
            read as for a filter, and one with extractions `extracts`,
            a list of `{"id", "tail"}` read the same way: the question,
            the instruction, and the answer template through the cue
            that opens the answer.

    Returns:
        The pieces as plain lists, with every stage of the query named.
    """
    if not isinstance(pieces, dict) or not isinstance(
            pieces.get("tokenizer"), str) or not pieces["tokenizer"]:
        raise ValueError("prompt pieces need a tokenizer name")
    checked = {"tokenizer": pieces["tokenizer"],
               "preamble": _token_list(pieces.get("preamble", ()), "preamble"),
               "filters": [], "joins": [], "classifies": [], "scores": [],
               "extracts": []}
    stages = {filter_spec.id for filter_spec in spec.info.filters}
    for item in pieces.get("filters", ()):
        operator_id = item.get("id")
        if operator_id not in stages:
            raise ValueError(
                f"prompt pieces: unknown filter operator {operator_id!r}"
            )
        stages.remove(operator_id)
        checked["filters"].append({
            "id": operator_id,
            "tail": _token_list(item.get("tail", ()), "filter tail")})
    if stages:
        raise ValueError(
            f"prompt pieces: missing filter operators {sorted(stages)}"
        )
    joins = {join.id: join for join in spec.info.joins}
    operator_ids = set(joins)
    for item in pieces.get("joins", ()):
        operator_id = item.get("id")
        if operator_id not in operator_ids:
            raise ValueError(
                f"prompt pieces: unknown join operator {operator_id!r}"
            )
        operator_ids.remove(operator_id)
        checked["joins"].append(
            _pair_piece(item, joins[operator_id].relations, "join"))
    if operator_ids:
        raise ValueError(
            f"prompt pieces: missing join operators {sorted(operator_ids)}"
        )
    classifies = {operator.id: operator for operator in spec.info.classifies}
    operator_ids = set(classifies)
    for item in pieces.get("classifies", ()):
        operator_id = item.get("id")
        if operator_id not in operator_ids:
            raise ValueError(
                f"prompt pieces: unknown classify operator {operator_id!r}")
        operator_ids.remove(operator_id)
        operator = classifies[operator_id]
        if operator.partner is None:
            checked["classifies"].append({
                "id": operator_id,
                "tail": _token_list(item.get("tail", ()), "classify tail")})
        else:
            checked["classifies"].append(
                _pair_piece(item, operator.relations, "classify"))
    if operator_ids:
        raise ValueError(
            f"prompt pieces: missing classify operators {sorted(operator_ids)}")
    operator_ids = {operator.id for operator in spec.info.scores}
    for item in pieces.get("scores", ()):
        operator_id = item.get("id")
        if operator_id not in operator_ids:
            raise ValueError(
                f"prompt pieces: unknown score operator {operator_id!r}")
        operator_ids.remove(operator_id)
        checked["scores"].append({
            "id": operator_id,
            "tail": _token_list(item.get("tail", ()), "score tail")})
    if operator_ids:
        raise ValueError(
            f"prompt pieces: missing score operators {sorted(operator_ids)}")
    operator_ids = {operator.id for operator in spec.info.extracts}
    for item in pieces.get("extracts", ()):
        operator_id = item.get("id")
        if operator_id not in operator_ids:
            raise ValueError(
                f"prompt pieces: unknown extract operator {operator_id!r}")
        operator_ids.remove(operator_id)
        checked["extracts"].append({
            "id": operator_id,
            "tail": _token_list(item.get("tail", ()), "extract tail")})
    if operator_ids:
        raise ValueError(
            f"prompt pieces: missing extract operators {sorted(operator_ids)}")
    return checked


def _pair_piece(item, relations, kind) -> dict:
    """Check the anchor, frame, label, and tail of a two-document request."""
    if item.get("anchor") not in relations:
        raise ValueError(
            f"prompt pieces: {kind} {item.get('id')!r} anchors on an alias "
            f"it does not read: {item.get('anchor')!r}")
    return {"id": item["id"], "anchor": item["anchor"],
            **{name: _token_list(item.get(name, ()), f"{kind} {name}")
               for name in ("frame", "label", "tail")}}


@lru_cache(maxsize=4)
def load_tokenizer(name: str):
    """Return a callable tokenizing a list of texts with Gigatoken.

    Gigatoken loads the Hugging Face tokenizer `name` and encodes a
    batch in parallel, without special tokens. Each document's ids are
    a view into one flat array.
    """
    import awkward
    from gigatoken import Tokenizer

    tokenizer = Tokenizer(name)

    def encode(texts):
        batch = tokenizer.encode_batch(list(texts))
        flat = _tokens(awkward.to_numpy(awkward.flatten(batch)))
        ends = np.cumsum(awkward.to_numpy(awkward.num(batch)))
        return np.split(flat, ends[:-1]) if len(ends) else []

    return encode


class DocumentTokens:
    """Token ids of documents, tokenized on first use and kept.

    Args:
        corpus_rows: Table name to its rows.
        tokenizer: Callable(list of texts) -> list of token id lists.
    """

    def __init__(self, corpus_rows, tokenizer):
        self.corpus_rows = corpus_rows
        self.tokenizer = tokenizer
        self._texts = {}
        self._tokens = {}

    def _text(self, table, column, row_id):
        key = (table, column)
        if key not in self._texts:
            rows = self.corpus_rows[table]
            values = (rows.column(column).to_pylist()
                      if hasattr(rows, "column") else [row[column] for row in rows])
            self._texts[key] = dict(zip(map(str, _ids(rows)), values))
        return self._texts[key][row_id]

    def fetch(self, documents) -> None:
        """Tokenize the (table, column, id) documents not seen yet."""
        missing = [document for document in dict.fromkeys(documents)
                   if document not in self._tokens]
        if not missing:
            return
        encoded = self.tokenizer([self._text(*document) for document in missing])
        for document, ids in zip(missing, encoded):
            self._tokens[document] = _tokens(ids)

    def __getitem__(self, document) -> np.ndarray:
        return self._tokens[document]


class _PartnerTrie:
    """Prefix trie sizes of partner documents that one tail follows.

    The documents are sorted once. Two sorted neighbours of a subset
    share the shortest common prefix of the sorted run between them, so
    a subset's trie size is its length minus a range minimum per
    neighbour pair.

    Args:
        documents: The document tokens, fetched.
        keys: The ((table, column), id) partner documents.
        tail: Token ids after every partner document.
    """

    def __init__(self, documents, keys, tail):
        keys = list(keys)
        tail = _tokens(tail)
        packed = [np.concatenate((documents[(*table_set, row_id)], tail))
                  .astype(">u4").tobytes() for table_set, row_id in keys]
        order = sorted(range(len(keys)), key=packed.__getitem__)
        self.rank = {keys[index]: rank for rank, index in enumerate(order)}
        self.length = np.array([len(packed[index]) // 4 for index in order],
                               dtype=np.int64)
        shared = np.zeros(max(len(order) - 1, 0), dtype=np.int64)
        for position, (earlier, later) in enumerate(zip(order, order[1:])):
            left, right = packed[earlier], packed[later]
            length = min(len(left), len(right))
            differs = (np.frombuffer(left, np.uint8, length)
                       != np.frombuffer(right, np.uint8, length))
            first = int(differs.argmax()) if differs.any() else length
            shared[position] = first // 4
        self.levels = [shared]
        width = 1
        while 2 * width <= len(shared):
            last = self.levels[-1]
            self.levels.append(np.minimum(last[:-width], last[width:]))
            width *= 2

    def size(self, members) -> int:
        """Return the prefix trie size of the member documents with the tail."""
        ranks = np.sort(np.fromiter((self.rank[key] for key in members),
                                    dtype=np.int64, count=len(members)))
        total = int(self.length[ranks].sum())
        if len(ranks) < 2:
            return total
        low, high = ranks[:-1], ranks[1:]
        level = np.floor(np.log2(high - low)).astype(np.int64)
        for step in np.unique(level).tolist():
            pick = level == step
            table = self.levels[step]
            total -= int(np.minimum(
                table[low[pick]], table[high[pick] - (1 << step)]).sum())
        return total


def _encoded(column) -> tuple[np.ndarray, pa.Array]:
    """Return a string id column as dictionary indices and the dictionary."""
    column = pc.cast(column, pa.string())
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    encoded = pc.dictionary_encode(column)
    return np.asarray(encoded.indices), encoded.dictionary


def minimum_input_tokens(spec, pieces, filter_answers, join_answers,
                         documents: DocumentTokens,
                         classify_answers=None, score_answers=None,
                         extract_answers=None) -> int:
    """Return the fewest input tokens the run's requests need.

    Rows with the same text are one document, whatever their ids or
    tables. A join can hold tens of millions of pairs, so the pairs are
    grouped by anchor in numpy; Python visits documents and anchors.

    Args:
        spec: The query.
        pieces: Validated prompt pieces (`validate_prompt_pieces`).
        filter_answers: Filter operator ID to a table with the relation's
            alias and answers, one row per document asked.
        join_answers: Join operator ID to a table with one ID column per
            relation and answers, one row per evaluated pair.
        documents: The document tokens.
        classify_answers: Classify operator ID to a table with one ID
            column per classified alias, one row per document or joined
            row classified.
        score_answers: Score operator ID to a table with the relation's
            alias column, one row per document scored.
        extract_answers: Extract operator ID to a table with the
            relation's alias column, one row per document read.

    Returns:
        The token count.
    """
    sets = {
        relation.alias: (relation.table, relation.text_column)
        for relation in spec.info.relations
    }
    pre = _tokens(pieces["preamble"])
    records: dict = {}

    def record(alias, row_id) -> _Document:
        key = (sets[alias], str(row_id))
        if key not in records:
            records[key] = _Document(alias=alias, row_id=str(row_id))
        return records[key]

    tails = {item["id"]: tuple(item["tail"]) for item in pieces["filters"]}
    filters = {
        filter_spec.id: filter_spec for filter_spec in spec.info.filters
    }
    for operator_id, table in filter_answers.items():
        alias = filters[operator_id].relation
        question = tails[operator_id]
        ids = pc.unique(pc.cast(table.column(alias), pa.string()))
        for row_id in ids.to_pylist():
            record(alias, row_id).suffixes.add(question)
    # a score asks a filter question and reads its belief, so it is a
    # suffix of the document like a filter
    scores = {operator.id: operator for operator in spec.info.scores}
    for piece in pieces["scores"]:
        table = (score_answers or {}).get(piece["id"])
        if table is None:
            continue
        alias = scores[piece["id"]].relation
        ids = pc.unique(pc.cast(table.column(alias), pa.string()))
        for row_id in ids.to_pylist():
            record(alias, row_id).suffixes.add(tuple(piece["tail"]))
    # an extraction asks a question and reads the answer from the
    # document's own tokens, so it is a suffix of the document like a
    # filter; the tokens fed to read the answer are the engine's regret
    extracts = {operator.id: operator for operator in spec.info.extracts}
    for piece in pieces["extracts"]:
        table = (extract_answers or {}).get(piece["id"])
        if table is None:
            continue
        alias = extracts[piece["id"]].relation
        ids = pc.unique(pc.cast(table.column(alias), pa.string()))
        for row_id in ids.to_pylist():
            record(alias, row_id).suffixes.add(tuple(piece["tail"]))
    pairs = {item["id"]: item for item in pieces["joins"]}
    relations = {join.id: join.relations for join in spec.info.joins}
    pair_answers = dict(join_answers)
    classifies = {operator.id: operator for operator in spec.info.classifies}
    for piece in pieces["classifies"]:
        table = (classify_answers or {}).get(piece["id"])
        if table is None:
            continue
        operator = classifies[piece["id"]]
        if operator.partner is not None:
            pairs[piece["id"]] = piece
            relations[piece["id"]] = operator.relations
            pair_answers[piece["id"]] = table
            continue
        question = tuple(piece["tail"])
        ids = pc.unique(pc.cast(table.column(operator.relation), pa.string()))
        for row_id in ids.to_pylist():
            record(operator.relation, row_id).suffixes.add(question)
    # a member key names one distinct partner set of one join or joined-row
    # classification: the sorted partner indices of its answer table
    members_of: dict = {}
    for operator_id, table in pair_answers.items():
        piece = pairs[operator_id]
        anchor = piece["anchor"]
        (partner,) = [alias for alias in relations[operator_id]
                      if alias != anchor]
        group = (tuple(piece["frame"]), tuple(piece["label"]), tuple(piece["tail"]))
        partner_set = sets[partner]
        anchors, anchor_ids = _encoded(table.column(anchor))
        partners, partner_ids = _encoded(table.column(partner))
        if not len(anchors):
            continue
        order = np.lexsort((partners, anchors))
        anchors, partners = anchors[order], partners[order]
        starts = np.flatnonzero(np.r_[True, anchors[1:] != anchors[:-1]])
        ends = np.r_[starts[1:], len(anchors)]
        for start, end in zip(starts.tolist(), ends.tolist()):
            members = np.unique(partners[start:end])
            key = (operator_id, members.tobytes())
            if key not in members_of:
                members_of[key] = frozenset(
                    (partner_set, row_id) for row_id in pc.take(
                        partner_ids, pa.array(members)).to_pylist())
            document = record(anchor, anchor_ids[int(anchors[start])].as_py())
            document.suffixes.add(group[0] + group[1])
            document.groups.setdefault(group, set()).add(key)

    documents.fetch([(*table_set, row_id) for table_set, row_id in records]
                    + [(*member_set, row_id)
                       for members in members_of.values()
                       for member_set, row_id in members])
    total = prefix_trie_size(
        np.concatenate((pre, documents[(*table_set, row_id)]))
        for table_set, row_id in records)
    suffix_sizes: dict = {}
    partner_sizes: dict = {}
    partner_tries: dict = {}
    partners = frozenset().union(*members_of.values())
    for document in _merged_by_text(records, documents).values():
        suffixes = frozenset(document.suffixes)
        if suffixes not in suffix_sizes:
            suffix_sizes[suffixes] = prefix_trie_size(suffixes)
        total += suffix_sizes[suffixes]
        for (_, _, tail), keys in document.groups.items():
            key = (tail, frozenset(keys))
            if key not in partner_sizes:
                if tail not in partner_tries:
                    partner_tries[tail] = _PartnerTrie(documents, partners, tail)
                partner_sizes[key] = partner_tries[tail].size(
                    frozenset().union(*(members_of[member] for member in keys)))
            total += partner_sizes[key]
    return total


def _merged_by_text(records, documents) -> dict:
    """Merge the records of rows with the same text into one each."""
    merged: dict = {}
    for (table_set, row_id), document in records.items():
        text = documents[(*table_set, row_id)].tobytes()
        same = merged.setdefault(
            text, _Document(alias=document.alias, row_id=document.row_id))
        same.suffixes |= document.suffixes
        for group, keys in document.groups.items():
            same.groups.setdefault(group, set()).update(keys)
    return merged


def _fresh_tokens(measurements) -> int | None:
    if "fresh_tokens" not in measurements:
        return None
    fresh = measurements["fresh_tokens"]
    if isinstance(fresh, bool) or not isinstance(fresh, int) or fresh < 0:
        raise ValueError("fresh_tokens must be a nonnegative integer")
    return fresh


def _reported_input_tokens(measurements) -> int | None:
    if "input_tokens" not in measurements:
        return None
    total = measurements["input_tokens"]
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("input_tokens must be a nonnegative integer")
    return total


def input_tokens(spec, pieces, filter_answers, join_answers,
                 documents: DocumentTokens,
                 classify_answers=None, score_answers=None,
                 extract_answers=None) -> int | None:
    """Count full prompt inputs, or return None for missing answer tables."""
    sets = {
        relation.alias: (relation.table, relation.text_column)
        for relation in spec.info.relations
    }

    def document_tokens(table, alias):
        indices, ids = _encoded(table.column(alias))
        keys = [(*sets[alias], row_id) for row_id in ids.to_pylist()]
        documents.fetch(keys)
        counts = np.bincount(indices, minlength=len(keys))
        return sum(int(count) * len(documents[key])
                   for key, count in zip(keys, counts))

    total = 0
    preamble = len(pieces["preamble"])
    filters = {item.id: item for item in spec.info.filters}
    for piece in pieces["filters"]:
        table = filter_answers.get(piece["id"])
        if table is None:
            return None
        alias = filters[piece["id"]].relation
        total += len(table) * (preamble + len(piece["tail"]))
        total += document_tokens(table, alias)
    joins = {item.id: item for item in spec.info.joins}
    for piece in pieces["joins"]:
        table = join_answers.get(piece["id"])
        if table is None:
            return None
        total += len(table) * (preamble + sum(
            len(piece[name]) for name in ("frame", "label", "tail")))
        total += sum(document_tokens(table, alias)
                     for alias in joins[piece["id"]].relations)
    classifies = {operator.id: operator for operator in spec.info.classifies}
    for piece in pieces["classifies"]:
        table = (classify_answers or {}).get(piece["id"])
        if table is None:
            return None
        operator = classifies[piece["id"]]
        total += len(table) * (preamble + sum(
            len(piece.get(name, ())) for name in ("frame", "label", "tail")))
        total += sum(document_tokens(table, alias)
                     for alias in operator.relations)
    scores = {operator.id: operator for operator in spec.info.scores}
    for piece in pieces["scores"]:
        table = (score_answers or {}).get(piece["id"])
        if table is None:
            return None
        total += len(table) * (preamble + len(piece["tail"]))
        total += document_tokens(table, scores[piece["id"]].relation)
    extracts = {operator.id: operator for operator in spec.info.extracts}
    for piece in pieces["extracts"]:
        table = (extract_answers or {}).get(piece["id"])
        if table is None:
            return None
        total += len(table) * (preamble + len(piece["tail"]))
        total += document_tokens(table, extracts[piece["id"]].relation)
    return total


def token_metrics(spec, output, corpus_rows, stores=None) -> dict:
    """Compute token metrics from the most detailed data the engine provides.

    - With prompt pieces, derive input and minimum tokens from the answer
      tables. Fresh tokens are required. Regret is fresh minus minimum.
    - With prompt pieces and an engine input-token total, the engine
      tokenized each full prompt itself, so its tokens can differ from the
      pieces where two pieces meet. The minimum is scaled by the engine
      total over the piece total, regret is floored at zero, and
      `regret_approximate` is True.
    - Without prompt pieces, use the engine's input-token total. Minimum
      and regret are unavailable because a total does not describe prefixes.
    - A missing answer table makes input tokens unavailable. An empty answer
      table contributes zero input tokens.

    Args:
        spec: The query.
        output: The run output.
        corpus_rows: Table name to its rows.
        stores: Tokenizer name -> DocumentTokens, kept across the
            queries of one run so each document is tokenized once.
    """
    fresh = _fresh_tokens(output.measurements)
    reported_input = _reported_input_tokens(output.measurements)
    if output.prompt_pieces is None:
        return {"input_tokens": reported_input, "fresh_tokens": fresh,
                "minimum_tokens": None, "regret_tokens": None,
                "regret_approximate": False}
    if fresh is None:
        raise ValueError("prompt pieces need a fresh_tokens measurement")
    if output.filter_answers is None or output.join_answers is None:
        raise ValueError("prompt pieces need filter and join answers")
    if spec.info.classifies and output.classify_answers is None:
        raise ValueError("prompt pieces need classify answers")
    if spec.info.scores and output.score_answers is None:
        raise ValueError("prompt pieces need score answers")
    if spec.info.extracts and output.extract_answers is None:
        raise ValueError("prompt pieces need extract answers")
    pieces = validate_prompt_pieces(spec, output.prompt_pieces)
    stores = {} if stores is None else stores
    name = pieces["tokenizer"]
    if name not in stores:
        stores[name] = DocumentTokens(corpus_rows, load_tokenizer(name))
    answers = (output.filter_answers, output.join_answers, stores[name],
               output.classify_answers, output.score_answers,
               output.extract_answers)
    minimum = minimum_input_tokens(spec, pieces, *answers)
    piece_input = input_tokens(spec, pieces, *answers)
    if reported_input is not None:
        if piece_input:
            minimum = round(minimum * reported_input / piece_input)
        return {"input_tokens": reported_input, "fresh_tokens": fresh,
                "minimum_tokens": minimum,
                "regret_tokens": max(fresh - minimum, 0),
                "regret_approximate": True}
    if fresh < minimum:
        raise ValueError(
            f"fresh_tokens ({fresh}) is below the minimum the requests need "
            f"({minimum})")
    return {
        "input_tokens": piece_input,
        "fresh_tokens": fresh, "minimum_tokens": minimum,
        "regret_tokens": fresh - minimum, "regret_approximate": False,
    }
