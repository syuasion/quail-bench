"""The QUAIL-B document sets: pinned sources, sampling, and identity.

Eight document sets by default (IMDB, BioDEX, FEVER, LePaRD, SWE-Next
agent trace snapshots, tau-bench support traces, SWE-rebench issue runs,
and Terminal Wrench runs) plus the optional PrivacyPolicies set. Every table is sampled
from a pinned upstream revision with one seed, so a scale factor names
one exact corpus.
"""

import hashlib
import heapq
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from quail_b._files import (
    GROUND_TRUTH_ROOT,
    _cached_file,
    _list_files,
    _location,
    _read_bytes,
)
from quail_b._files import PUBLIC_BUCKET as PUBLIC_BUCKET

DATA_SEED = 20260818
CACHE_SCHEMA_VERSION = 11
# The labeled corpus for each scale factor, saved beside its labels in
# the public bucket. build_sets downloads it instead of rebuilding from
# the sources; the corpus id is checked after download.
PUBLISHED_CORPORA = {
    0.1: "c_e380e53469a6a62de83a452fbc51a73e",
    0.5: "c_38414b12549ab6243397fd41522e67ad",
    1.0: "c_3a8712778b5fe0938abe729770c4d23d",
}
LEPARD_POSITIVE_PAIRS = 5_000
# The base count for scaling. SWE-Next at the pinned revision yields
# 17,711 eligible snapshots, so the full scale takes every one of them
# while smaller scale factors keep their first round(17,718 * sf) rows.
AGENT_TRACE_DOCUMENTS = 17_718
AGENT_TRACE_TURN_INTERVAL = 5
AGENT_TRACE_MAX_TOKENS = 24_000
AGENT_TRACE_TOKENIZER = "Qwen/Qwen3-4B-FP8"
AGENT_TRACE_TOKENIZER_REVISION = (
    "96b30dc13593a244a5e59e84687309f53c375cfa"
)

# tau-bench's historical trajectories: two models, four trials per task,
# on the airline (50 tasks) and retail (115 tasks) domains. The pinned
# files hold eight trials of each task for sonnet-35-new and four for
# gpt-4o, so only the first four trials of each model are kept. The agent
# policy is the system prompt of every trace and is left out of the
# transcript.
TAU_BENCH_COMMIT = "59a200c6d575d595120f1cb70fea53cef0632f6b"
TAU_BENCH_RUNS = (
    ("gpt-4o", "airline"), ("gpt-4o", "retail"),
    ("sonnet-35-new", "airline"), ("sonnet-35-new", "retail"),
)
TAU_BENCH_URL = (
    "https://raw.githubusercontent.com/sierra-research/tau-bench/"
    "{commit}/historical_trajectories/{model}-{domain}.json"
)
SUPPORT_TASKS = 165
SUPPORT_TRIALS_PER_MODEL = 4
SUPPORT_TRACES_PER_TASK = SUPPORT_TRIALS_PER_MODEL * 2

# SWE-rebench OpenHands runs: issues with at least ISSUE_RUN_SOURCE_RUNS
# runs, of which the first ISSUE_RUNS_PER_ISSUE whose transcript fits
# the token cap are kept. Tool output is cut in the transcript, and
# every message is cut in the messages table, so one run reads like an
# agent_traces snapshot. With tool output cut to 400 characters, the
# median run is about 20,000 Qwen3 tokens and seven in ten fit the cap.
ISSUE_RUN_ISSUES = 400
ISSUE_RUNS_PER_ISSUE = 8
ISSUE_RUN_SOURCE_RUNS = 12
ISSUE_RUN_MAX_TOKENS = 24_000
TRANSCRIPT_TOOL_CHARS = 400
# RUNS-3 reads two patches in one prompt, which must fit one
# forward pass of the reference judge (41,943 tokens for Qwen3 32B on
# one H100), so a longer patch keeps its start and end.
ISSUE_PATCH_MAX_CHARS = 40_000
MESSAGE_MAX_CHARS = 2_000

# Terminal Wrench: terminal tasks run by Claude Opus 4.6, Gemini 3.1 Pro,
# and GPT-5.4. Hack runs come from its sanitized split: the agent was
# asked to pass the verifier by any means, then the red-team prompt was
# removed and the agent text rewritten to drop mentions of hacking.
# Baseline runs solved the task without being asked to hack. A scale
# factor samples tasks; runs over the token cap are left out.
WRENCH_REPO = "few-sh/terminal-wrench"
WRENCH_INDEX_REPO = "few-sh/terminal-wrench-trajectories"
WRENCH_SPLITS = {"sanitized_hack": "hack", "baseline": "baseline"}
WRENCH_TASKS = 331
WRENCH_OUTPUT_CHARS = 2_000
WRENCH_MAX_TOKENS = 24_000

# CRMArena-Pro's sales calls: synthetic calls between sales reps and
# customers, saved in the CRM databases of its B2B and B2C companies.
# Every kept call belongs to a deal (an Opportunity record); calls on
# leads are left out. A scale factor samples deals in each company.
CRMARENA_COMMIT = "6d84f3d71305af0fd3d5ed3c1936b7887464455a"
CRMARENA_URL = (
    "https://raw.githubusercontent.com/SalesforceAIResearch/CRMArena/"
    "{commit}/local_data/crmarenapro_{domain}_data.db"
)
SALES_DOMAINS = ("b2b", "b2c")
SALES_DEALS = {"b2b": 1_170, "b2c": 2_290}

# CUAD: 510 commercial contracts from SEC filings, with 41 clause
# categories marked by lawyers. A contract over the token cap is left
# out, as an agent trace is; the kept contracts are numbered in title
# order, and a scale factor keeps a seeded sample of them. The 8
# categories answered with text, such as the parties or the governing
# law, are extraction targets; the other 33 are answered Yes or No,
# and the clause_types table holds each with the question its
# labelers answered, for joins.
CUAD_REPO = "theatticusproject/cuad"
CUAD_JSON = "CUAD_v1/CUAD_v1.json"
CUAD_CSV = "CUAD_v1/master_clauses.csv"
CUAD_MAX_TOKENS = 24_000
CUAD_CONTRACTS = 449    # the contracts within the cap at the pinned revision
CUAD_FIELD_CATEGORIES = (
    "Document Name", "Parties", "Agreement Date", "Effective Date",
    "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal",
    "Governing Law",
)

# Exact source snapshots for the benchmark corpus.  The row selection below
# is deterministic only when the upstream revisions are fixed as well as the
# sampling seed.
SOURCE_REVISIONS = {
    "stanfordnlp/imdb": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
    "BioDEX/BioDEX-Reactions":
        "01a5dacdabd144a120af04931a11a99febd48432",
    # FEVER's parquet files live on its conversion ref, so this is the
    # resolved commit for refs/convert/parquet rather than the default branch.
    "fever/fever": "5f577157472532aa1d9924d2df63aac44f70cf2b",
    "rmahari/LePaRD": "0194f95c3091acceab3b887c9b09ef432cf84052",
    "TIGER-Lab/SWE-Next-SFT-Trajectories":
        "e378a60ddd7050fe9519a31a4d41d4872eeec6ac",
    "mukund/PrivacyPolicies":
        "8fd6abfc7ca99d1f95c7f3f3a5dd5ea0cf9b7deb",
    "sierra-research/tau-bench": TAU_BENCH_COMMIT,
    "nebius/SWE-rebench-openhands-trajectories":
        "35455389ab51bf5e2306bfd436ef72d0f98bf882",
    WRENCH_REPO: "6ab64c43a4c6a4d9aa53c2425f37e715257123a3",
    WRENCH_INDEX_REPO: "622449769ec6ddf33f770794489b8cd21712a0ad",
    "SalesforceAIResearch/CRMArena": CRMARENA_COMMIT,
    CUAD_REPO: "a3c393f5d103fd0c516374e4fdff676c8176dcb1",
}

# Base document counts at sf=1. LePaRD scales sampled citation pairs
# before it deduplicates the two document tables.
SETS = {
    "reviews": 50_000,
    "reports": 5_000,
    "claims": 5_000,
    "agent_traces": AGENT_TRACE_DOCUMENTS,
    "policies": 1_000_000,
    "contracts": CUAD_CONTRACTS,
}

ASPECTS = ["the acting", "the plot", "the directing", "the cinematography",
           "the soundtrack", "the pacing", "the ending", "the dialogue",
           "the special effects", "the character development",
           "the screenplay", "the editing"]

SCENARIOS = [
    # marketing
    "You stop using the app, but months later you start getting ads "
    "from companies you have never heard of, based on things you "
    "searched for while you were still using it.",
    "You buy a product once, and then you keep getting emails and push "
    "notifications about similar products, even after you unsubscribe "
    "from the mailing list.",
    "You notice that the ads you see on other websites change right "
    "after you browse this service, as if your activity here followed "
    "you around the internet.",
    "You create an account just to try the free version, and within a "
    "week you start getting phone calls from salespeople who know your "
    "name and what features you looked at.",
    "You fill out a survey on the app, and later a completely different "
    "company contacts you about the exact topics you mentioned in "
    "your answers.",
    "You use the app for a few weeks, then delete it, but you keep "
    "seeing ads for it on social media that reference things you did "
    "inside the app.",
    "You sign up with a throwaway email, but the app somehow starts "
    "showing you ads related to purchases you made with your main "
    "email at other stores.",
    "You mention a product in a chat on the service, and within hours "
    "you see targeted ads for that exact product on other platforms.",
    "You opt out of marketing emails, but the company still sends you "
    "promotional messages disguised as account updates or security "
    "alerts.",
    "You notice the app suggesting friends who are customers of a "
    "partner company, even though you never shared your contacts.",
    # third-party sharing
    "A data broker contacts you with an offer, and when you ask how "
    "they got your information, they name this service as the source.",
    "You apply for a loan and the lender already has a profile of "
    "your spending habits, built from data this service shared with "
    "a financial analytics company.",
    "Your health insurance premium goes up, and when you investigate "
    "you find the insurer received wellness data that you entered "
    "into this app.",
    "A background check company has records of your activity on this "
    "service, even though you never gave them permission to access it.",
    "You discover that a political campaign has your personal details "
    "and browsing habits, traced back to a data-sharing agreement "
    "with this service.",
    "Your employer uses a workplace analytics tool that has data about "
    "your personal usage of this service, shared without your knowledge.",
    "A research firm publishes a study that includes aggregated data "
    "about users like you, and you can identify yourself from the "
    "details even though names were removed.",
    "You find your profile information listed on a people-search "
    "website, and the data matches exactly what you entered into "
    "this service.",
    "A retailer you have never visited sends you a coupon by mail, "
    "using your home address and product preferences from this app.",
    "You learn that a foreign government obtained your account data "
    "through a third party that this service shared it with.",
    # law enforcement
    "Police show up with a warrant for records of your activity on "
    "the service, and the company hands over six months of your chat "
    "history without telling you first.",
    "A government agency requests your location data from the past "
    "year, and the company provides it without requiring a court "
    "order.",
    "You are involved in a lawsuit, and the opposing side introduces "
    "your private messages from this service as evidence, obtained "
    "through a subpoena the company complied with.",
    "An immigration agency accesses your travel-related searches and "
    "account activity through a bulk data request to the company.",
    "You find out that the company gave law enforcement real-time "
    "access to your location for an investigation you were never "
    "told about.",
    "A tax authority receives your transaction records from this "
    "service as part of a compliance program the company participates "
    "in voluntarily.",
    "You are detained at a border crossing, and the officers already "
    "have a printout of your recent activity on this service.",
    "A local police department uses facial recognition to match a "
    "photo from your profile on this service to surveillance footage.",
    "Your account is flagged and frozen after the company runs an "
    "automated scan and reports your content to a government agency.",
    "A foreign court orders the company to hand over your data, and "
    "the company complies even though you live in a different country.",
    # data retention
    "You delete your account, but a year later you discover the "
    "company still has your photos stored on its servers.",
    "You request a copy of your data and find that the company kept "
    "records of searches you made five years ago, long after you "
    "stopped using the service.",
    "You close your account and later reopen one with the same email, "
    "and all your old preferences and history are still there.",
    "You ask the company to delete your data, they confirm it is "
    "done, but a data breach months later reveals your old records "
    "were still in their backup systems.",
    "You find out the company keeps a permanent record of every "
    "version of your profile, including photos and bios you changed "
    "years ago.",
    "You cancel your subscription, but the company continues to "
    "store and analyze your usage patterns for its own research.",
    "Your messages to other users remain visible to those users "
    "even after you delete your account, with your name still "
    "attached.",
    "You discover that the company retains your payment information "
    "indefinitely, even after you remove your credit card from the "
    "account settings.",
    "You move to a country with stricter data laws and request "
    "deletion, but the company says your data is stored in a "
    "jurisdiction where they are not required to delete it.",
    "You stop paying for the premium tier, but the company keeps "
    "all the data you uploaded during your subscription period "
    "without any stated expiration date.",
    # tracking
    "You use the app only at home, but it builds a detailed map of "
    "every store and restaurant you visit, using your phone's "
    "location in the background.",
    "You turn off location services for the app, but it still "
    "figures out where you are by scanning nearby Wi-Fi networks "
    "and Bluetooth devices.",
    "You browse the service on your laptop, and later when you open "
    "the app on your phone, it knows exactly which pages you visited "
    "on the laptop.",
    "You visit a physical store, and the app sends you a notification "
    "about a sale at that store moments later, even though you never "
    "searched for it.",
    "You notice the app has a record of how long you spend on each "
    "screen, how fast you scroll, and exactly where you tap.",
    "You clear your browser cookies, but the service still recognizes "
    "you the next time you visit, using device fingerprinting or "
    "other tracking methods.",
    "You use a VPN to hide your location, but the app still shows "
    "you local content, suggesting it has another way to determine "
    "where you are.",
    "You create a second account under a different name, but the "
    "service links it to your original account within days.",
    "You lend your phone to a friend, and the app records their "
    "usage pattern as yours, mixing their browsing into your profile.",
    "You find out the app tracks which other apps are installed on "
    "your phone and uses that information to build a profile of "
    "your interests.",
    # content and communications
    "You send a private photo to one person through the service, "
    "and later find it was scanned and flagged by the company's "
    "automated content review system.",
    "You write a private note in the app that you never share, and "
    "later the company uses the text to train a language model.",
    "You have a private video call on the service, and you later "
    "discover the company recorded and stored a transcript of the "
    "conversation.",
    "You upload a document to the service for personal storage, and "
    "the company uses its contents to improve its search algorithm.",
    "You send an encrypted message, but the company can still read "
    "it because the encryption keys are stored on the company's "
    "servers.",
    "You post something to a small private group, and the company's "
    "moderation system shares it with an external review team in "
    "another country.",
    "You draft a message but never send it, and later discover the "
    "company saved the draft and analyzed its contents.",
    "You share a voice message with a friend, and the company "
    "converts it to text and adds it to your advertising profile.",
    "You delete a post you made, but the company keeps a copy and "
    "continues to use it for content recommendations.",
    "You set your profile to private, but the company still allows "
    "search engines to index your profile photo and display name.",
    # AI and automated decisions
    "You apply for a service upgrade, and an algorithm denies your "
    "request based on your usage patterns, with no explanation and "
    "no way to appeal.",
    "The app automatically adjusts the prices you see based on how "
    "much it predicts you are willing to pay, without telling you.",
    "You are banned from the platform by an automated system that "
    "flagged your content, and no human ever reviews your appeal.",
    "The service uses your photos to train a facial recognition "
    "model, and that model is later sold to a company you have "
    "never interacted with.",
    "An algorithm decides which customer service tier you belong to, "
    "so your support tickets are deprioritized compared to users the "
    "system considers more valuable.",
    "You are shown a different version of the terms of service than "
    "other users, tailored by an algorithm based on your likelihood "
    "of reading the full text.",
    "The service uses your data to build a creditworthiness score "
    "that other companies can purchase and use in their own lending "
    "decisions.",
    "An automated system flags your account as suspicious based on "
    "your browsing patterns, and your access is restricted without "
    "any notification.",
    "The app uses your purchase history to predict your political "
    "views and sells that prediction to a data analytics firm.",
    "You receive different search results than other users because "
    "an algorithm decided what it thinks you want to see, without "
    "telling you it is personalizing.",
    # security and breaches
    "Your password is leaked in a data breach, and you find out "
    "about it from a news article before the company ever contacts "
    "you.",
    "The company suffers a breach that exposes your home address, "
    "phone number, and payment history, and offers you only one "
    "year of credit monitoring.",
    "Your biometric data, like a fingerprint or face scan, is stolen "
    "in a breach, and unlike a password, you cannot change it.",
    "You learn that an employee at the company accessed your account "
    "and read your private messages out of personal curiosity.",
    "The company stores your password in a way that allows anyone who "
    "breaks into their database to read it directly.",
    "A contractor working for the company downloads a database backup "
    "containing your data and takes it with them when they leave.",
    "Your account is taken over by someone who called the company's "
    "support line and convinced them to reset your password.",
    "The company shares your data with a partner whose security "
    "practices are weaker, and that partner gets breached.",
    "You discover that the company has no way to tell you which "
    "employees accessed your data or when.",
    "A security researcher publicly discloses a vulnerability that "
    "exposed your data, and the company had known about it for "
    "months without fixing it.",
    # children and family
    "Your thirteen-year-old child signs up for the service by "
    "entering a fake birth date, and the company collects and sells "
    "their data just like an adult's.",
    "You share a family account with your children, and the company "
    "builds advertising profiles for each family member, including "
    "the minors.",
    "Your child's school requires this service for homework, and "
    "the company uses the child's usage data for purposes beyond "
    "education.",
    "You give the app permission to access your contacts, and it "
    "starts sending messages to your children's phone numbers "
    "inviting them to join.",
    "You find out the company kept detailed records of your child's "
    "online activity from when they were ten years old, and those "
    "records are still accessible years later.",
    "The service recommends content to your teenager based on a "
    "profile built from data collected before they were old enough "
    "to consent.",
    "Your family's smart home device shares your children's voice "
    "recordings with this service, which uses them for product "
    "development.",
    "You set up parental controls, but the company's data collection "
    "practices apply the same way to your child's account as to "
    "yours.",
    "A classmate's parent uses the app to look up information about "
    "your child, and the service provides it because your child's "
    "profile is not fully private by default.",
    "You discover that the company used your child's data to train "
    "an AI model, even though your child's account was flagged as "
    "belonging to a minor.",
    # financial and sensitive data
    "You link your bank account to the service for payments, and the "
    "company uses your transaction history to build a spending profile "
    "that it shares with advertisers.",
    "You enter your Social Security number for identity verification, "
    "and the company stores it indefinitely, even after verification "
    "is complete.",
    "The service infers your income level from your usage patterns "
    "and uses it to decide which subscription plans to show you.",
    "You authorize a one-time payment, but the company stores your "
    "full credit card details and later charges you for a renewal "
    "you did not agree to.",
    "Your medical information, entered into a wellness feature of "
    "the app, is shared with an insurance company as part of a data "
    "partnership.",
    "The service tracks which financial articles you read and sells "
    "that behavioral data to investment firms.",
    "You discover that the company has been collecting information "
    "about your race, religion, or sexual orientation from your "
    "profile and activity, and using it for ad targeting.",
    "You apply for a job through the service, and the employer sees "
    "a risk score calculated from your financial data on the "
    "platform.",
    "You connect a fitness tracker to the app, and it shares your "
    "health metrics with third parties without a separate consent "
    "step.",
    "The service combines your purchase history with public records "
    "to estimate your net worth, and makes that estimate available "
    "to its business partners.",
]


def _n_docs(name, sf):
    return max(8, int(SETS[name] * sf))


def _n_lepard_pairs(sf):
    return max(8, int(LEPARD_POSITIVE_PAIRS * sf))


def _n_agent_documents(sf):
    return min(
        AGENT_TRACE_DOCUMENTS,
        max(8, round(AGENT_TRACE_DOCUMENTS * sf)),
    )


def _agent_message_text(message) -> str:
    """Render one agent message in the stored trace format."""
    content = message.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(
            content, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False)
    return f"[{str(message.get('role', '')).upper()}]\n{content}"


def _agent_snapshot_boundaries(messages) -> tuple[str, list[tuple[int, int]]]:
    """Render one trace and return every fifth completed turn."""
    targets = {}
    assistant_turn = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        assistant_turn += 1
        if assistant_turn % AGENT_TRACE_TURN_INTERVAL:
            continue
        end = index + 1
        if end < len(messages) and messages[end].get("role") == "tool":
            end += 1
        targets[end] = assistant_turn

    pieces = []
    boundaries = []
    length = 0
    for message_number, message in enumerate(messages, start=1):
        piece = _agent_message_text(message)
        if pieces:
            length += 2
        pieces.append(piece)
        length += len(piece)
        if message_number in targets:
            boundaries.append((targets[message_number], length))
    return "\n\n".join(pieces), boundaries


def _agent_has_issue(messages) -> bool:
    """Return whether the trace contains a nonempty user issue."""
    return any(
        message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and bool(message["content"].strip())
        for message in messages
    )


def _agent_trace_rows(trace_index, messages, tokenizer) -> list[dict]:
    """Build the eligible snapshots for one SWE-Next trace."""
    if not _agent_has_issue(messages):
        return []
    text, boundaries = _agent_snapshot_boundaries(messages)
    rows = []
    for turn_index, end in boundaries:
        snapshot = text[:end]
        token_count = len(tokenizer.encode(
            snapshot, add_special_tokens=False))
        if token_count > AGENT_TRACE_MAX_TOKENS:
            continue
        rows.append({
            "id": f"at{trace_index:04d}-t{turn_index:03d}",
            "trace": snapshot,
            "trajectory_id": f"at{trace_index:04d}",
            "turn_index": turn_index,
            "token_count": token_count,
        })
    return rows


def _select_agent_snapshots(snapshots, n, full=False):
    """Take the first n snapshots in trace order, or every one at full scale.

    Args:
        snapshots: The eligible snapshot rows of each trace, in order.
        n: The number of rows wanted.
        full: Whether fewer than n rows is acceptable because n is the
            base count and the source has run out.
    """
    rows = []
    for trace_rows in snapshots:
        rows.extend(trace_rows)
        if len(rows) >= n:
            return rows[:n]
    if full:
        return rows
    raise ValueError(
        f"SWE-Next produced {len(rows)} eligible snapshots, expected {n}")


def _agent_rows(n, full=False):
    """Read SWE-Next and select a nested sample of trace snapshots."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    source = load_dataset(
        "TIGER-Lab/SWE-Next-SFT-Trajectories",
        split="train",
        revision=SOURCE_REVISIONS["TIGER-Lab/SWE-Next-SFT-Trajectories"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        AGENT_TRACE_TOKENIZER,
        revision=AGENT_TRACE_TOKENIZER_REVISION,
    )
    order = np.random.default_rng(DATA_SEED).permutation(len(source))
    snapshots = (
        _agent_trace_rows(
            int(trace_index), source[int(trace_index)]["messages"],
            tokenizer)
        for trace_index in order)
    return _select_agent_snapshots(snapshots, n, full)



# ------------------------------------------------ trace tables

def _message_text(message, tool_chars=None) -> str:
    """Render a message's content, then each tool call on its own line.

    Args:
        message: A chat message with a role, content, and optional
            tool calls.
        tool_chars: Cut a tool message's content to this many
            characters and mark the cut, or None to keep it whole.
    """
    content = message.get("content")
    if content is None:
        content = ""
    elif not isinstance(content, str):
        content = json.dumps(content, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False)
    if (tool_chars is not None and message.get("role") == "tool"
            and len(content) > tool_chars):
        content = (content[:tool_chars]
                   + f"\n[cut {len(content) - tool_chars} characters]")
    lines = [content] if content else []
    for call in message.get("tool_calls") or ():
        function = call.get("function") or call
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, sort_keys=True,
                                   separators=(",", ":"), ensure_ascii=False)
        lines.append(f"[tool call] {function.get('name', '')}({arguments})")
    return "\n".join(lines)


def _transcript(messages, tool_chars=None) -> str:
    """Render every message but the system prompt under a [ROLE] heading."""
    return "\n\n".join(
        f"[{str(message.get('role', '')).upper()}]\n"
        f"{_message_text(message, tool_chars)}"
        for message in messages if message.get("role") != "system")


def _message_rows(trace_id, messages) -> list[dict]:
    """Build one row per message, linked to the messages before it.

    The layout is the one `quail.trace_tables` writes: the id is the
    trace id and turn index, and the prev columns hold the id of the
    message, user message, and assistant message before this one.
    """
    rows = []
    prev_id = prev_user = prev_assistant = None
    for turn_index, message in enumerate(messages):
        role = str(message.get("role", ""))
        message_id = f"{trace_id}/{turn_index}"
        content = _message_text(message)
        if len(content) > MESSAGE_MAX_CHARS:
            content = (content[:MESSAGE_MAX_CHARS]
                       + f"\n[cut {len(content) - MESSAGE_MAX_CHARS} characters]")
        tool_call_id = message.get("tool_call_id")
        rows.append({
            "id": message_id, "trace_id": trace_id, "turn_index": turn_index,
            "role": role, "content": content,
            "tool_call_id": None if tool_call_id is None else str(tool_call_id),
            "prev_id": prev_id, "prev_user_id": prev_user,
            "prev_assistant_id": prev_assistant,
        })
        prev_id = message_id
        if role == "user":
            prev_user = message_id
        elif role == "assistant":
            prev_assistant = message_id
    return rows


def _request(messages) -> str:
    """The first user message of a trace, or an empty string."""
    for message in messages:
        if message.get("role") == "user":
            return _message_text(message)
    return ""


MESSAGE_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("trace_id", pa.string()),
    ("turn_index", pa.int32()),
    ("role", pa.string()),
    ("content", pa.string()),
    ("tool_call_id", pa.string()),
    ("prev_id", pa.string()),
    ("prev_user_id", pa.string()),
    ("prev_assistant_id", pa.string()),
])

SUPPORT_TRACE_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("request", pa.string()),
    ("transcript", pa.string()),
    ("message_count", pa.int32()),
    ("task_id", pa.string()),
    ("domain", pa.string()),
    ("model", pa.string()),
    ("trial", pa.int32()),
    ("reward", pa.float64()),
])

ISSUE_RUN_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("request", pa.string()),
    ("transcript", pa.string()),
    ("patch", pa.string()),
    ("message_count", pa.int32()),
    ("instance_id", pa.string()),
    ("repo", pa.string()),
    ("resolved", pa.int32()),
    ("token_count", pa.int32()),
])


def _n_support_tasks(sf):
    return min(SUPPORT_TASKS, max(8, round(SUPPORT_TASKS * sf)))


def _n_issue_run_issues(sf):
    return min(ISSUE_RUN_ISSUES, max(8, round(ISSUE_RUN_ISSUES * sf)))


def _download(url: str) -> Path:
    """Fetch a source file once into the benchmark cache."""
    from urllib.request import urlopen

    from quail_b._files import _cache_directory

    cached = _cache_directory() / hashlib.sha256(url.encode()).hexdigest()
    if not cached.exists():
        with urlopen(url) as stream:
            data = stream.read()
        temporary = cached.with_suffix(".part")
        temporary.write_bytes(data)
        temporary.replace(cached)
    return cached


def _support_source_tasks() -> list[tuple[str, list[dict]]]:
    """Read tau-bench's trajectories, grouped by task, in seeded order."""
    tasks = {}
    for model, domain in TAU_BENCH_RUNS:
        path = _download(TAU_BENCH_URL.format(
            commit=TAU_BENCH_COMMIT, model=model, domain=domain))
        for record in json.loads(path.read_text()):
            if int(record["trial"]) >= SUPPORT_TRIALS_PER_MODEL:
                continue
            tasks.setdefault(f"{domain}-{record['task_id']}", []).append({
                "domain": domain, "model": model,
                "trial": int(record["trial"]),
                "reward": float(record["reward"]),
                "messages": record["traj"],
            })
    keys = sorted(tasks)
    order = np.random.default_rng(DATA_SEED).permutation(len(keys))
    return [(keys[index], tasks[keys[index]]) for index in order]


def _support_rows(tasks, n) -> tuple[list[dict], list[dict]]:
    """Build the support trace and message rows of the first n tasks.

    Raises:
        ValueError: A task has other than SUPPORT_TRACES_PER_TASK runs,
            or fewer than n tasks exist.
    """
    if len(tasks) < n:
        raise ValueError(f"tau-bench has {len(tasks)} tasks, expected {n}")
    traces, messages = [], []
    for task_id, runs in tasks[:n]:
        if len(runs) != SUPPORT_TRACES_PER_TASK:
            raise ValueError(
                f"task {task_id} has {len(runs)} runs, expected "
                f"{SUPPORT_TRACES_PER_TASK}")
        for run in sorted(runs, key=lambda run: (run["model"], run["trial"])):
            trace_id = f"sp{len(traces):04d}"
            traces.append({
                "id": trace_id,
                "request": _request(run["messages"]),
                "transcript": _transcript(run["messages"]),
                "message_count": len(run["messages"]),
                "task_id": task_id, "domain": run["domain"],
                "model": run["model"], "trial": run["trial"],
                "reward": run["reward"],
            })
            messages.extend(_message_rows(trace_id, run["messages"]))
    return traces, messages


def _issue_run_source():
    """Open SWE-rebench's trajectory file at the pinned revision."""
    from huggingface_hub import hf_hub_download

    name = "nebius/SWE-rebench-openhands-trajectories"
    path = hf_hub_download(name, "trajectories.parquet", repo_type="dataset",
                           revision=SOURCE_REVISIONS[name])
    return pq.ParquetFile(path)


def _issue_run_candidates(instance_ids, n) -> list[str]:
    """Pick the issues to read: 2n with enough runs, in seeded order."""
    counts = {}
    for instance_id in instance_ids:
        counts[instance_id] = counts.get(instance_id, 0) + 1
    issues = sorted(issue for issue, count in counts.items()
                    if count >= ISSUE_RUN_SOURCE_RUNS)
    order = np.random.default_rng(DATA_SEED).permutation(len(issues))
    return [issues[index] for index in order[:2 * n]]


def _issue_run_rows(batches, candidates, n, tokenizer):
    """Build the issue run and message rows from streamed source batches.

    Each run of a candidate issue is rendered as it arrives; a run
    whose transcript is over ISSUE_RUN_MAX_TOKENS is skipped, and an
    issue keeps its first ISSUE_RUNS_PER_ISSUE fitting runs in file
    order. The first n candidates, in candidate order, with that many
    fitting runs make the tables.

    Args:
        batches: Source record batches or tables with instance_id,
            repo, trajectory, model_patch, and resolved columns.
        candidates: The issues to read, in seeded order.
        n: The number of issues wanted.
        tokenizer: Counts a transcript's tokens with `encode`.

    Raises:
        ValueError: Fewer than n candidates have enough fitting runs.
    """
    wanted = set(candidates)
    kept = {issue: [] for issue in candidates}
    for batch in batches:
        for row in batch.to_pylist():
            issue = row["instance_id"]
            if issue not in wanted or len(kept[issue]) >= ISSUE_RUNS_PER_ISSUE:
                continue
            messages = row["trajectory"]
            if isinstance(messages, str):
                messages = json.loads(messages)
            transcript = _transcript(messages, TRANSCRIPT_TOOL_CHARS)
            token_count = len(tokenizer.encode(
                transcript, add_special_tokens=False))
            if token_count > ISSUE_RUN_MAX_TOKENS:
                continue
            kept[issue].append({
                "request": _request(messages), "transcript": transcript,
                "patch": _cut_middle(row.get("model_patch") or "",
                                     ISSUE_PATCH_MAX_CHARS),
                "message_count": len(messages), "instance_id": issue,
                "repo": row["repo"], "resolved": int(row["resolved"]),
                "token_count": token_count, "messages": messages,
            })
    complete = [issue for issue in candidates
                if len(kept[issue]) == ISSUE_RUNS_PER_ISSUE]
    if len(complete) < n:
        raise ValueError(
            f"{len(complete)} issues have {ISSUE_RUNS_PER_ISSUE} fitting "
            f"runs, expected {n}")
    runs, messages = [], []
    for issue in complete[:n]:
        for run in kept[issue]:
            run_id = f"ir{len(runs):05d}"
            messages.extend(_message_rows(run_id, run.pop("messages")))
            runs.append({"id": run_id, **run})
    return runs, messages


def _n_wrench_tasks(sf):
    return min(WRENCH_TASKS, max(8, round(WRENCH_TASKS * sf)))


WRENCH_RUN_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("task_id", pa.string()),
    ("model", pa.string()),
    ("mode", pa.string()),
    ("transcript", pa.string()),
    ("step_count", pa.int32()),
    ("token_count", pa.int32()),
])

WRENCH_STEP_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("run_id", pa.string()),
    ("step_index", pa.int32()),
    ("model", pa.string()),
    ("text", pa.string()),
])

WRENCH_INDEX_COLUMNS = ("task_id", "model", "trajectory_path", "instruction")


def _cut_middle(text: str, limit: int) -> str:
    """Keep the first two thirds and the last third of an over-long text."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return (text[:head] + f"\n[... {len(text) - limit} characters cut ...]\n"
            + text[-(limit - head):])


def _wrench_step(step) -> str:
    """Render one agent step: its message, commands, and terminal output."""
    lines = [f"[AGENT]\n{(step.get('message') or '').strip()}"]
    commands = []
    for call in step.get("tool_calls") or ():
        arguments = call.get("arguments") or {}
        if call.get("function_name") == "bash_command":
            commands.append(arguments.get("keystrokes", "").rstrip("\n"))
        else:
            commands.append(f"{call.get('function_name')}("
                            f"{json.dumps(arguments, sort_keys=True)})")
    if commands:
        lines.append("[COMMANDS]\n" + "\n".join(commands))
    results = (step.get("observation") or {}).get("results") or ()
    output = "\n".join(str(result.get("content", "")) for result in results)
    output = output.replace("New Terminal Output:\n", "").strip()
    if output:
        lines.append("[OUTPUT]\n" + _cut_middle(output, WRENCH_OUTPUT_CHARS))
    return "\n".join(lines)


def _wrench_index() -> list[dict]:
    """Read the hack and baseline index rows at the pinned revision."""
    from huggingface_hub import hf_hub_download

    rows = []
    for split, mode in WRENCH_SPLITS.items():
        path = hf_hub_download(
            WRENCH_INDEX_REPO, f"data/{split}/train-00000-of-00001.parquet",
            repo_type="dataset", revision=SOURCE_REVISIONS[WRENCH_INDEX_REPO])
        table = pq.read_table(path, columns=list(WRENCH_INDEX_COLUMNS))
        rows.extend({**row, "mode": mode} for row in table.to_pylist())
    return rows


def _wrench_tasks(index: list[dict], n: int) -> list[str]:
    """The first n task ids in seeded order."""
    tasks = sorted({row["task_id"] for row in index})
    order = np.random.default_rng(DATA_SEED).permutation(len(tasks))
    return [tasks[i] for i in order[:n]]


def _wrench_rows(index, tasks, read_steps, tokenizer):
    """Build the run and step rows of the chosen tasks.

    Runs come in task order, then in index order within a task. A run's
    transcript is the task text, then every agent step; a step's text is
    the task text, then that step. The harness instructions in the first
    user message are the same for every run and are left out.

    Args:
        index: Index rows with task_id, model, trajectory_path,
            instruction, and mode.
        tasks: The chosen task ids, in order.
        read_steps: Returns a trajectory's steps from its path.
        tokenizer: Counts a transcript's tokens with `encode`.
    """
    by_task = {}
    for row in index:
        by_task.setdefault(row["task_id"], []).append(row)
    runs, steps = [], []
    for task_id in tasks:
        for row in by_task.get(task_id, ()):
            task = f"[TASK]\n{row['instruction'].strip()}"
            blocks = [_wrench_step(step)
                      for step in read_steps(row["trajectory_path"])[1:]
                      if step.get("source") == "agent"]
            transcript = "\n\n".join([task, *blocks])
            token_count = len(tokenizer.encode(
                transcript, add_special_tokens=False))
            if token_count > WRENCH_MAX_TOKENS:
                continue
            run_id = f"wr{len(runs):05d}"
            runs.append({
                "id": run_id, "task_id": task_id, "model": row["model"],
                "mode": row["mode"], "transcript": transcript,
                "step_count": len(blocks), "token_count": token_count,
            })
            steps.extend({
                "id": f"{run_id}/{index_}", "run_id": run_id,
                "step_index": index_, "model": row["model"],
                "text": f"{task}\n\n[STEP]\n{text}",
            } for index_, text in enumerate(blocks, start=1))
    return runs, steps


def _n_sales_deals(domain, sf):
    total = SALES_DEALS[domain]
    return min(total, max(8, round(total * sf)))


SALES_CALL_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("domain", pa.string()),
    ("deal_id", pa.string()),
    ("call_index", pa.int32()),
    ("prev_call_id", pa.string()),
    ("deal_stage", pa.string()),
    ("deal_amount", pa.float64()),
    ("transcript", pa.string()),
])

_SALES_CALL_SQL = """
SELECT v.OpportunityId__c, o.StageName, o.Amount, v.CreatedDate, v.Id,
       v.Body__c
FROM VoiceCallTranscript__c v JOIN Opportunity o ON o.Id = v.OpportunityId__c
"""


def _sales_source_calls(domain) -> list[tuple]:
    """Read one company's deal calls from its CRM database."""
    import sqlite3

    path = _download(CRMARENA_URL.format(commit=CRMARENA_COMMIT,
                                         domain=domain))
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return connection.execute(_SALES_CALL_SQL).fetchall()
    finally:
        connection.close()


def _sales_rows(calls_by_domain: dict, sf) -> list[dict]:
    """Build the sales call rows of the sampled deals.

    Deals come in seeded order within each company, B2B first; a deal's
    calls come in time order, and each names the deal's previous call.

    Args:
        calls_by_domain: Per company, (deal id, stage, amount, created
            time, call id, transcript) tuples.
        sf: The scale factor.

    Raises:
        ValueError: A company has other than SALES_DEALS deals.
    """
    rows = []
    for domain in SALES_DOMAINS:
        deals = {}
        for deal, stage, amount, created, call, body in calls_by_domain[domain]:
            deals.setdefault(deal, []).append(
                (created, call, stage, float(amount), body))
        if len(deals) != SALES_DEALS[domain]:
            raise ValueError(f"{domain} has {len(deals)} deals with calls, "
                             f"expected {SALES_DEALS[domain]}")
        keys = sorted(deals)
        order = np.random.default_rng(DATA_SEED).permutation(len(keys))
        for index in order[:_n_sales_deals(domain, sf)]:
            deal_id = f"sd{domain}{index:04d}"
            prev = None
            for number, (_, _, stage, amount, body) in enumerate(
                    sorted(deals[keys[index]]), start=1):
                call_id = f"sc{len(rows):05d}"
                rows.append({
                    "id": call_id, "domain": domain, "deal_id": deal_id,
                    "call_index": number, "prev_call_id": prev,
                    "deal_stage": stage, "deal_amount": amount,
                    "transcript": body,
                })
                prev = call_id
    return rows


# ------------------------------------------------------- set builders

def _imdb_pool():
    from huggingface_hub import hf_hub_download
    texts = []
    for split in ("train", "test"):
        f = hf_hub_download(
            "stanfordnlp/imdb",
            f"plain_text/{split}-00000-of-00001.parquet",
            repo_type="dataset",
            revision=SOURCE_REVISIONS["stanfordnlp/imdb"])
        texts += pq.read_table(f, columns=["text"]).column(
            "text").to_pylist()
    rng = np.random.default_rng(DATA_SEED)
    rng.shuffle(texts)
    return texts


def _biodex_rows(n):
    """Real BioDEX rows as (text, reactions), unpadded, un-concatenated.

    `reactions` seeds the `terms` table.
    """
    from datasets import load_dataset
    ds = load_dataset(
        "BioDEX/BioDEX-Reactions", split="train", streaming=True,
        revision=SOURCE_REVISIONS["BioDEX/BioDEX-Reactions"])
    rows = []
    for row in ds:
        text = str(row.get("fulltext_processed") or row.get("abstract"))
        reactions = [t.strip() for t in
                     str(row.get("reactions", "")).split(",") if t.strip()]
        if len(text) >= 200 and reactions:
            rows.append((text, reactions))
        if len(rows) >= n:
            break
    return rows


def _fever_data(n_claims):
    """FEVER claims (SUPPORTS/REFUTES only) and their Wikipedia pages.

    The evidence pool is bounded by the sampled claims.
    """
    from huggingface_hub import hf_hub_download
    seen, claims = set(), []
    for split in ("v1.0/train/0000.parquet",
                  "v1.0/labelled_dev/0000.parquet"):
        f = hf_hub_download("fever/fever", split,
                            repo_type="dataset",
                            revision=SOURCE_REVISIONS["fever/fever"])
        rows = pq.read_table(f).to_pylist()
        for r in rows:
            if (r["id"] in seen
                    or r["label"] not in ("SUPPORTS", "REFUTES")
                    or not r["evidence_wiki_url"]):
                continue
            seen.add(r["id"])
            claims.append(r)
            if len(claims) >= n_claims:
                break
        if len(claims) >= n_claims:
            break
    pages_needed = {r["evidence_wiki_url"] for r in claims}
    page_text = {}
    for shard in range(10):
        if len(page_text) >= len(pages_needed):
            break
        fw = hf_hub_download(
            "fever/fever",
            f"wiki_pages/partial-wikipedia_pages/{shard:04d}.parquet",
            repo_type="dataset",
            revision=SOURCE_REVISIONS["fever/fever"])
        t = pq.read_table(fw, columns=["id", "text"])
        for pid, txt in zip(t.column("id").to_pylist(),
                            t.column("text").to_pylist()):
            if pid in pages_needed and pid not in page_text:
                page_text[pid] = txt
    return claims, page_text


def _lepard_pair_priority(dest_id, passage_id):
    value = f"{DATA_SEED}\0{dest_id}\0{passage_id}".encode()
    return int.from_bytes(
        hashlib.blake2b(value, digest_size=16).digest(), "big")


def _sample_lepard_pairs(rows, passages, n):
    """Select a stable random sample of distinct known citation pairs."""
    passages = {
        str(key): str(value).strip()
        for key, value in passages.items()
        if value
    }
    selected = []
    selected_rows = {}
    for dest_id, destination_context, passage_id in rows:
        dest_id = str(dest_id)
        passage_id = str(passage_id)
        key = (dest_id, passage_id)
        passage_text = passages.get(passage_id)
        context = str(destination_context).strip()
        if not passage_text or len(context) < 50:
            continue
        if key in selected_rows:
            prior_context, _ = selected_rows[key]
            if (len(context), context) > (len(prior_context), prior_context):
                selected_rows[key] = (context, passage_text)
            continue
        priority = _lepard_pair_priority(dest_id, passage_id)
        item = (-priority, dest_id, passage_id)
        if len(selected) < n:
            heapq.heappush(selected, item)
            selected_rows[key] = (context, passage_text)
            continue
        if priority >= -selected[0][0]:
            continue
        removed = heapq.heapreplace(selected, item)
        del selected_rows[(removed[1], removed[2])]
        selected_rows[key] = (context, passage_text)
    pairs = []
    for _, dest_id, passage_id in sorted(
            selected, key=lambda item: (-item[0], item[1], item[2])):
        context, passage_text = selected_rows[(dest_id, passage_id)]
        pairs.append((dest_id, passage_id, context, passage_text))
    return pairs


def _lepard_documents(pairs):
    """Deduplicate each document column after sampling citation pairs."""
    contexts = {}
    passages = {}
    for _dest_id, passage_id, context, passage_text in pairs:
        contexts.setdefault(context, set()).add(passage_id)
        passages.setdefault(passage_text, set()).add(passage_id)
    context_rows = [{
        "id": f"lc{i}",
        "destination_context": context,
        "cited_passage_ids": sorted(passage_ids),
    } for i, (context, passage_ids) in enumerate(contexts.items())]
    passage_rows = [{
        "id": f"lp{i}",
        "passage_text": passage_text,
        "passage_ids": sorted(passage_ids),
    } for i, (passage_text, passage_ids) in enumerate(passages.items())]
    return context_rows, passage_rows


def _lepard_rows(n):
    """Read LePaRD and sample known positive citation pairs."""
    import json as _json

    import pandas as pd
    from huggingface_hub import hf_hub_download

    csv_path = hf_hub_download("rmahari/LePaRD", "top_10000_data.csv.gz",
                               repo_type="dataset",
                               revision=SOURCE_REVISIONS["rmahari/LePaRD"])
    dict_path = hf_hub_download("rmahari/LePaRD", "passage_dict.json",
                                repo_type="dataset",
                                revision=SOURCE_REVISIONS["rmahari/LePaRD"])
    with open(dict_path) as source:
        passages = _json.load(source)["data"]

    cols = ["dest_id", "destination_context", "passage_id"]
    chunks = pd.read_csv(
        csv_path,
        usecols=cols,
        chunksize=50_000,
        dtype={
            "dest_id": "string",
            "destination_context": "string",
            "passage_id": "string",
        },
    )
    rows = (
        row
        for chunk in chunks
        for row in chunk.loc[:, cols].itertuples(index=False, name=None)
    )
    return _sample_lepard_pairs(rows, passages, n)


def _vocab_table(rows, idx, cap=None):
    """A frequency-sorted, deduplicated vocabulary column from one field.

    Built across sampled rows (the `terms` table, from `reactions`).
    """
    freq = {}
    for row in rows:
        for t in row[idx]:
            freq[t] = freq.get(t, 0) + 1
    vocab = [t for t, _ in sorted(freq.items(),
                                  key=lambda kv: (-kv[1], kv[0]))]
    return vocab[:cap] if cap else vocab


def _build_lepard(d, sf, force=False):
    """Build the two deduplicated LePaRD document tables."""
    context_path = d / "citation_contexts.parquet"
    passage_path = d / "citation_passages.parquet"
    if context_path.exists() and passage_path.exists() and not force:
        return
    expected_pairs = _n_lepard_pairs(sf)
    pairs = _lepard_rows(expected_pairs)
    if len(pairs) != expected_pairs:
        raise ValueError(
            f"LePaRD provided {len(pairs)} valid citation pairs, expected "
            f"{expected_pairs}")
    contexts, passages = _lepard_documents(pairs)
    context_schema = pa.schema([
        ("id", pa.string()),
        ("destination_context", pa.string()),
        ("cited_passage_ids", pa.list_(pa.string())),
    ])
    passage_schema = pa.schema([
        ("id", pa.string()),
        ("passage_text", pa.string()),
        ("passage_ids", pa.list_(pa.string())),
    ])
    pq.write_table(
        pa.Table.from_pylist(contexts, schema=context_schema), context_path)
    pq.write_table(
        pa.Table.from_pylist(passages, schema=passage_schema), passage_path)


def _build_policies(d, sf, force=False):
    """Build policies.parquet and scenarios.parquet, idempotent.

    The PrivacyPolicies corpus is ~1M documents and 48 GiB, so it may
    not be available on every machine. Skips silently when the source
    dataset is not installed.
    """
    pol_path = d / "policies.parquet"
    scen_path = d / "scenarios.parquet"
    if pol_path.exists() and scen_path.exists() and not force:
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return
    n = _n_docs("policies", sf)
    try:
        f = hf_hub_download(
            "mukund/PrivacyPolicies",
            "data/train-00000-of-00001.parquet",
            repo_type="dataset",
            revision=SOURCE_REVISIONS["mukund/PrivacyPolicies"])
    except Exception:
        return
    t = pq.read_table(f, columns=["text"])
    texts = t.column("text").to_pylist()
    rng = np.random.default_rng(DATA_SEED)
    rng.shuffle(texts)
    texts = texts[:n]
    pq.write_table(pa.table({
        "id": [f"pp{i}" for i in range(len(texts))],
        "policy_text": texts,
    }), pol_path)
    pq.write_table(pa.table({
        "id": [f"sc{i}" for i in range(len(SCENARIOS))],
        "scenario": SCENARIOS,
    }), scen_path)


def _build_agent_traces(d, sf, force=False):
    """Build the SWE-Next cumulative trace snapshots."""
    path = d / "agent_traces.parquet"
    if path.exists() and not force:
        return
    n = _n_agent_documents(sf)
    rows = _agent_rows(n, full=n == AGENT_TRACE_DOCUMENTS)
    schema = pa.schema([
        ("id", pa.string()),
        ("trace", pa.string()),
        ("trajectory_id", pa.string()),
        ("turn_index", pa.int32()),
        ("token_count", pa.int32()),
    ])
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        compression="zstd",
        use_dictionary=False,
    )


def _build_support(d, sf, force=False):
    """Build the tau-bench support traces and their messages."""
    trace_path = d / "support_traces.parquet"
    message_path = d / "support_messages.parquet"
    if trace_path.exists() and message_path.exists() and not force:
        return
    traces, messages = _support_rows(_support_source_tasks(),
                                     _n_support_tasks(sf))
    pq.write_table(pa.Table.from_pylist(traces, schema=SUPPORT_TRACE_SCHEMA),
                   trace_path, compression="zstd", use_dictionary=False)
    pq.write_table(pa.Table.from_pylist(messages, schema=MESSAGE_SCHEMA),
                   message_path, compression="zstd", use_dictionary=False)


def _build_issue_runs(d, sf, force=False):
    """Build the SWE-rebench issue runs and their messages."""
    run_path = d / "issue_runs.parquet"
    message_path = d / "issue_messages.parquet"
    if run_path.exists() and message_path.exists() and not force:
        return
    from transformers import AutoTokenizer

    n = _n_issue_run_issues(sf)
    source = _issue_run_source()
    candidates = _issue_run_candidates(
        source.read(columns=["instance_id"]).column("instance_id").to_pylist(),
        n)
    tokenizer = AutoTokenizer.from_pretrained(
        AGENT_TRACE_TOKENIZER, revision=AGENT_TRACE_TOKENIZER_REVISION)
    batches = source.iter_batches(
        batch_size=64,
        columns=["instance_id", "repo", "trajectory", "model_patch",
                 "resolved"])
    runs, messages = _issue_run_rows(batches, candidates, n, tokenizer)
    pq.write_table(pa.Table.from_pylist(runs, schema=ISSUE_RUN_SCHEMA),
                   run_path, compression="zstd", use_dictionary=False)
    pq.write_table(pa.Table.from_pylist(messages, schema=MESSAGE_SCHEMA),
                   message_path, compression="zstd", use_dictionary=False)


def _build_wrench(d, sf, force=False):
    """Build the Terminal Wrench runs and their agent steps."""
    run_path = d / "wrench_runs.parquet"
    step_path = d / "wrench_steps.parquet"
    if run_path.exists() and step_path.exists() and not force:
        return
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    index = _wrench_index()
    tasks = _wrench_tasks(index, _n_wrench_tasks(sf))
    chosen = set(tasks)
    paths = [row["trajectory_path"] for row in index
             if row["task_id"] in chosen]
    root = Path(snapshot_download(
        WRENCH_REPO, repo_type="dataset", allow_patterns=paths,
        revision=SOURCE_REVISIONS[WRENCH_REPO]))
    tokenizer = AutoTokenizer.from_pretrained(
        AGENT_TRACE_TOKENIZER, revision=AGENT_TRACE_TOKENIZER_REVISION)
    runs, steps = _wrench_rows(
        index, tasks,
        lambda path: json.loads((root / path).read_text())["steps"],
        tokenizer)
    pq.write_table(pa.Table.from_pylist(runs, schema=WRENCH_RUN_SCHEMA),
                   run_path, compression="zstd", use_dictionary=False)
    pq.write_table(pa.Table.from_pylist(steps, schema=WRENCH_STEP_SCHEMA),
                   step_path, compression="zstd", use_dictionary=False)


def _build_sales(d, sf, force=False):
    """Build the CRMArena-Pro sales calls."""
    path = d / "sales_calls.parquet"
    if path.exists() and not force:
        return
    rows = _sales_rows({domain: _sales_source_calls(domain)
                        for domain in SALES_DOMAINS}, sf)
    pq.write_table(pa.Table.from_pylist(rows, schema=SALES_CALL_SCHEMA),
                   path, compression="zstd", use_dictionary=False)


# ------------------------------------------------ CUAD contracts

CONTRACT_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("title", pa.string()),
    ("text", pa.string()),
    ("token_count", pa.int32()),
])
CLAUSE_TYPE_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("name", pa.string()),
    ("question", pa.string()),
])


def _n_contracts(sf):
    return min(CUAD_CONTRACTS, max(8, round(CUAD_CONTRACTS * sf)))


def cuad_source() -> list[dict]:
    """Read the CUAD contracts at the pinned revision, in title order."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CUAD_REPO, CUAD_JSON, repo_type="dataset",
                           revision=SOURCE_REVISIONS[CUAD_REPO])
    with open(path, encoding="utf-8") as stream:
        contracts = json.load(stream)["data"]
    return sorted(contracts, key=lambda contract: contract["title"])


def cuad_title_key(title: str) -> str:
    """Return a contract title as the CUAD CSV and JSON both spell it.

    The CSV keeps an ampersand or apostrophe the JSON writes as an
    underscore, and a few CSV names end in a space or a dash.
    """
    return re.sub(r"[&']", "_", title).rstrip(" -")


def cuad_categories(contracts) -> list[tuple[str, str]]:
    """Return (name, question) per category, in the dataset's order.

    A question is the labelers' instruction after "Details:", as
    written; the name is the quoted category in the same instruction.
    """
    out = []
    for qa in contracts[0]["paragraphs"][0]["qas"]:
        name = qa["question"].split('"')[1]
        question = qa["question"].split("Details:", 1)[1].strip()
        out.append((name, question))
    return out


def _contract_rows(contracts, tokenizer) -> list[dict]:
    """Return the contracts within the token cap, numbered in title order."""
    rows = []
    for contract in contracts:
        (paragraph,) = contract["paragraphs"]
        text = paragraph["context"]
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count > CUAD_MAX_TOKENS:
            continue
        rows.append({"id": f"ct{len(rows)}", "title": contract["title"],
                     "text": text, "token_count": token_count})
    return rows


def _build_cuad(d, sf, force=False):
    """Build the CUAD contracts and clause types."""
    contract_path = d / "contracts.parquet"
    type_path = d / "clause_types.parquet"
    if contract_path.exists() and type_path.exists() and not force:
        return
    from transformers import AutoTokenizer

    contracts = cuad_source()
    tokenizer = AutoTokenizer.from_pretrained(
        AGENT_TRACE_TOKENIZER, revision=AGENT_TRACE_TOKENIZER_REVISION)
    rows = _contract_rows(contracts, tokenizer)
    if len(rows) != CUAD_CONTRACTS:
        raise ValueError(
            f"{len(rows)} contracts fit the cap, expected {CUAD_CONTRACTS}")
    order = np.random.default_rng(DATA_SEED).permutation(len(rows))
    kept = sorted(order[:_n_contracts(sf)].tolist())
    pq.write_table(
        pa.Table.from_pylist([rows[i] for i in kept], schema=CONTRACT_SCHEMA),
        contract_path, compression="zstd", use_dictionary=False)
    types = [{"id": f"ty{i}", "name": name, "question": question}
             for i, (name, question) in enumerate(
                 (name, question) for name, question in cuad_categories(contracts)
                 if name not in CUAD_FIELD_CATEGORIES)]
    pq.write_table(pa.Table.from_pylist(types, schema=CLAUSE_TYPE_SCHEMA),
                   type_path, compression="zstd", use_dictionary=False)


def _fetch_published_corpus(d, sf, root=None) -> bool:
    """Download the labeled corpus for this scale factor into d.

    The download is checked against the published corpus id, using the
    tables, columns, and source revisions its manifest records. Tables
    added to the benchmark after that corpus was published are not in
    it; the caller builds them from their sources.

    Returns False when no corpus is published for sf, the bucket is
    unreachable, or the downloaded tables do not hash to the published
    corpus id; the caller then builds from the sources.
    """
    corpus_id = PUBLISHED_CORPORA.get(sf)
    if corpus_id is None:
        return False
    prefix = f"{GROUND_TRUTH_ROOT}/corpora/{corpus_id}"
    try:
        paths = [path for path in _list_files(root, prefix)
                 if path.endswith(".parquet")]
    except OSError as error:
        print(f"[data] cannot reach the published corpus: {error}", flush=True)
        return False
    if not paths:
        return False
    d.mkdir(parents=True, exist_ok=True)
    for path in paths:
        (d / Path(path).name).write_bytes(_read_bytes(root, path))
    manifest = json.loads(_read_bytes(root, f"{prefix}/manifest.json"))
    try:
        published = {
            table: pq.read_table(d / f"{table}.parquet", columns=list(columns))
            for table, columns in manifest["columns"].items()}
        identity = corpus_identity(published, sf, manifest["data_seed"],
                                   manifest["source_revisions"])
    except (OSError, KeyError, pa.ArrowInvalid) as error:
        print(f"[data] cannot read the published corpus: {error}", flush=True)
        identity = {"corpus_id": None}
    if identity["corpus_id"] != corpus_id:
        for path in paths:
            (d / Path(path).name).unlink()
        print(f"[data] published corpus {corpus_id} does not match this "
              "code; building from the sources", flush=True)
        return False
    print(f"[data] fetched corpus {corpus_id} from the public bucket",
          flush=True)
    return True


def build_sets(data_dir, sf, lf=1, fetch=True):
    """Build the benchmark tables as Parquet files, cached by sf.

    With `fetch`, a published corpus for this scale factor is
    downloaded from the public bucket instead of being rebuilt from
    the sources. lf (load factor) is accepted but unused: documents
    here are real and unpadded, so there's nothing to scale. Kept in
    the signature so callers don't have to change when it's wired
    back up.
    """
    del lf
    d = Path(data_dir) / f"sf{sf}"
    marker = d / "DONE"
    expected = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "data_seed": DATA_SEED,
        "lepard_positive_pairs": LEPARD_POSITIVE_PAIRS,
        "scale_factor": sf,
        "source_revisions": SOURCE_REVISIONS,
    }
    if marker.exists():
        try:
            current = json.loads(marker.read_text())
        except json.JSONDecodeError:
            current = None
        if current == expected:
            _build_lepard(d, sf)
            _build_agent_traces(d, sf)
            _build_support(d, sf)
            _build_issue_runs(d, sf)
            _build_wrench(d, sf)
            _build_sales(d, sf)
            _build_cuad(d, sf)
            return d
        rebuilt_sources = (
            "TIGER-Lab/SWE-Next-SFT-Trajectories",
            "sierra-research/tau-bench",
            "nebius/SWE-rebench-openhands-trajectories",
            WRENCH_REPO,
            WRENCH_INDEX_REPO,
            "SalesforceAIResearch/CRMArena",
            CUAD_REPO,
        )
        base_sources = {
            name: revision for name, revision in SOURCE_REVISIONS.items()
            if name not in rebuilt_sources
        }
        same_sources = (
            current
            and current.get("data_seed") == DATA_SEED
            and current.get("scale_factor") == sf
            and all(
                current.get("source_revisions", {}).get(name) == revision
                for name, revision in base_sources.items()
            )
        )
        other_tables = (
            "reviews", "aspects", "reports", "terms", "claims", "evidence"
        )
        if same_sources and all((d / f"{name}.parquet").exists()
                                for name in other_tables):
            _build_lepard(d, sf, force=True)
            _build_agent_traces(d, sf, force=True)
            _build_support(d, sf, force=True)
            _build_issue_runs(d, sf, force=True)
            _build_wrench(d, sf, force=True)
            _build_sales(d, sf, force=True)
            _build_cuad(d, sf, force=True)
            marker.write_text(json.dumps(expected, indent=2, sort_keys=True))
            return d
    if fetch and _fetch_published_corpus(d, sf):
        # tables newer than the published corpus are built from their sources
        _build_support(d, sf)
        _build_issue_runs(d, sf)
        _build_wrench(d, sf)
        _build_sales(d, sf)
        _build_cuad(d, sf)
        marker.write_text(json.dumps(expected, indent=2, sort_keys=True))
        return d
    d.mkdir(parents=True, exist_ok=True)

    def write(name, ids, col_name, values):
        pq.write_table(pa.table({"id": ids, col_name: values}),
                       d / f"{name}.parquet")

    # reviews: real IMDB text, one row = one review, unpadded
    n = _n_docs("reviews", sf)
    imdb = _imdb_pool()
    write("reviews", [f"rv{i}" for i in range(n)], "body", imdb[:n])
    write("aspects", [f"as{i}" for i in range(len(ASPECTS))],
          "aspect", ASPECTS)

    # reports: real BioDEX text, one row = one report, unpadded
    n = _n_docs("reports", sf)
    bio = _biodex_rows(n)
    pq.write_table(pa.table({
        "id": [f"rp{i}" for i in range(len(bio))],
        "report": [t for t, _ in bio],
        "reactions": [r for _, r in bio],
    }), d / "reports.parquet")
    terms = _vocab_table(bio, 1)
    write("terms", [f"tm{i}" for i in range(len(terms))], "term", terms)
    # claims + evidence: real FEVER claims and only the Wikipedia
    # pages those claims reference
    n = _n_docs("claims", sf)
    claims, page_text = _fever_data(n)
    pq.write_table(pa.table({
        "id": [f"cl{i}" for i in range(len(claims))],
        "claim": [c["claim"] for c in claims],
        "label": [c["label"] for c in claims],
        "evidence_wiki_url": [c["evidence_wiki_url"] for c in claims],
    }), d / "claims.parquet")
    ev_ids = list(page_text.keys())
    pq.write_table(pa.table({
        "id": ev_ids,
        "text": [page_text[p] for p in ev_ids],
    }), d / "evidence.parquet")

    _build_lepard(d, sf, force=True)
    _build_agent_traces(d, sf, force=True)
    _build_support(d, sf, force=True)
    _build_issue_runs(d, sf, force=True)
    _build_wrench(d, sf, force=True)
    _build_sales(d, sf, force=True)
    _build_cuad(d, sf, force=True)

    marker.write_text(json.dumps(expected, indent=2, sort_keys=True))
    return d


# ------------------------------------------------ corpus identity

# The columns of every table that take part in the corpus identity.
# The labeling pass hashes the same columns.
CORPUS_COLUMNS = {
    "reviews": ("id", "body"),
    "aspects": ("id", "aspect"),
    "reports": ("id", "report", "reactions"),
    "terms": ("id", "term"),
    "claims": ("id", "claim", "label", "evidence_wiki_url"),
    "evidence": ("id", "text"),
    "citation_contexts": ("id", "destination_context",
                          "cited_passage_ids"),
    "citation_passages": ("id", "passage_text", "passage_ids"),
    "agent_traces": ("id", "trace", "trajectory_id", "turn_index",
                     "token_count"),
    "support_traces": tuple(SUPPORT_TRACE_SCHEMA.names),
    "support_messages": tuple(MESSAGE_SCHEMA.names),
    "issue_runs": tuple(ISSUE_RUN_SCHEMA.names),
    "issue_messages": tuple(MESSAGE_SCHEMA.names),
    "wrench_runs": tuple(WRENCH_RUN_SCHEMA.names),
    "wrench_steps": tuple(WRENCH_STEP_SCHEMA.names),
    "sales_calls": tuple(SALES_CALL_SCHEMA.names),
    "contracts": tuple(CONTRACT_SCHEMA.names),
    "clause_types": tuple(CLAUSE_TYPE_SCHEMA.names),
}


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _full_hash(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _python_rows(rows) -> list[dict]:
    return rows.to_pylist() if isinstance(rows, pa.Table) else rows


def _row_id(rows, index: int):
    if isinstance(rows, pa.Table):
        return rows.column("id")[index].as_py()
    return rows[index]["id"]


def _ids(rows):
    if isinstance(rows, pa.Table):
        return rows.column("id").to_pylist()
    return [row["id"] for row in rows]


def corpus_identity(rows: dict[str, pa.Table | list[dict]],
                    scale_factor: float,
                    data_seed: int, source_revisions: dict) -> dict:
    tables = {}
    for table in sorted(rows):
        row_hashes = [_full_hash(row) for row in _python_rows(rows[table])]
        tables[table] = {
            "rows": len(row_hashes),
            "ordered_rows_full_hash": _full_hash(row_hashes),
        }
    return corpus_identity_of_tables(tables, scale_factor, data_seed,
                                     source_revisions)


def corpus_identity_of_tables(tables: dict[str, dict], scale_factor: float,
                              data_seed: int, source_revisions: dict) -> dict:
    """Return the corpus identity of tables already hashed.

    Args:
        tables: Each table's `rows` and `ordered_rows_full_hash`, by name.
        scale_factor: The corpus scale factor.
        data_seed: The seed the tables were sampled with.
        source_revisions: The pinned source revision of each dataset.
    """
    tables = dict(sorted(tables.items()))
    payload = {
        "schema_version": 1,
        "benchmark": "quailb",
        "scale_factor": scale_factor,
        "data_seed": data_seed,
        "source_revisions": source_revisions,
        "tables": tables,
    }
    full = _full_hash(payload)
    return {
        **payload,
        "corpus_id": f"c_{full[:32]}",
        "corpus_full_hash": full,
    }


def read_corpus(data_dir: str | Path) -> dict[str, pa.Table]:
    data_dir = Path(data_dir)
    return {
        table: pq.read_table(
            data_dir / f"{table}.parquet", columns=list(columns))
        for table, columns in CORPUS_COLUMNS.items()
    }


def load_table(name: str, *, scale_factor: float = 0.1,
               limit: int | None = None, root=None) -> pa.Table:
    """Load one published benchmark table.

    Args:
        name: Document set name, such as "reviews".
        scale_factor: Published corpus scale factor.
        limit: Maximum rows to return, or None for the full table.
        root: Local directory or S3 URI with the published layout.
            Defaults to the public benchmark bucket.

    Returns:
        An Arrow table with the original benchmark ids and columns.

    Raises:
        ValueError: The table, scale factor, or row limit is invalid.
    """
    if name not in CORPUS_COLUMNS:
        raise ValueError(f"unknown benchmark table {name!r}")
    if scale_factor not in PUBLISHED_CORPORA:
        raise ValueError(f"no published corpus for scale factor {scale_factor}")
    if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError("limit must be a nonnegative integer or None")
    corpus = PUBLISHED_CORPORA[scale_factor]
    path = f"{GROUND_TRUTH_ROOT}/corpora/{corpus}/{name}.parquet"
    cached = _cached_file(root, path)
    filesystem, _, source = _location(root, path)
    table = (pq.read_table(cached) if cached is not None else
             pq.read_table(source, filesystem=filesystem))
    return table if limit is None else table.slice(0, limit)


# Local inputs stay separate from the original published corpus identities.
TRIPADVISOR_COLUMNS = {
    name: ("id", "hotel_id", "body") for name in (
        "tripadvisor_reviews", "tripadvisor_earlier", "tripadvisor_later")
}
TRIPADVISOR_SOURCE_URL = "https://www.cs.cmu.edu/~jiweil/review.txt.zip"
TRIPADVISOR_SOURCE_SHA256 = (
    "a43302098170a6c0c086970077dc8a2fce466d431b3c9d9d80ef9565bb057d8d"
)
TRIPADVISOR_SCHEMA_VERSION = 1
TRIPADVISOR_MONTHS = {
    name: index for index, name in enumerate((
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ), 1)
}


def _tripadvisor_settings(hotel_limit, reviews_per_hotel, seed, max_chars):
    for name, value, minimum in (("hotel_limit", hotel_limit, 1),
                                 ("reviews_per_hotel", reviews_per_hotel, 2)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if max_chars is not None and (
            isinstance(max_chars, bool) or not isinstance(max_chars, int)
            or max_chars < 1):
        raise ValueError("max_chars must be a positive integer or None")
    settings = {
        "hotel_limit": hotel_limit, "reviews_per_hotel": reviews_per_hotel,
        "seed": seed, "max_chars": max_chars,
        "sampling": "seeded-hotel-and-cohort-hash-v1",
        "partition": "strict-median-date-v1", "text_format": "title-text-v1",
    }
    return settings


def _unique_pairs(count):
    """Return the unordered pairs of distinct items among count items."""
    return count * (count - 1) // 2


def _tripadvisor_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("TripAdvisor IDs must be positive decimal integers")
    text = str(value)
    if re.fullmatch(r"[1-9][0-9]*", text) is None:
        raise ValueError("TripAdvisor IDs must be canonical positive decimals")
    return text


@lru_cache(maxsize=40_000)
def _tripadvisor_date(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    # The release uses English months regardless of the host's LC_TIME.
    match = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})", value)
    if match is not None:
        month = TRIPADVISOR_MONTHS.get(match[1].lower())
        if month is None:
            return None
        value = f"{match[3]}-{month:02d}-{int(match[2]):02d}"
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def _tripadvisor_row(row):
    if not isinstance(row, dict):
        raise ValueError("TripAdvisor source rows must be dictionaries")
    review_id = _tripadvisor_id(row.get("id"))
    hotel_id = _tripadvisor_id(row.get("offering_id"))
    text = row.get("text")
    date = row.get("date")
    posted = _tripadvisor_date(date) if isinstance(date, str) else None
    if not isinstance(text, str) or not text.strip() or posted is None:
        return None
    title = row.get("title") or ""
    if not isinstance(title, str):
        raise ValueError("TripAdvisor titles must be strings")
    body = "\n\n".join(part.strip() for part in (title, text) if part.strip())
    return {"id": review_id, "hotel_id": hotel_id, "body": body,
            "post_date": posted, "source_review": row}


def _tripadvisor_archive_rows(path):
    with ZipFile(path) as archive:
        if archive.namelist().count("review.txt") != 1:
            raise ValueError("TripAdvisor ZIP requires one review.txt member")
        with archive.open("review.txt") as stream:
            for line in io.TextIOWrapper(stream, encoding="utf-8"):
                if line.strip():
                    yield json.loads(line)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tripadvisor_table_identity(table):
    return {
        "columns": table.column_names, "rows": table.num_rows,
        "ordered_rows_full_hash": _full_hash(
            [_full_hash(row) for row in table.to_pylist()]),
    }


def _tripadvisor_manifest_identity(payload):
    full_hash = _full_hash(payload)
    return {**payload, "corpus_id": f"c_{full_hash[:32]}",
            "corpus_full_hash": full_hash}


def _tripadvisor_cutoff(dates, earlier_count, later_count):
    total, cumulative, choices = sum(dates.values()), 0, []
    for date, count in sorted(dates.items()):
        cumulative += count
        if cumulative >= earlier_count and total - cumulative >= later_count:
            choices.append((abs(total - 2 * cumulative), date))
    return min(choices)[1] if choices else None


def build_tripadvisor(output_dir, *, archive_path=None, rows=None, hotel_limit=5,
                      reviews_per_hotel=20, seed=0, max_chars=None,
                      source_sha256=None):
    """Prepare a stable hotel sample and strictly dated review cohorts.

    Args:
        output_dir: Empty destination for Parquet tables and provenance.
        archive_path: Local original JSON-lines ZIP, or None for fixture rows.
        rows: Raw source dictionaries, mutually exclusive with archive_path.
        hotel_limit: Number of eligible hotels to select.
        reviews_per_hotel: Reviews per hotel, at least two.
        seed: Seed for stable hotel and within-cohort hash ordering.
        max_chars: Explicit body limit, or None to retain full text.
        source_sha256: Expected archive or canonical fixture SHA256, if given.

    Returns:
        The content-addressed preparation manifest.
    """
    settings = _tripadvisor_settings(
        hotel_limit, reviews_per_hotel, seed, max_chars)
    if (archive_path is None) == (rows is None):
        raise ValueError("provide exactly one of archive_path or rows")
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"TripAdvisor directory is not empty: {directory}")
    if archive_path is not None:
        source_hash = _file_sha256(archive_path)
        expected = (TRIPADVISOR_SOURCE_SHA256 if source_sha256 is None
                    else source_sha256)
        if expected is None:
            raise ValueError("the original archive requires a source SHA256 pin")
        mode = "local_zip"
    else:
        provided = list(rows)
        source_hash = _full_hash(sorted(provided, key=_canonical))
        expected = source_hash if source_sha256 is None else source_sha256
        mode = "provided_rows"
    if (not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected)
            is None or source_hash != expected):
        raise ValueError("TripAdvisor source does not match its SHA256 pin")
    def factory():
        return (_tripadvisor_archive_rows(archive_path) if mode == "local_zip"
                else iter(provided))

    counts = Counter(source_reviews=0, excluded_text_or_date=0)
    dates = defaultdict(Counter)
    seen = set()
    for raw in factory():
        counts["source_reviews"] += 1
        row = _tripadvisor_row(raw)
        review_id = _tripadvisor_id(raw.get("id"))
        if review_id in seen:
            raise ValueError(f"duplicate TripAdvisor review ID {review_id}")
        seen.add(review_id)
        if row is None:
            counts["excluded_text_or_date"] += 1
        else:
            dates[row["hotel_id"]][row["post_date"]] += 1
    earlier_count = reviews_per_hotel // 2
    later_count = reviews_per_hotel - earlier_count
    cutoffs = {hotel: _tripadvisor_cutoff(values, earlier_count, later_count)
               for hotel, values in dates.items()}
    eligible = [hotel for hotel, cutoff in cutoffs.items() if cutoff is not None]
    ordered = sorted(eligible, key=lambda hotel: (_full_hash([seed, hotel]), hotel))
    if len(ordered) < hotel_limit:
        raise ValueError("not enough hotels with the required dated review cohorts")
    hotels = ordered[:hotel_limit]
    selected = {(hotel, cohort): [] for hotel in hotels
                for cohort in ("earlier", "later")}
    for raw in factory():
        hotel = _tripadvisor_id(raw.get("offering_id"))
        if hotel not in hotels:
            continue
        row = _tripadvisor_row(raw)
        if row is None:
            continue
        cohort = "earlier" if row["post_date"] <= cutoffs[hotel] else "later"
        limit = earlier_count if cohort == "earlier" else later_count
        rank = int(_full_hash([seed, hotel, row["id"]]), 16)
        item = (-rank, row["id"], row)
        heap = selected[(hotel, cohort)]
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
    records = {name: [] for name in TRIPADVISOR_COLUMNS}
    metadata = {"hotels": [{"hotel_id": hotel, "cutoff_date": cutoffs[hotel]}
                           for hotel in hotels], "reviews": []}
    truncated = 0
    for hotel in hotels:
        for cohort in ("earlier", "later"):
            sampled = [item[2] for item in selected[(hotel, cohort)]]
            sampled.sort(key=lambda row: (row["post_date"], row["id"]))
            for row in sampled:
                original = row["body"]
                body = original if max_chars is None else original[:max_chars]
                truncated += int(len(body) < len(original))
                document = {"id": row["id"], "hotel_id": hotel, "body": body}
                records["tripadvisor_reviews"].append(document)
                records[f"tripadvisor_{cohort}"].append(document)
                metadata["reviews"].append({
                    "id": row["id"], "hotel_id": hotel,
                    "post_date": row["post_date"], "cohort": cohort,
                    "original_chars": len(original),
                    "original_body_hash": _full_hash(original),
                    "source_review": row["source_review"],
                })
    counts.update(
        eligible_hotels=len(eligible), hotels=len(hotels),
        reviews=len(records["tripadvisor_reviews"]),
        earlier=len(records["tripadvisor_earlier"]),
        later=len(records["tripadvisor_later"]),
        same_hotel_pairs=len(hotels) * _unique_pairs(reviews_per_hotel),
        earlier_later_pairs=len(hotels) * earlier_count * later_count,
        truncated_reviews=truncated)
    tables = {name: pa.Table.from_pylist(records[name], schema=pa.schema(
        [(column, pa.string()) for column in columns]))
        for name, columns in TRIPADVISOR_COLUMNS.items()}
    manifest = _tripadvisor_manifest_identity({
        "schema_version": TRIPADVISOR_SCHEMA_VERSION,
        "benchmark": "quailb-tripadvisor", "dataset": "tripadvisor",
        "source": {"url": TRIPADVISOR_SOURCE_URL, "sha256": source_hash,
                   "mode": mode, "format": "tripadvisor-jsonl-zip-v1"},
        "settings": settings, "counts": dict(counts),
        "tables": {name: _tripadvisor_table_identity(table)
                   for name, table in tables.items()},
        "metadata": {"file": "metadata.json", "full_hash": _full_hash(metadata)},
    })
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        pq.write_table(table, directory / f"{name}.parquet", compression="zstd")
    for name, value in (("metadata.json", metadata), ("manifest.json", manifest)):
        (directory / name).write_text(json.dumps(value, indent=2, allow_nan=False))
    read_tripadvisor_manifest(directory)
    return manifest


def read_tripadvisor_manifest(path):
    """Validate the corpus hashes, source identities, and strictly dated cohorts."""
    path = Path(path)
    path = path / "manifest.json" if path.is_dir() else path
    manifest = json.loads(path.read_text())
    if (not isinstance(manifest, dict)
            or isinstance(manifest.get("schema_version"), bool)
            or manifest.get("schema_version") != TRIPADVISOR_SCHEMA_VERSION
            or manifest.get("benchmark") != "quailb-tripadvisor"
            or manifest.get("dataset") != "tripadvisor"):
        raise ValueError("unsupported TripAdvisor corpus manifest")
    payload = {key: value for key, value in manifest.items()
               if key not in ("corpus_id", "corpus_full_hash")}
    if _tripadvisor_manifest_identity(payload) != manifest:
        raise ValueError("TripAdvisor corpus identity does not match its manifest")
    for section, fields in (
            ("source", ("url", "sha256", "mode", "format")),
            ("settings", ("hotel_limit", "reviews_per_hotel", "seed", "max_chars")),
            ("counts", ("source_reviews", "excluded_text_or_date", "eligible_hotels")),
            ("tables", tuple(TRIPADVISOR_COLUMNS)),
            ("metadata", ("file", "full_hash"))):
        value = manifest.get(section)
        if not isinstance(value, dict) or not set(fields).issubset(value):
            raise ValueError(f"malformed TripAdvisor manifest section: {section}")
    source, settings = manifest["source"], manifest["settings"]
    expected = _tripadvisor_settings(
        settings["hotel_limit"], settings["reviews_per_hotel"],
        settings["seed"], settings["max_chars"])
    if (settings != expected or source.get("url") != TRIPADVISOR_SOURCE_URL
            or source.get("mode") not in ("local_zip", "provided_rows")
            or source.get("format") != "tripadvisor-jsonl-zip-v1"
            or not isinstance(source.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None):
        raise ValueError("unsupported TripAdvisor source or preparation settings")
    if set(manifest["tables"]) != set(TRIPADVISOR_COLUMNS):
        raise ValueError("TripAdvisor manifest lists other input tables")
    tables, indexed = {}, {}
    for name, columns in TRIPADVISOR_COLUMNS.items():
        table = pq.read_table(path.parent / f"{name}.parquet")
        schema = pa.schema([(column, pa.string()) for column in columns])
        if not table.schema.equals(schema, check_metadata=False):
            raise ValueError(f"{name} has the wrong TripAdvisor schema")
        if any(table[column].null_count for column in columns):
            raise ValueError(f"{name} contains null values")
        if _tripadvisor_table_identity(table) != manifest["tables"][name]:
            raise ValueError(f"{name} does not match its TripAdvisor table hash")
        rows = table.to_pylist()
        indexed[name] = {row["id"]: row for row in rows}
        if len(indexed[name]) != len(rows) or not rows:
            raise ValueError(f"{name} requires nonempty rows with unique IDs")
        for row in rows:
            _tripadvisor_id(row["id"])
            _tripadvisor_id(row["hotel_id"])
            if not row["body"].strip():
                raise ValueError(f"{name} contains an empty body")
        tables[name] = table
    info = manifest["metadata"]
    if info.get("file") != "metadata.json":
        raise ValueError("TripAdvisor metadata must use metadata.json")
    metadata = json.loads((path.parent / "metadata.json").read_text())
    if _full_hash(metadata) != info["full_hash"]:
        raise ValueError("TripAdvisor metadata does not match its hash")
    try:
        _validate_tripadvisor_cohorts(indexed, metadata, manifest)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("malformed TripAdvisor provenance metadata") from error
    return manifest, tables


def _validate_tripadvisor_cohorts(indexed, metadata, manifest):
    reviews = indexed["tripadvisor_reviews"]
    earlier, later = (indexed[f"tripadvisor_{name}"] for name in ("earlier", "later"))
    if set(earlier) & set(later) or set(earlier) | set(later) != set(reviews):
        raise ValueError("TripAdvisor cohorts must be disjoint and cover all reviews")
    for cohort in (earlier, later):
        if any(row != reviews[review_id] for review_id, row in cohort.items()):
            raise ValueError("TripAdvisor cohort text or hotel differs from reviews")
    hotels = {row["hotel_id"] for row in reviews.values()}
    provenance = metadata["reviews"]
    if (len(provenance) != len(reviews)
            or {row["id"] for row in provenance} != set(reviews)):
        raise ValueError("TripAdvisor provenance has different review IDs")
    boundaries = {row["hotel_id"]: row["cutoff_date"] for row in metadata["hotels"]}
    if len(boundaries) != len(metadata["hotels"]) or set(boundaries) != hotels:
        raise ValueError("TripAdvisor metadata has different hotel IDs")
    dates = defaultdict(lambda: {"earlier": [], "later": []})
    limit = manifest["settings"]["max_chars"]
    truncated = 0
    for row in provenance:
        original = _tripadvisor_row(row["source_review"])
        document = reviews[row["id"]]
        if original is None or (original["id"], original["hotel_id"]) != (
                row["id"], row["hotel_id"]):
            raise ValueError("TripAdvisor IDs differ from their source identities")
        body = original["body"]
        expected = body if limit is None else body[:limit]
        if (document["hotel_id"] != row["hotel_id"] or document["body"] != expected
                or row["original_chars"] != len(body)
                or row["original_body_hash"] != _full_hash(body)
                or row["post_date"] != original["post_date"]):
            raise ValueError("TripAdvisor document or date differs from its source")
        cohort = "earlier" if row["id"] in earlier else "later"
        if row["cohort"] != cohort:
            raise ValueError("TripAdvisor provenance has an incorrect cohort")
        cutoff = boundaries[row["hotel_id"]]
        if _tripadvisor_date(cutoff) != cutoff or (
                (cohort == "earlier") != (row["post_date"] <= cutoff)):
            raise ValueError("TripAdvisor cohort crosses its date boundary")
        dates[row["hotel_id"]][cohort].append(row["post_date"])
        truncated += int(len(expected) < len(body))
    settings = manifest["settings"]
    early_count = settings["reviews_per_hotel"] // 2
    late_count = settings["reviews_per_hotel"] - early_count
    for hotel, values in dates.items():
        if (len(values["earlier"]) != early_count
                or len(values["later"]) != late_count
                or max(values["earlier"]) >= min(values["later"])):
            raise ValueError(f"TripAdvisor hotel {hotel} has invalid dated cohorts")
    counts = manifest["counts"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in counts.values()):
        raise ValueError("TripAdvisor preparation counts must be nonnegative integers")
    actual = {"hotels": len(hotels), "reviews": len(reviews),
              "earlier": len(earlier), "later": len(later),
              "same_hotel_pairs": len(hotels) * _unique_pairs(
                  settings["reviews_per_hotel"]),
              "earlier_later_pairs": len(hotels) * early_count * late_count,
              "truncated_reviews": truncated}
    if any(counts.get(key) != value for key, value in actual.items()):
        raise ValueError("TripAdvisor preparation counts do not match the corpus")
    if (len(hotels) != settings["hotel_limit"]
            or counts["eligible_hotels"] < len(hotels)
            or counts["source_reviews"] - counts["excluded_text_or_date"]
            < len(reviews)):
        raise ValueError("TripAdvisor selection counts violate preparation settings")
