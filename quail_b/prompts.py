"""QUAIL-B prompt templates.

Filter templates take the document as `{0}`. Join templates take
one DOCUMENT marker per joined table as `{0}` and `{1}`.
"""

F1 = ("Judge strictly from the review above whether it mentions at "
      "least one positive aspect of the movie.\n\n{0}\n\nInstruction: "
      "answer TRUE if the review mentions at least one positive aspect "
      "of the movie, FALSE otherwise.")

F4 = ("Judge strictly from the review above whether it discusses the "
      "ending of the movie.\n\n{0}\n\nInstruction: answer TRUE if the "
      "review discusses the ending of the movie, FALSE otherwise.")

F5 = ("Judge strictly from the review above whether it mentions any "
      "specific actor or actress by name.\n\n{0}\n\nInstruction: "
      "answer TRUE if the review mentions a specific actor or actress "
      "by name, FALSE otherwise.")

DISCUSS_ASPECT = ("Does the review in DOCUMENT {0} discuss the movie "
                  "aspect in DOCUMENT {1}?")


# IMDB-8 only: a second question over the same aspects table, joined
# under a second alias (a2) - a 2-join star, both joins anchored on
# reviews so the second stage runs over whatever DISCUSS_ASPECT
# already kept.
ASPECT_SENTIMENT = ("Does the review in DOCUMENT {0} express positive "
                    "sentiment about the movie aspect in DOCUMENT {1}?")

SERIOUS_ADVERSE_EVENT = (
    "Judge strictly from the report above whether it describes a serious "
    "or life-threatening adverse event.\n\n{0}\n\nInstruction: answer TRUE "
    "if the report describes a serious or life-threatening adverse event, "
    "FALSE otherwise."
)

REACTION = ("Does the medical report in DOCUMENT {0} describe the "
            "reaction in DOCUMENT {1} as something the patient "
            "experienced?")

NEUROLOGICAL_REACTION = (
    "Is this reaction neurological, affecting the nervous system? {0}"
)

CARDIOVASCULAR_REACTION = (
    "Is this reaction cardiovascular, affecting the heart or blood vessels? {0}"
)

AGENT_RECOVERED = (
    "Judge strictly from the agent trace above whether the agent recovered "
    "after pursuing an approach that did not work. Recovery means the agent "
    "recognized or moved past the unsuccessful approach and then made useful "
    "progress with a different or corrected approach.\n\n{0}\n\n"
    "Instruction: answer TRUE if the trace shows the agent recovering after "
    "an unsuccessful approach, FALSE otherwise."
)

AGENT_IMPLEMENTED_FIX = (
    "Judge strictly from the agent trace above whether, by the end of the "
    "trace, the agent has implemented a plausible fix that directly addresses "
    "the reported issue. A fix must include a code or configuration change "
    "whose purpose is to correct the issue. Inspection, reproduction, tests "
    "without a fix, and unrelated edits do not count.\n\n{0}\n\nInstruction: "
    "answer TRUE if the agent has implemented a plausible fix that directly "
    "addresses the reported issue. Answer FALSE otherwise."
)

F11 = ("Judge strictly from the claim above whether it asserts "
       "something about a person, rather than an organization, place, "
       "or event.\n\n{0}\n\nInstruction: answer TRUE if the claim "
       "asserts something about a person, FALSE otherwise.")

F12 = ("Judge strictly from the claim above whether it contains a "
       "specific date or year.\n\n{0}\n\nInstruction: answer TRUE if "
       "the claim contains a specific date or year, FALSE otherwise.")

F14 = ("Judge strictly from the claim above whether it references a "
       "specific place (a city, country, or other named location).\n\n"
       "{0}\n\nInstruction: answer TRUE if the claim references a "
       "specific place, FALSE otherwise.")

SUPPORT = ("Does the Wikipedia passage in DOCUMENT {1} support the "
           "claim in DOCUMENT {0}?")

REFUTE = ("Does the Wikipedia passage in DOCUMENT {1} refute or "
          "contradict the claim in DOCUMENT {0}?")

F13 = ("Judge strictly from the Wikipedia passage above whether it "
       "primarily describes a specific person (their life, actions, "
       "or role), rather than an organization, place, or event.\n\n"
       "{0}\n\nInstruction: answer TRUE if the passage primarily "
       "describes a specific person, FALSE otherwise.")

# LePaRD predicates: "excerpt" for destination_context throughout,
# to avoid colliding with this dataset's own use of "passage" for
# the quoted/cited text.
LEP1 = ("Judge strictly from the excerpt above whether it argues that "
        "the cited case's reasoning does not apply here.\n\n{0}\n\n"
        "Instruction: answer TRUE if the excerpt argues the cited "
        "case's reasoning does not apply here, FALSE otherwise.")

LEP2 = ("Judge strictly from the excerpt above whether it discusses a "
        "procedural or jurisdictional issue.\n\n{0}\n\nInstruction: "
        "answer TRUE if the excerpt discusses a procedural or "
        "jurisdictional issue, FALSE otherwise.")

LEP3 = ("Judge strictly from the excerpt above whether it treats the "
        "cited passage as binding precedent.\n\n{0}\n\nInstruction: "
        "answer TRUE if the excerpt treats the cited passage as "
        "binding precedent, FALSE otherwise.")

LEP4 = ("Judge strictly from the excerpt above whether it cites the "
        "passage to support a conclusion about a party's liability or "
        "guilt.\n\n{0}\n\nInstruction: answer TRUE if the excerpt "
        "cites the passage to support a conclusion about a party's "
        "liability or guilt, FALSE otherwise.")

LEP5 = ("Judge strictly from the excerpt above whether it acknowledges "
        "disagreement between courts on the issue.\n\n{0}\n\n"
        "Instruction: answer TRUE if the excerpt acknowledges "
        "disagreement between courts on the issue, FALSE otherwise.")

# LEP-5 only: filters the passage side of the self-join, not just the
# excerpt (anchor) side.
LEPS1 = ("Judge strictly from the passage above whether it states a "
         "general legal rule.\n\n{0}\n\nInstruction: answer TRUE if "
         "the passage states a general legal rule, FALSE otherwise.")

# The LEP-2..LEP-5 join predicate. Ground truth comes from the
# dataset's own passage_id, not a judge pass.
LEPJOIN = ("Is the passage in DOCUMENT {1} cited by the legal excerpt "
           "in DOCUMENT {0}?")


# PrivacyPolicies predicates: user-language questions about user
# outcomes, not legal language about company practices. The vocabulary
# mismatch means keyword search, regex, and embeddings cannot solve
# these.
P_MSG = ("Judge strictly from the policy above whether, if a user sent "
         "a private message through this service, an employee of the "
         "company could read it.\n\n{0}\n\nInstruction: answer TRUE if "
         "an employee could read the user's private messages, FALSE "
         "otherwise.")

P_LOC = ("Judge strictly from the policy above whether this service "
         "would keep track of the user's physical location, even when "
         "the user is not actively using the service.\n\n{0}\n\n"
         "Instruction: answer TRUE if the service would track the "
         "user's location while they are not using it, FALSE otherwise.")

SCENARIO_MATCH = ("Based on the privacy policy in DOCUMENT {0}, could "
                  "the situation described in DOCUMENT {1} happen to a "
                  "user of this service?")



# Classification templates. Each call also carries its labels and
# optional descriptions, defined next to it. A template is used with
# exactly one label list, so a template names one predicate. A
# one-document template takes the document as `{0}`; a pair template
# takes the anchor as `{0}` and its partner as `{1}`.

IMDB_SENTIMENT = (
    "Judge strictly from the review above which category best describes "
    "the overall sentiment it expresses about the movie.\n\n{0}"
)
IMDB_SENTIMENT_LABELS = ("positive", "negative", "neutral", "mixed")

# A classification of joined rows: the review is DOCUMENT {0}, the
# aspect DOCUMENT {1}, and the label describes the two together.
IMDB_ASPECT_SENTIMENT = (
    "Judge strictly from the review in DOCUMENT {0} what sentiment it "
    "expresses about the movie aspect in DOCUMENT {1}."
)
IMDB_ASPECT_SENTIMENT_LABELS = IMDB_SENTIMENT_LABELS

# IMDb's own genre list.
IMDB_GENRE = (
    "Judge strictly from the review above which genre best describes the "
    "movie it reviews.\n\n{0}"
)
IMDB_GENRE_LABELS = (
    "action", "adventure", "animation", "biography", "comedy", "crime",
    "documentary", "drama", "family", "fantasy", "history", "horror",
    "music", "musical", "mystery", "romance", "sci-fi", "sport", "thriller",
    "war", "western",
)

IMDB_COMPLAINT = (
    "Judge strictly from the review above what it criticizes most about "
    "the movie.\n\n{0}"
)
IMDB_COMPLAINT_LABELS = (
    "poor acting", "poor writing", "poor visual effects", "slow pacing",
    "weak ending", "too long", "too violent", "no specific complaint",
)

# MedDRA's system organ classes, less "product issues", which a
# reaction term does not describe.
BIO_ORGAN_CLASS = (
    "Judge strictly from the reaction above which MedDRA system organ "
    "class it belongs to.\n\n{0}"
)
BIO_ORGAN_CLASS_LABELS = (
    "cardiac disorders", "vascular disorders", "nervous system disorders",
    "psychiatric disorders", "eye disorders", "ear and labyrinth disorders",
    "respiratory, thoracic and mediastinal disorders",
    "gastrointestinal disorders", "hepatobiliary disorders",
    "renal and urinary disorders", "skin and subcutaneous tissue disorders",
    "musculoskeletal and connective tissue disorders",
    "blood and lymphatic system disorders", "immune system disorders",
    "endocrine disorders", "metabolism and nutrition disorders",
    "infections and infestations",
    "neoplasms benign, malignant and unspecified",
    "injury, poisoning and procedural complications", "investigations",
    "general disorders and administration site conditions",
    "reproductive system and breast disorders",
    "pregnancy, puerperium and perinatal conditions",
    "congenital, familial and genetic disorders",
    "surgical and medical procedures", "social circumstances",
)

FEV_TOPIC = (
    "Judge strictly from the claim above which topic it is about.\n\n{0}"
)
FEV_TOPIC_LABELS = (
    "politics", "sports", "film and television", "music", "literature",
    "science", "history", "geography", "business", "religion", "other",
)

LEP_AREA = (
    "Judge strictly from the excerpt above which area of law it "
    "concerns.\n\n{0}"
)
LEP_AREA_LABELS = (
    "constitutional law", "criminal law", "civil procedure", "civil rights",
    "contracts", "torts", "property", "administrative law",
    "employment law", "intellectual property", "tax law", "family law",
    "immigration law", "bankruptcy", "antitrust",
)

# The traces are prefixes of agent runs, so these ask what the trace
# shows so far, not how the run ended
AGENT_PROGRESS = (
    "Judge strictly from the agent trace above how far the agent has "
    "gotten so far.\n\n{0}"
)
AGENT_PROGRESS_LABELS = (
    "has not located the relevant code", "located the relevant code",
    "changed the code, not checked", "changed the code, check passes",
    "changed the code, check fails",
)
AGENT_PROGRESS_DESCRIPTIONS = (
    "has not yet found the code that causes the issue",
    "found the code that causes the issue but has not changed it",
    "edited the code but has not run a test or reproduction since",
    "edited the code and the latest test or reproduction run succeeds",
    "edited the code and the latest test or reproduction run still fails",
)
AGENT_CHANGED_CODE = AGENT_PROGRESS_LABELS[2:]

AGENT_TEST_RESULT = (
    "Judge strictly from the agent trace above what the most recent test "
    "or reproduction run showed.\n\n{0}"
)
AGENT_TEST_RESULT_LABELS = (
    "bug fixed, tests pass", "bug still occurs",
    "bug fixed, other tests fail", "run errored",
)
AGENT_TEST_RESULT_DESCRIPTIONS = (
    "the reported bug no longer occurs and other tests pass",
    "the reported bug still happens",
    "the reported bug no longer occurs but other tests fail",
    "the run crashed or errored before testing anything",
)

# PyPI's "Topic ::" trove classifiers, https://pypi.org/classifiers/
AGENT_DOMAIN = (
    "Judge strictly from the agent trace above which PyPI topic best "
    "describes the project the issue is filed against.\n\n{0}"
)
AGENT_DOMAIN_LABELS = (
    "Communications", "Database", "Documentation", "File Formats",
    "Internet :: WWW/HTTP", "Multimedia :: Graphics",
    "Multimedia :: Sound/Audio", "Multimedia :: Video",
    "Office/Business :: Financial",
    "Scientific/Engineering :: Artificial Intelligence",
    "Scientific/Engineering :: Artificial Life",
    "Scientific/Engineering :: Bio-Informatics",
    "Scientific/Engineering :: GIS", "Scientific/Engineering :: Mathematics",
    "Scientific/Engineering :: Medical Science Apps.", "Security",
    "Software Development :: Build Tools", "Software Development :: Compilers",
    "Software Development :: Embedded Systems",
    "Software Development :: Quality Assurance",
    "Software Development :: Testing",
    "Software Development :: User Interfaces",
    "System :: Distributed Computing", "System :: Networking",
    "Text Processing :: Linguistic", "Text Processing :: Markup",
    "Utilities",
)

# IBM's Orthogonal Defect Classification defect types (Chillarege et al.,
# IEEE TSE 1992)
AGENT_ROOT_CAUSE = (
    "Judge strictly from the agent trace above what kind of defect causes "
    "the reported issue.\n\n{0}"
)
AGENT_ROOT_CAUSE_LABELS = (
    "assignment", "checking", "algorithm", "function", "interface",
    "timing or serialization", "build, package, or merge", "documentation",
)
AGENT_ROOT_CAUSE_DESCRIPTIONS = (
    "a value is set or initialized wrong",
    "a condition or validation is missing or wrong",
    "the steps that compute a result are wrong",
    "a needed function, class, or feature is missing",
    "components pass the wrong arguments, return values, or messages",
    "shared resources, concurrency, or the order of events are wrong",
    "a dependency, version, build, or packaging setting is wrong",
    "documentation, comments, or messages are wrong",
)

# TripAdvisor questions analyze reported experiences, not verified defects.
TA_1_1 = (
    "Judge strictly from the review above whether it reports a specific "
    "hotel service or facility problem that affected the stay.\n\n{0}\n\n"
    "Instruction: answer TRUE if the review reports such a problem, "
    "FALSE otherwise."
)

TA_1_2 = (
    "Judge strictly from the review above whether the guest asked hotel "
    "staff for help and says the problem remained unresolved.\n\n{0}\n\n"
    "Instruction: answer TRUE only if both are stated, FALSE otherwise."
)

TA_2 = (
    "Judge strictly from the review above which department should handle "
    "its hotel complaint, using the category descriptions.\n\n{0}"
)
TA_2_LABELS = (
    "housekeeping", "maintenance", "front_desk", "billing",
    "manual_review", "no_action",
)
TA_2_DESCRIPTIONS = (
    "cleaning or replenishing linens and room supplies",
    "repair of equipment, fixtures, or building facilities",
    "check-in, reservations, staff assistance, or room allocation",
    "charges, payments, or refunds",
    "multiple departments, unclear ownership, or another responsibility",
    "no actionable hotel complaint, praise, or external conditions only",
)

TA_3_1 = (
    "Does DOCUMENT {0} explicitly report a concrete hotel problem or complaint? "
    "Answer TRUE if it does, otherwise FALSE."
)

TA_3 = (
    "Do BOTH hotel reviews in DOCUMENT {0} and DOCUMENT {1} explicitly report "
    "the SAME specific hotel problem? Answer TRUE only if each review describes "
    "a concrete hotel failure or complaint and the problems match. Answer FALSE "
    "if either review reports no problem, contains only praise, or describes a "
    "different problem. Shared amenities, positive experiences, broad topics, "
    "and general sentiment do not count as a match."
)

REL_TA_1 = (
    "Judge strictly from the review above whether a hotel problem forced "
    "the guest to change rooms, leave early, or move to another hotel. "
    "\n\n{0}\n\nInstruction: answer TRUE only if the review reports "
    "that this happened, FALSE otherwise."
)
