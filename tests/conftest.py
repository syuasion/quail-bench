"""Put the project root on sys.path so tests run without an install."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def tripadvisor_corpus(tmp_path):
    """Prepare fabricated dated hotel reviews without a source download."""
    from quail_b.data import build_tripadvisor

    rows = [{
        "id": hotel * 100 + day, "offering_id": hotel,
        "title": "Room problem",
        "text": "The shower failed. Staff could not fix it. We moved rooms.",
        "date": f"January {day}, 2012", "author": {"id": "guest"},
        "ratings": {"overall": 2},
    } for hotel in (10, 20) for day in range(1, 5)]
    directory = tmp_path / "tripadvisor-corpus"
    build_tripadvisor(directory, rows=rows, hotel_limit=2, reviews_per_hotel=4,
                      pair_mode="all_unique_pairs")
    return directory / "manifest.json"
