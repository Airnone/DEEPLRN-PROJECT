from __future__ import annotations

from deeplrn.splitting import CorpusDocument, create_split_manifest, near_duplicate_pairs


def documents():
    repeated = "audit observation unsupported payment procurement documents " * 12
    return [
        CorpusDocument("a-2022", "LGU A", 2022, repeated + " alpha"),
        CorpusDocument("a-2023", "LGU A", 2023, "different annual text for alpha municipality"),
        CorpusDocument("b-2023", "LGU B", 2023, repeated + " beta"),
        CorpusDocument("c-2023", "LGU C", 2023, "cash advance liquidation finding unique gamma"),
        CorpusDocument("d-2023", "LGU D", 2023, "infrastructure project unique delta"),
        CorpusDocument("e-2023", "LGU E", 2023, "procurement eligibility unique epsilon"),
    ]


def test_near_duplicates_are_detected():
    pairs = near_duplicate_pairs(documents(), threshold=0.70)
    ids = {(left, right) for left, right, _ in pairs}
    assert ("a-2022", "b-2023") in ids


def test_whole_lgu_and_duplicate_clusters_share_a_split():
    manifest = create_split_manifest(documents(), seed=7, duplicate_threshold=0.70)
    split = {item["doc_id"]: item["split"] for item in manifest["documents"]}
    assert split["a-2022"] == split["a-2023"]
    assert split["a-2022"] == split["b-2023"]


def test_split_is_deterministic():
    first = create_split_manifest(documents(), seed=19, duplicate_threshold=0.70)
    second = create_split_manifest(documents(), seed=19, duplicate_threshold=0.70)
    assert first == second
