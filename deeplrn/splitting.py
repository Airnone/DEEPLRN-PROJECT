"""Leakage-resistant whole-LGU corpus splitting with near-duplicate controls."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple
import argparse
import json
import random
import re


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    lgu: str
    year: int
    text: str
    source_path: str = ""


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def word_shingles(text: str, size: int = 5) -> Set[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        return set()
    if len(tokens) < size:
        return {" ".join(tokens)}
    return {" ".join(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}


def jaccard(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _minhash(shingles: Set[str], permutations: int = 64) -> Tuple[int, ...]:
    if not shingles:
        return tuple(0 for _ in range(permutations))
    signature = []
    for seed in range(permutations):
        minimum = min(
            int.from_bytes(
                blake2b(
                    shingle.encode("utf-8"),
                    digest_size=8,
                    person=f"dlrn{seed:04d}".encode("ascii"),
                ).digest(),
                "big",
            )
            for shingle in shingles
        )
        signature.append(minimum)
    return tuple(signature)


def near_duplicate_pairs(
    documents: Sequence[CorpusDocument],
    threshold: float = 0.85,
    shingle_size: int = 5,
    bands: int = 16,
) -> List[Tuple[str, str, float]]:
    """Return candidate pairs confirmed by exact shingle Jaccard similarity."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if 64 % bands:
        raise ValueError("bands must divide the 64-value MinHash signature")

    shingle_sets = {doc.doc_id: word_shingles(doc.text, shingle_size) for doc in documents}
    signatures = {doc.doc_id: _minhash(shingle_sets[doc.doc_id]) for doc in documents}
    rows = 64 // bands
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[str]] = {}
    for document in documents:
        signature = signatures[document.doc_id]
        for band in range(bands):
            key = (band, signature[band * rows:(band + 1) * rows])
            buckets.setdefault(key, []).append(document.doc_id)

    candidates: Set[Tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        ordered = sorted(set(members))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                candidates.add((left, right))

    confirmed = []
    for left, right in sorted(candidates):
        score = jaccard(shingle_sets[left], shingle_sets[right])
        if score >= threshold:
            confirmed.append((left, right, score))
    return confirmed


def create_split_manifest(
    documents: Sequence[CorpusDocument],
    *,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 13,
    duplicate_threshold: float = 0.85,
) -> Dict[str, Any]:
    if not documents:
        raise ValueError("at least one document is required")
    if abs(sum(ratios) - 1.0) > 1e-9 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("split ratios must be positive and sum to one")
    doc_ids = [doc.doc_id for doc in documents]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("document IDs must be unique")

    by_id = {doc.doc_id: doc for doc in documents}
    lgu_names = sorted({doc.lgu for doc in documents})
    lgu_union = UnionFind(lgu_names)
    duplicate_union = UnionFind(doc_ids)
    duplicate_pairs = near_duplicate_pairs(documents, threshold=duplicate_threshold)
    for left, right, _ in duplicate_pairs:
        duplicate_union.union(left, right)
        lgu_union.union(by_id[left].lgu, by_id[right].lgu)

    components: Dict[str, List[CorpusDocument]] = {}
    for document in documents:
        components.setdefault(lgu_union.find(document.lgu), []).append(document)

    rng = random.Random(seed)
    component_items = list(components.items())
    rng.shuffle(component_items)
    component_items.sort(key=lambda item: len(item[1]), reverse=True)

    split_names = ("train", "validation", "test")
    targets = {name: len(documents) * ratio for name, ratio in zip(split_names, ratios)}
    counts = {name: 0 for name in split_names}
    assignments: Dict[str, str] = {}
    for _, members in component_items:
        # Pick the partition with the largest proportional deficit. Stable
        # split-name order provides deterministic tie-breaking.
        chosen = max(
            split_names,
            key=lambda name: (targets[name] - counts[name]) / max(targets[name], 1.0),
        )
        for document in members:
            assignments[document.doc_id] = chosen
        counts[chosen] += len(members)

    entries = []
    for document in sorted(documents, key=lambda item: item.doc_id):
        entries.append(
            {
                "doc_id": document.doc_id,
                "lgu": document.lgu,
                "year": document.year,
                "split": assignments[document.doc_id],
                "near_duplicate_cluster": duplicate_union.find(document.doc_id),
                "source_path": document.source_path,
            }
        )
    return {
        "seed": seed,
        "ratios": dict(zip(split_names, ratios)),
        "duplicate_threshold": duplicate_threshold,
        "counts": counts,
        "near_duplicate_pairs": [
            {"left": left, "right": right, "jaccard": round(score, 6)}
            for left, right, score in duplicate_pairs
        ],
        "documents": entries,
    }


def load_corpus_documents(records_dir: str | Path) -> List[CorpusDocument]:
    documents = []
    for path in sorted(Path(records_dir).glob("*.json")):
        with path.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
        documents.append(
            CorpusDocument(
                doc_id=str(record["doc_id"]),
                lgu=str(record["lgu"]),
                year=int(record["year"]),
                text=str(record.get("document_text", "")),
                source_path=str(path.resolve()),
            )
        )
    return documents


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="Directory of training-record JSON files")
    parser.add_argument("--output", required=True, help="Output split-manifest JSON path")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--duplicate-threshold", type=float, default=0.85)
    args = parser.parse_args(argv)

    manifest = create_split_manifest(
        load_corpus_documents(args.records),
        seed=args.seed,
        duplicate_threshold=args.duplicate_threshold,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)


if __name__ == "__main__":
    main()
