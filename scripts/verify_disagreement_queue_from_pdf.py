"""Verify review candidates against cited PDF pages and replace malformed records."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pdfplumber


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _normalize(text: str) -> str:
    replacements = {
        "â‚±": "peso ",
        "₱": "peso ",
        "â€™": "'",
        "â€“": "-",
        "â€”": "-",
        "\u00a0": " ",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0).casefold(), match.start(), match.end())
        for match in re.finditer(r"[A-Za-z0-9]+", text)
    ]


def _find_token_sequence(
    haystack: list[tuple[str, int, int]],
    needle: list[str],
    *,
    reverse: bool = False,
) -> int | None:
    starts = range(len(haystack) - len(needle), -1, -1) if reverse else range(len(haystack))
    for start in starts:
        if start + len(needle) > len(haystack):
            continue
        if [token for token, _begin, _end in haystack[start : start + len(needle)]] == needle:
            return start
    return None


def _fresh_pdf_excerpt(old_text: str, pages: list[str]) -> str:
    """Slice a clean excerpt from cited PDF pages using old boundary anchors."""

    page_text = "\n\n".join(pages)
    page_tokens = _tokens_with_spans(page_text)
    old_tokens = [token for token, _begin, _end in _tokens_with_spans(old_text)]
    anchor_size = min(16, len(old_tokens))
    if anchor_size < 5:
        raise ValueError("observation has too few tokens for PDF boundary anchoring")
    start = _find_token_sequence(page_tokens, old_tokens[:anchor_size])
    end = _find_token_sequence(page_tokens, old_tokens[-anchor_size:], reverse=True)
    if start is None or end is None or end < start:
        raise ValueError("observation boundary anchors were not found in cited PDF pages")
    start_character = page_tokens[start][1]
    end_character = page_tokens[end + anchor_size - 1][2]
    excerpt = page_text[start_character:end_character]
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", excerpt)).strip()


def _record(item: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(item["source_record"]).read_text(encoding="utf-8"))


def _page_texts(items: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
    requested: dict[str, set[int]] = {}
    for item in items:
        record = _record(item)
        requested.setdefault(str(record["source_pdf"]), set()).update(
            int(page) for page in record.get("evidence_page_numbers", [])
        )
    result: dict[tuple[str, int], str] = {}
    for pdf_path, page_numbers in requested.items():
        with pdfplumber.open(pdf_path) as pdf:
            for page_number in sorted(page_numbers):
                result[(pdf_path, page_number)] = pdf.pages[page_number - 1].extract_text() or ""
    return result


def _quality_flags(text: str) -> list[str]:
    normalized = _normalize(text)
    words = normalized.split()
    flags: list[str] = []
    if len(text) < 150 or len(words) < 25:
        flags.append("short_or_fragmentary")
    if len(text) > 20_000:
        flags.append("overbroad_boundary")
    corrupted_characters = sum(
        character in "âãæéðï¿½" or "\u3000" <= character <= "\u9fff" for character in text
    )
    if corrupted_characters >= 4:
        flags.append("corrupted_text_layer")
    if re.fullmatch(
        r"(?:project objective|required manpower|schedule of activities|"
        r"course outline and resource speaker|budgetary requirements|"
        r"share from goccs.*|inter local transfer.*|other shares from national tax collections.*)",
        normalized,
    ):
        flags.append("heading_or_table_row")
    finding_signal = re.search(
        r"\b(?:not|without|lack(?:ed|ing|s)?|fail(?:ed|ure)?|remain(?:ed|s)?|"
        r"unliquidated|unsupported|overstat(?:ed|ement)?|understat(?:ed|ement)?|"
        r"doubtful|deficien(?:cy|cies|t)|contrary|excessive|unnecessary|"
        r"erroneously|could not|unable|irregular|ineligible|unreconciled)\b",
        normalized,
    )
    if not finding_signal:
        flags.append("no_explicit_finding_signal")
    if normalized.startswith(("section ", "fundamental principles ", "approval of disbursements ")):
        flags.append("legal_background_fragment")
    if normalized.startswith(("to start ", "keep a complete ")):
        flags.append("recommendation_fragment")
    if re.search(r"\b4\s+10\s+lgu counterpart\b", normalized):
        flags.append("multiple_top_level_findings")
    return flags


def _page_support(
    text: str,
    pages: list[str],
) -> tuple[float, bool]:
    normalized_text = _normalize(text)
    normalized_pages = _normalize(" ".join(pages))
    if not normalized_text:
        return 0.0, False
    tokens = normalized_text.split()
    windows = [
        " ".join(tokens[: min(24, len(tokens))]),
        " ".join(tokens[-min(24, len(tokens)) :]),
    ]
    supported = [window in normalized_pages for window in windows if window]
    score = sum(supported) / len(supported) if supported else 0.0
    return score, score >= 0.5


def build_verified_queue(
    original_queue: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    limit: int,
    validation_cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pages = _page_texts(pool)
    evaluated: list[dict[str, Any]] = []
    for item in pool:
        record = _record(item)
        page_values = [
            pages[(str(record["source_pdf"]), int(page))]
            for page in record.get("evidence_page_numbers", [])
        ]
        support_score, supported = _page_support(item["observation_text"], page_values)
        fresh_text = ""
        if supported:
            try:
                fresh_text = _fresh_pdf_excerpt(item["observation_text"], page_values)
            except ValueError:
                supported = False
        flags = _quality_flags(fresh_text or item["observation_text"])
        evaluated.append(
            {
                **item,
                "original_extracted_text": item["observation_text"],
                "observation_text": fresh_text or item["observation_text"],
                "pdf_page_numbers_verified": record.get("evidence_page_numbers", []),
                "pdf_text_support_score": round(support_score, 6),
                "pdf_text_supported": supported,
                "boundary_quality_flags": flags,
                "reextraction_status": "pdf_verified" if supported and not flags else "rejected",
            }
        )

    acceptable = [item for item in evaluated if item["reextraction_status"] == "pdf_verified"]
    acceptable.sort(key=lambda item: (-float(item["priority_score"]), item["doc_id"]))
    original_ids = {item["doc_id"] for item in original_queue}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    validation_count = 0

    def add(item: dict[str, Any]) -> bool:
        nonlocal validation_count
        if item["doc_id"] in selected_ids or len(selected) >= limit:
            return False
        if item["split"] == "validation" and validation_count >= validation_cap:
            return False
        selected.append(item)
        selected_ids.add(item["doc_id"])
        validation_count += item["split"] == "validation"
        return True

    for item in acceptable:
        if item["doc_id"] in original_ids:
            add(item)
    for item in acceptable:
        add(item)
        if len(selected) >= limit:
            break
    if len(selected) != limit:
        raise ValueError(f"only {len(selected)} PDF-verified candidates for requested {limit}")
    for rank, item in enumerate(selected, start=1):
        item["queue_rank"] = rank
        item["review_decision"] = None
        item["corrected_labels"] = []
        item["review_notes"] = ""
    return selected, evaluated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--validation-cap", type=int, default=20)
    args = parser.parse_args()

    original = _read_jsonl(args.queue)
    pool = _read_jsonl(args.pool)
    selected, evaluated = build_verified_queue(
        original, pool, limit=args.limit, validation_cap=args.validation_cap
    )
    for path, rows in ((args.output, selected), (args.audit, evaluated)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    original_ids = {item["doc_id"] for item in original}
    selected_ids = {item["doc_id"] for item in selected}
    split_counts = Counter(item["split"] for item in selected)
    summary = {
        "queue_size": len(selected),
        "original_records_retained": len(original_ids & selected_ids),
        "original_records_replaced": len(original_ids - selected_ids),
        "selected_split_counts": dict(sorted(split_counts.items())),
        "all_selected_pdf_text_supported": all(item["pdf_text_supported"] for item in selected),
        "all_selected_boundary_quality_flags_empty": all(
            not item["boundary_quality_flags"] for item in selected
        ),
        "test_split_accessed": False,
        "labels_changed": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
