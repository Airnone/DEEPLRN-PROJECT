"""
Tests for the preprocessing module — PDF extractor logic.

Run with: pytest tests/test_pdf_extractor.py -v
"""

from __future__ import annotations

import pytest

from deeplrn.preprocessing.pdf_extractor import (
    _normalise_whitespace,
    _detect_headers,
    PageData,
    WordBox,
)


# ── Whitespace normalisation ─────────────────────────────────────────────────

class TestNormaliseWhitespace:

    def test_collapses_spaces(self):
        assert _normalise_whitespace("hello   world") == "hello world"

    def test_collapses_tabs(self):
        assert _normalise_whitespace("hello\t\tworld") == "hello world"

    def test_preserves_newlines(self):
        result = _normalise_whitespace("line1\nline2")
        assert "\n" in result

    def test_normalises_windows_line_endings(self):
        result = _normalise_whitespace("line1\r\nline2\r\nline3")
        assert "\r" not in result
        assert result == "line1\nline2\nline3"

    def test_normalises_bare_carriage_return(self):
        result = _normalise_whitespace("line1\rline2")
        assert "\r" not in result
        assert result == "line1\nline2"

    def test_replaces_non_breaking_space(self):
        result = _normalise_whitespace("hello\xa0world")
        assert "\xa0" not in result
        assert result == "hello world"

    def test_strips_leading_trailing(self):
        assert _normalise_whitespace("  hello  ") == "hello"

    def test_empty_string(self):
        assert _normalise_whitespace("") == ""
        assert _normalise_whitespace("   ") == ""


# ── Header detection ─────────────────────────────────────────────────────────

class TestDetectHeaders:

    def test_part_roman_numeral(self):
        assert "PART IV" in _detect_headers("PART IV\nSome text here.")

    def test_part_with_D_and_M_roman(self):
        """Roman numerals D and M should be matched."""
        assert "CHAPTER MD" in _detect_headers("CHAPTER MD\nContent.")

    def test_numbered_section(self):
        headers = _detect_headers("1. Introduction\nSome text.")
        assert any("1. Introduction" in h for h in headers)

    def test_multi_level_numbered_section(self):
        """Multi-level section numbers like 1.1 or 2.3.1 should match."""
        headers = _detect_headers("1.1. Background\nSome text.")
        assert any("1.1. Background" in h for h in headers)

    def test_all_caps_with_digits(self):
        """ALL-CAPS headers containing digits should match."""
        headers = _detect_headers("CY 2022 FINANCIAL AUDIT REPORT")
        assert len(headers) >= 1

    def test_observation_singular_and_plural(self):
        headers_s = _detect_headers("Observation\nContent.")
        headers_p = _detect_headers("Observations\nContent.")
        assert len(headers_s) >= 1
        assert len(headers_p) >= 1

    def test_recommendation_variations(self):
        headers = _detect_headers("Recommendations\nAction items here.")
        assert len(headers) >= 1

    def test_status_of_implementation(self):
        headers = _detect_headers("Status of Implementation\nDetails follow.")
        assert len(headers) >= 1

    def test_no_false_positive_short_text(self):
        """Regular body text should not be detected as headers."""
        headers = _detect_headers("This is a normal sentence with words.")
        assert len(headers) == 0


# ── PageData / WordBox dataclasses ───────────────────────────────────────────

class TestDataContainers:

    def test_page_data_defaults(self):
        page = PageData(page_number=1, text="Test")
        assert page.source == "pdfplumber"
        assert page.word_boxes == []
        assert page.headers == []
        assert page.metadata == {}

    def test_word_box_bbox(self):
        wb = WordBox(text="hello", x0=10.0, top=20.0, x1=50.0, bottom=30.0)
        assert wb.bbox == (10.0, 20.0, 50.0, 30.0)
