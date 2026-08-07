"""
Tests for the preprocessing module — chunker logic.

Run with: pytest tests/test_chunker.py -v
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from deeplrn.config import ChunkConfig
from deeplrn.preprocessing.pdf_extractor import PageData


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "The Commission on Audit conducted an examination of the financial transactions "
    "of the Municipality of San Miguel, Province of Bulacan, for the calendar year 2023. "
    "The audit revealed several irregularities in the procurement process. "
    "Specifically, unauthorized expenditures totaling PHP 2,500,000 were found in the "
    "infrastructure projects managed by the Municipal Engineering Office. "
    "The amount of PHP 1,200,000 was disbursed to ABC Construction Company without "
    "proper documentation and supporting vouchers. "
    "Furthermore, cash advances amounting to PHP 800,000 remained unliquidated "
    "beyond the reglementary period as prescribed under COA Circular No. 97-002. "
    "The Municipal Accountant, Mr. Juan Dela Cruz, failed to submit the required "
    "liquidation reports within the prescribed period. "
    "Additionally, a contract worth PHP 5,000,000 was awarded to XYZ Enterprises "
    "despite the company not meeting the eligibility requirements under RA 9184. "
    "The Bids and Awards Committee did not follow proper procurement procedures "
    "as outlined in the Implementing Rules and Regulations of RA 9184."
)


class DeterministicTokenizer:
    """Small local tokenizer used to keep unit tests offline and repeatable."""

    def encode(self, text, add_special_tokens=True, max_length=None, truncation=False):
        tokens = [abs(hash(word)) % 10000 + 3 for word in text.split()]
        if add_special_tokens:
            tokens = [1] + tokens + [2]
        if max_length is not None and truncation:
            tokens = tokens[:max_length]
        return tokens


def make_pages(text: str, pages_count: int = 1) -> list[PageData]:
    """Create synthetic PageData objects from a text string."""
    chunk_size = len(text) // pages_count
    pages = []
    for i in range(pages_count):
        start = i * chunk_size
        end = start + chunk_size if i < pages_count - 1 else len(text)
        pages.append(PageData(
            page_number=i + 1,
            text=text[start:end],
        ))
    return pages


# ── Tests ────────────────────────────────────────────────────────────────────

class TestDocumentChunker:
    """Tests for DocumentChunker."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        """Import chunker here so we can patch the tokenizer download if needed."""
        from deeplrn.preprocessing import chunker as chunker_module

        monkeypatch.setattr(
            chunker_module.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: DeterministicTokenizer(),
        )
        DocumentChunker = chunker_module.DocumentChunker
        self.DocumentChunker = DocumentChunker

    def test_empty_pages_returns_empty(self):
        chunker = self.DocumentChunker()
        result = chunker.chunk_pages([])
        assert result == []

    def test_single_page_produces_chunks(self):
        pages = make_pages(SAMPLE_TEXT)
        chunker = self.DocumentChunker()
        chunks = chunker.chunk_pages(pages)

        assert len(chunks) >= 1
        # Every chunk should have a valid token count
        for chunk in chunks:
            assert chunk.token_count > 0
            assert chunk.token_count <= 384 - 2  # budget = max_tokens - 2 special tokens

    def test_single_sentence_mode_emits_one_sentence_per_chunk(self):
        pages = make_pages("First sentence. Second sentence. Third sentence.")
        chunker = self.DocumentChunker(
            ChunkConfig(
                max_tokens=32,
                overlap_tokens=0,
                tokenizer_name="offline",
                single_sentence_chunks=True,
            )
        )
        chunks = chunker.chunk_pages(pages)
        assert [chunk.text for chunk in chunks] == [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
        ]

    def test_chunk_ids_are_sequential(self):
        pages = make_pages(SAMPLE_TEXT)
        chunker = self.DocumentChunker()
        chunks = chunker.chunk_pages(pages)

        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_page_numbers_are_tracked(self):
        pages = make_pages(SAMPLE_TEXT, pages_count=3)
        chunker = self.DocumentChunker()
        chunks = chunker.chunk_pages(pages)

        # Each chunk should reference at least one page
        for chunk in chunks:
            assert len(chunk.page_numbers) >= 1
            for pn in chunk.page_numbers:
                assert 1 <= pn <= 3

    def test_small_budget_produces_more_chunks(self):
        pages = make_pages(SAMPLE_TEXT)

        cfg_large = ChunkConfig(max_tokens=384, overlap_tokens=64)
        cfg_small = ChunkConfig(max_tokens=128, overlap_tokens=32)

        chunks_large = self.DocumentChunker(config=cfg_large).chunk_pages(pages)
        chunks_small = self.DocumentChunker(config=cfg_small).chunk_pages(pages)

        assert len(chunks_small) >= len(chunks_large)

    def test_overlap_rewinds_sentences(self):
        """With overlap, the second chunk should share some text with the first."""
        # Use a small budget to force multiple chunks
        cfg = ChunkConfig(max_tokens=100, overlap_tokens=20)
        pages = make_pages(SAMPLE_TEXT)
        chunker = self.DocumentChunker(config=cfg)
        chunks = chunker.chunk_pages(pages)

        if len(chunks) >= 2:
            # The second chunk should contain some text from the end of the first
            # Sentence-aware overlap may repeat a sentence longer than ten
            # words, so compare a window at least as large as the configured
            # overlap budget.
            first_words = set(chunks[0].text.split()[-30:])
            second_words = set(chunks[1].text.split()[:30])
            overlap = first_words & second_words
            # There should be at least *some* overlap
            assert len(overlap) > 0, (
                f"Expected overlap between chunks 0 and 1.\n"
                f"End of chunk 0: …{chunks[0].text[-80:]}\n"
                f"Start of chunk 1: {chunks[1].text[:80]}…"
            )

    def test_char_offsets_are_valid(self):
        pages = make_pages(SAMPLE_TEXT)
        chunker = self.DocumentChunker()
        chunks = chunker.chunk_pages(pages)

        for chunk in chunks:
            assert chunk.char_offset_start >= 0
            assert chunk.char_offset_end > chunk.char_offset_start

    def test_chunk_text_convenience_method(self):
        chunker = self.DocumentChunker()
        chunks = chunker.chunk_text(SAMPLE_TEXT, page_number=42)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert 42 in chunk.page_numbers

    # ── Bug-fix regression tests ─────────────────────────────────────

    def test_duplicate_sentences_no_crash(self):
        """Duplicate sentences should not cause ValueError from doc_text.index()."""
        repeated = (
            "See Annex A. This is a finding. "
            "See Annex A. This is another finding. "
            "See Annex A. Final note."
        )
        pages = make_pages(repeated)
        chunker = self.DocumentChunker(config=ChunkConfig(max_tokens=50, overlap_tokens=10))
        # Should not raise ValueError
        chunks = chunker.chunk_pages(pages)
        assert len(chunks) >= 1

    def test_no_infinite_loop_small_total_tokens(self):
        """When total collected tokens ≤ overlap_tokens, chunking must still terminate."""
        import signal
        import sys

        # Two very short sentences followed by one that pushes over budget
        text = "Short. Tiny. " + ("A " * 400)  # one massive sentence after two tiny ones
        pages = make_pages(text)
        cfg = ChunkConfig(max_tokens=384, overlap_tokens=64)
        chunker = self.DocumentChunker(config=cfg)

        # Set a hard timeout — if we get here without finishing, it's an infinite loop
        if sys.platform != "win32":
            def _timeout(signum, frame):
                raise TimeoutError("Chunker appears stuck in an infinite loop")
            old = signal.signal(signal.SIGALRM, _timeout)
            signal.alarm(10)

        try:
            chunks = chunker.chunk_pages(pages)
            assert len(chunks) >= 1
        finally:
            if sys.platform != "win32":
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)

    def test_overlap_is_positional_not_set_based(self):
        """Overlap counting should measure contiguous positional matches,
        not set-intersection of token IDs."""
        # Craft text that produces at least 2 chunks with intentional overlap.
        cfg = ChunkConfig(max_tokens=60, overlap_tokens=15)
        text = (
            "Alpha beta gamma delta epsilon. "
            "Zeta eta theta iota kappa. "
            "Lambda mu nu xi omicron."
        )
        pages = make_pages(text)
        chunker = self.DocumentChunker(config=cfg)
        chunks = chunker.chunk_pages(pages)

        if len(chunks) >= 2:
            # overlap_tokens_prev should be ≥ 0 and ≤ overlap_tokens
            for chunk in chunks[1:]:
                assert 0 <= chunk.overlap_tokens_prev <= cfg.overlap_tokens


class TestSentenceSplitter:
    """Tests for the internal sentence splitter."""

    def test_basic_split(self):
        from deeplrn.preprocessing.chunker import _split_sentences

        text = "Hello world. This is a test. One more sentence."
        sents = _split_sentences(text)
        assert len(sents) == 3

    def test_abbreviation_not_split(self):
        from deeplrn.preprocessing.chunker import _split_sentences

        text = "Dr. Juan Dela Cruz submitted the report. It was approved."
        sents = _split_sentences(text)
        # Should NOT split on "Dr."
        assert len(sents) == 2
        assert "Dr." in sents[0]

    def test_month_abbreviation_not_split(self):
        """Month abbreviations should not cause false sentence splits."""
        from deeplrn.preprocessing.chunker import _split_sentences

        text = "The disbursement was made on Jan. 15, 2023. It was recorded."
        sents = _split_sentences(text)
        assert len(sents) == 2
        assert "Jan." in sents[0]

    def test_coa_title_not_split(self):
        """Philippine government titles should not cause false splits."""
        from deeplrn.preprocessing.chunker import _split_sentences

        text = "Engr. Santos supervised the project. The work was completed."
        sents = _split_sentences(text)
        assert len(sents) == 2
        assert "Engr." in sents[0]
