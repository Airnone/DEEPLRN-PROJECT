"""
deeplrn.preprocessing.chunker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Splits extracted page text into overlapping, token-bounded chunks that
fit within RoBERTa's context window.

Design decisions
----------------
* **384-token max, 64-token overlap** — chosen so that two special tokens
  (``<s>`` and ``</s>``) still leave room and there's enough overlap to
  avoid splitting a sentence in half.
* **Sentence-boundary awareness** — the chunker tries to break at the
  nearest sentence boundary before the hard token limit, falling back to
  a hard cut only when a single sentence exceeds the budget.
* Each chunk carries provenance: source page number(s), character offsets,
  and the token span so downstream modules can map predictions back to
  the original PDF.

Usage
-----
>>> from deeplrn.preprocessing.chunker import DocumentChunker
>>> from deeplrn.preprocessing.pdf_extractor import PDFExtractor
>>> pages = PDFExtractor("report.pdf").extract()
>>> chunker = DocumentChunker()
>>> chunks = chunker.chunk_pages(pages)
>>> len(chunks)
42
>>> chunks[0].token_count
371

Bug-fix notes
-------------
* Character offsets are tracked cumulatively instead of using
  ``doc_text.index()`` to avoid ``ValueError`` on duplicate sentences.
* Overlap is measured by positional token count, not set intersection.
* Rewind logic caps ``rewind_count`` to prevent infinite loops when
  all collected sentences fit within the overlap budget.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from deeplrn.config import CHUNK_CFG, ChunkConfig
from deeplrn.preprocessing.pdf_extractor import PageData

logger = logging.getLogger(__name__)

# ── Simple sentence splitter ─────────────────────────────────────────────────
# Handles common abbreviations in Philippine government reports.
_ABBREVIATIONS = frozenset({
    # Common English
    "no", "dr", "mr", "mrs", "ms", "jr", "sr", "st", "dept", "gov",
    "sec", "gen", "col", "lt", "sgt", "vol", "pp", "vs", "etc",
    "approx", "est", "inc", "corp", "ref", "fig",
    # Philippine government / COA titles
    "hon", "engr", "atty", "dir", "supt", "brgy", "mun", "prov",
    # Calendar / fiscal
    "jan", "feb", "mar", "apr", "aug", "sept", "sep", "oct", "nov", "dec",
    "cy", "fy",
})

# Split on sentence-ending punctuation followed by whitespace and an uppercase char.
_SENT_BOUNDARY = re.compile(r"([.!?])\s+(?=[A-Z0-9\"])")


def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences, merging back false splits caused by abbreviations."""
    # First pass: split at every boundary candidate
    fragments = _SENT_BOUNDARY.split(text)

    # _SENT_BOUNDARY with a capture group gives us alternating [text, punct, text, …]
    # Reassemble into (text+punct) pairs
    raw_sents: List[str] = []
    i = 0
    while i < len(fragments):
        if i + 1 < len(fragments) and fragments[i + 1] in ".!?":
            raw_sents.append(fragments[i] + fragments[i + 1])
            i += 2
        else:
            raw_sents.append(fragments[i])
            i += 1

    # Second pass: merge back sentences that end with a known abbreviation
    merged: List[str] = []
    for sent in raw_sents:
        stripped = sent.strip()
        if not stripped:
            continue
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = merged[-1] + " " + stripped
        else:
            merged.append(stripped)

    return merged


def _ends_with_abbreviation(text: str) -> bool:
    """Check if *text* ends with a known abbreviation followed by a period."""
    text = text.rstrip()
    if not text.endswith("."):
        return False
    # Get the last word (without the period)
    last_word = text[:-1].split()[-1] if text[:-1].split() else ""
    return last_word.lower() in _ABBREVIATIONS


# ── Data container ───────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single token-bounded text window with provenance information."""

    chunk_id: int
    """Global zero-based index of this chunk in the document."""

    text: str
    """The raw text content of this chunk."""

    token_ids: List[int] = field(default_factory=list)
    """Sub-word token IDs produced by the RoBERTa tokenizer."""

    token_count: int = 0
    """Number of sub-word tokens (excluding special tokens)."""

    page_numbers: List[int] = field(default_factory=list)
    """1-indexed page(s) this chunk's text originates from."""

    char_offset_start: int = 0
    """Character offset of this chunk's text within the concatenated document string."""

    char_offset_end: int = 0
    """Character offset end (exclusive)."""

    overlap_tokens_prev: int = 0
    """How many leading tokens are shared with the previous chunk."""


# ── Core chunker ─────────────────────────────────────────────────────────────

class DocumentChunker:
    """Split a document's pages into overlapping, token-bounded chunks.

    Parameters
    ----------
    config : ChunkConfig, optional
        Override default chunk / overlap sizes.  Falls back to
        ``CHUNK_CFG`` singleton.
    tokenizer : PreTrainedTokenizerBase, optional
        A pre-loaded Hugging Face tokenizer.  If *None*, one is
        loaded from ``config.tokenizer_name``.
    """

    def __init__(
        self,
        config: ChunkConfig | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.cfg = config or CHUNK_CFG
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.cfg.tokenizer_name,
            use_fast=True,
        )
        # Budget *excluding* <s> and </s> special tokens
        self._budget = self.cfg.max_tokens - 2
        if self._budget <= self.cfg.overlap_tokens:
            raise ValueError(
                f"Token budget ({self._budget}) must exceed overlap ({self.cfg.overlap_tokens})"
            )

    # ── public API ───────────────────────────────────────────────────────

    def chunk_pages(self, pages: List[PageData]) -> List[TextChunk]:
        """Concatenate all pages and split into overlapping chunks.

        The method:
        1. Joins page texts with page-break markers so we can map chunks
           back to source pages.
        2. Splits the concatenated text into sentences.
        3. Greedily fills each chunk with sentences up to ``max_tokens``,
           then backs up to the overlap boundary for the next chunk.

        Returns
        -------
        list[TextChunk]
        """
        if not pages:
            return []

        # Step 1 — build a flat document string and a page-offset map
        doc_text, page_map = self._concatenate_pages(pages)

        # Step 2 — sentence-split
        sentences = _split_sentences(doc_text)
        if not sentences:
            logger.warning("No sentences found after splitting — returning empty chunk list.")
            return []

        # Step 3 — compute sentence char-offsets within doc_text for
        # provenance tracking (avoids fragile doc_text.index() calls).
        sent_char_offsets: List[Tuple[int, int]] = []
        search_from = 0
        for sent in sentences:
            pos = doc_text.find(sent, search_from)
            if pos == -1:
                # Fallback: sentence was reconstructed and doesn't appear
                # verbatim — use the current search cursor.
                pos = search_from
            sent_char_offsets.append((pos, pos + len(sent)))
            search_from = pos + len(sent)

        # Step 4 — tokenise each sentence individually (caching)
        sent_tokens: List[List[int]] = []
        for sent in sentences:
            ids = self.tokenizer.encode(sent, add_special_tokens=False)
            sent_tokens.append(ids)

        # Step 5 — greedy chunking with overlap
        chunks = self._greedy_chunk(
            sentences, sent_tokens, sent_char_offsets, doc_text, page_map,
        )

        logger.info(
            "Chunked %d pages → %d chunks  (budget=%d, overlap=%d)",
            len(pages),
            len(chunks),
            self._budget,
            self.cfg.overlap_tokens,
        )
        return chunks

    # ── internals ────────────────────────────────────────────────────────

    @staticmethod
    def _concatenate_pages(pages: List[PageData]) -> Tuple[str, List[Tuple[int, int, int]]]:
        """Join page texts with double-newlines.

        Returns
        -------
        doc_text : str
            The full document string.
        page_map : list of (start_char, end_char, page_number)
            Character spans for each page inside *doc_text*.
        """
        parts: List[str] = []
        page_map: List[Tuple[int, int, int]] = []
        offset = 0

        for page in pages:
            text = page.text.strip()
            if not text:
                continue
            start = offset
            parts.append(text)
            offset += len(text)
            # Add separator
            parts.append("\n\n")
            offset += 2
            page_map.append((start, offset - 2, page.page_number))

        doc_text = "".join(parts).rstrip("\n")
        return doc_text, page_map

    def _greedy_chunk(
        self,
        sentences: List[str],
        sent_tokens: List[List[int]],
        sent_char_offsets: List[Tuple[int, int]],
        doc_text: str,
        page_map: List[Tuple[int, int, int]],
    ) -> List[TextChunk]:
        """Fill chunks greedily from sentences, respecting token budget and overlap."""
        chunks: List[TextChunk] = []
        chunk_id = 0
        sent_idx = 0
        n_sents = len(sentences)

        while sent_idx < n_sents:
            start_sent_idx = sent_idx  # track for infinite-loop guard

            # ── accumulate sentences until budget is full ────────────────
            collected_sents: List[int] = []   # indices into `sentences`
            running_tokens = 0

            while sent_idx < n_sents:
                stoks = sent_tokens[sent_idx]
                cost = len(stoks)

                # If a single sentence exceeds the budget, force-include it
                # (it will be truncated to budget below).
                if cost > self._budget and not collected_sents:
                    collected_sents.append(sent_idx)
                    running_tokens += cost
                    sent_idx += 1
                    break

                if running_tokens + cost > self._budget:
                    break

                collected_sents.append(sent_idx)
                running_tokens += cost
                sent_idx += 1

            if not collected_sents:
                break

            # ── build chunk text & tokens ────────────────────────────────
            chunk_text = " ".join(sentences[i] for i in collected_sents)
            chunk_token_ids = self.tokenizer.encode(
                chunk_text,
                add_special_tokens=True,        # adds <s> … </s>
                max_length=self.cfg.max_tokens,
                truncation=True,
            )

            # Character offsets — use pre-computed sentence positions
            char_start = sent_char_offsets[collected_sents[0]][0]
            char_end = sent_char_offsets[collected_sents[-1]][1]

            # Page provenance
            pages_here = sorted({
                pm[2]
                for pm in page_map
                if pm[0] < char_end and pm[1] > char_start
            })

            # Overlap accounting — positional token count, not set intersection.
            # Counts how many leading tokens in this chunk are shared with
            # the trailing tokens of the previous chunk (by position).
            overlap_count = 0
            if chunks:
                prev_tokens = chunks[-1].token_ids  # includes <s>...<eos>
                # Strip special tokens for comparison: prev tail vs cur head.
                prev_content = prev_tokens[1:-1]  # remove <s> and </s>
                cur_content = chunk_token_ids[1:-1]
                max_check = min(self.cfg.overlap_tokens, len(prev_content), len(cur_content))
                if max_check > 0:
                    prev_tail = prev_content[-max_check:]
                    cur_head = cur_content[:max_check]
                    # Count matching positions from the start
                    for p, c in zip(prev_tail, cur_head):
                        if p == c:
                            overlap_count += 1
                        else:
                            break

            chunks.append(TextChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                token_ids=chunk_token_ids,
                token_count=len(chunk_token_ids) - 2,   # exclude <s>, </s>
                page_numbers=pages_here,
                char_offset_start=char_start,
                char_offset_end=char_end,
                overlap_tokens_prev=overlap_count,
            ))
            chunk_id += 1

            # ── rewind for overlap ───────────────────────────────────────
            # Walk backwards through collected sentences to find how many
            # we need to re-include so that the next chunk starts with
            # ~overlap_tokens of shared context.
            if sent_idx < n_sents:
                rewind_tokens = 0
                rewind_count = 0
                for ri in reversed(collected_sents):
                    stok_len = len(sent_tokens[ri])
                    if rewind_tokens + stok_len > self.cfg.overlap_tokens:
                        break
                    rewind_tokens += stok_len
                    rewind_count += 1

                # Guard: never rewind *all* collected sentences — that
                # would reset sent_idx to its pre-collection value and
                # create an infinite loop.
                if rewind_count >= len(collected_sents):
                    rewind_count = max(0, len(collected_sents) - 1)

                if rewind_count > 0:
                    sent_idx -= rewind_count

            # Final infinite-loop safeguard: if we haven't advanced at all,
            # force progress by one sentence.
            if sent_idx <= start_sent_idx:
                sent_idx = start_sent_idx + 1

        return chunks

    # ── convenience ──────────────────────────────────────────────────────

    def chunk_text(self, text: str, page_number: int = 1) -> List[TextChunk]:
        """Chunk a plain text string (wraps it in a single PageData)."""
        page = PageData(page_number=page_number, text=text)
        return self.chunk_pages([page])
