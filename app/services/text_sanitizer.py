"""Query-time sanitizer for retrieved knowledge-base chunks.

The Excel corpus (`docs/*.xls[x]`, `source_type == "excel_qa"`) stores each row as
``Question:\\n{question}\\n\\nAnswer:\\n{answer}`` where the trainer's *answer* cell
carries conversational boilerplate that is meaningless once the chunk is fed to the
RAG prompt:

* opening greetings -- ``درود بر شما سفیر سلامت شرکت پرشین وی`` and its ~2000
  spelling variants (``برشما`` / ``سفیرسلامت`` / ``شرکت پرشین`` / trailing emoji /
  appended ``همکار گرامی``),
* closing calls-to-action -- ``اگر خواستید، می‌تونم ... 🌱``,
* referral fallbacks -- ``درخواست شما به واحد مربوطه ارجاع داده شد``,
  ``تیکت شما ... انتقال داده می شود``,
  ``اطلاعات کافی در پایگاه دانش ... یافت نشد``,
* internal trainer annotations -- ``نیاز به نگارش این متن نمی باشد``.

The model reproduces the most salient opening of its context, so these lines leak
straight into user-visible answers. This module strips them at *query time*, which
means already-ingested vectors are fixed without a re-ingestion pass.

Design notes
------------
* Line oriented: a boilerplate line is either dropped outright or has its
  boilerplate *prefix/segment* removed while the remaining answer body survives.
  ``… شرکت پرشین وی همکار گرامی پیشنهاد میشه از حجم کم شروع کنید`` keeps
  ``پیشنهاد میشه از حجم کم شروع کنید`` -- dropping the whole line would destroy
  real answer content.
* Idempotent: ``sanitize_chunk(sanitize_chunk(x)) == sanitize_chunk(x)``.
* Ranking-neutral: scores, order and metadata are never touched; only
  ``page_content`` is rewritten.
"""

import logging
import re
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)

# A sanitized chunk shorter than this carries no usable answer content (it was a
# greeting-only / referral-only row) and is dropped from the context.
MIN_CONTENT_CHARS = 20

# Leading noise observed on real rows: stray quotes, a leading full stop, Arabic
# tatweel/harakat, bullet characters. Stripped *before* anchoring a match so that
# `"درود بر شما ...` and `. درود بر شما ...` are still recognised as greetings.
_LEADING_JUNK = re.compile(r"^[\s\"'«»\.,،:;؛\-–—_*#ٰ\u064B-\u0652\u200c]+")

# Runs of horizontal whitespace left behind by a removed segment.
_WHITESPACE_RUN = re.compile(r"[ \t\u00a0]+")

# Trailing/inline emoji and pictographs (`🌿` `🌱` `🌹` `🌷` `🙏`).
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]+"
)

# A line that still holds a letter/digit after stripping is worth keeping.
_HAS_CONTENT = re.compile(r"\w")

# `شرکت` is optional because the corpus also carries the truncated
# `درود برشما سفیر سلامت پرشین وی` form (845 occurrences).
_GREETING_SIGNATURE = (
    r"درود\s*بر\s*(?:(?:شما|سفیر|سلامت)\s*){0,3}(?:شرکت\s*)?پرشین(?:\s*وی)?"
)

# Letters (Persian/Arabic block + Arabic-Indic digits + ASCII) used as a negative
# lookahead so a greeting fragment can never bite into the middle of a longer
# word: `با سلامتی شما` must NOT become `تی شما`, and `درود بر سفیران` must not
# become `ان`.
_LETTER = r"[\u0621-\u06CC\u06F0-\u06F9A-Za-z]"

# Matched at the *start* of a line (after `_LEADING_JUNK` removal) and consumed;
# the remainder of the line is the answer body and survives.
# Order matters: the long signature must win over the bare `درود بر شما` form.
_GREETING_PREFIX = re.compile(
    r"(?:"
    + _GREETING_SIGNATURE
    + r"|درود\s*بر\s*شما|درود\s*بر\s*سفیر(?:\s*سلامت)?"
    + r"|سلام\s*وقت\s*بخیر(?:\s*،?\s*همکار\s*(?:گرامی|محترم))?"
    + r"|سلام\s*و\s*درود(?:\s*بر\s*شما)?|با\s*سلام(?:\s*و\s*احترام)?"
    + r"|همکار\s*(?:گرامی|محترم)"
    + r")(?!" + _LETTER + r")"
    + r"\s*[،,:;؛\.\-–—]*\s*"
)

# A line that *starts* with one of these is boilerplate from end to end: the
# closer/annotation is never followed by answer content worth keeping.
_WHOLE_LINE_BOILERPLATE = [
    # CTA closers ("اگر خواستید، می‌تونم بر اساس ... تنظیم کنم 🌱").
    re.compile(
        r"^\s*اگر\s*(?:خواستید|خواستی|بخواید|بخواین"
        r"|تمایل\s*داشته\s*باشید|مایل\s*بودید|ما\s*مایل\s*بودید)"
    ),
    # Internal **trainer** annotations accidentally stored in the answer cell.
    # Deliberately narrow: a real answer may legitimately open with
    # `نیاز به ارسال آزمایشات نیست. راهکاری که ...` (a 400+ char answer was
    # deleted by a broader `نیاز به (نگارش|پیشنهاد|ارسال)` rule), so the verb is
    # pinned to `نگارش` and a negative tail is required.
    re.compile(r"^\s*(?:نیاز|لازم)\s*به\s*نگارش\b[^\n]{0,40}?(?:نمی\s*باشد|نمیباشد|نیست)"),
    # Ticket acknowledgements ("تیکت شما ... انتقال داده شده است"). A line that
    # opens with this never carries answer content.
    re.compile(r"^\s*تیکت\s*شما"),
]

# Removed *wherever* they appear; the rest of the line is preserved.
_SEGMENT_PATTERNS = [
    # "پاسخ در تیکت قبلی ارسال شد" notes.
    re.compile(r"در\s*تیکت\s*(?:قبلی|[\d\u06F0-\u06F9]+)\s*پاسخ[^\.\n]*\.?"),
    re.compile(r"پاسخ\s*(?:سوال\s*)?شما\s*در\s*تیکت\s*قبلی[^\.\n]*\.?"),
    re.compile(r"در\s*تیکت\s*قبلی\s*پاسخ\s*(?:ارسال\s*شد|داده\s*شد)[^\.\n]*\.?"),
    # Referral fallbacks to the relevant department / ticket system.
    re.compile(r"با\s*شماره\s*تیکت\s*[\d\u06F0-\u06F9]+"),
    re.compile(r"تیکت\s*شماره\s*[\d\u06F0-\u06F9]+"),
    re.compile(r"درخواست\s*شما\s*به\s*واحد\s*مربوطه[^\.\n]*\.?"),
    re.compile(r"تیکت\s*شما\s*به\s*واحد\s*مربوطه[^\.\n]*\.?"),
    re.compile(r"پیشنهاد\s*شما\s*به\s*واحد\s*مربوطه[^\.\n]*\.?"),
    re.compile(
        r"(?:به\s*)?واحد\s*مربوطه\s*(?:نیز\s*)?(?:انتقال|ارجاع)\s*(?:داده\s*)?"
        r"(?:شد|خواهد\s*شد|می\s*شود|شده\s*است)[^\.\n]*\.?"
    ),
    re.compile(r"اطلاعات\s*کافی\s*در\s*پایگاه\s*دانش[^\.\n]*\.?"),
    re.compile(r"متاسفانه\s*اطلاعات\s*کافی[^\.\n]{0,80}?(?:یافت\s*نشد|در\s*دسترس\s*نیست|وجود\s*ندارد)[^\.\n]*\.?"),
    re.compile(r"اطلاعات\s*دقیقی\s*در\s*دسترس\s*نیست[^\.\n]*\.?"),
    # Signature closers.
    re.compile(r"با\s*تشکر،?\s*پرشین\s*وی\.?"),
    re.compile(r"باتشکر\s*پرشین\s*وی\.?"),
]


def _strip_boilerplate_from_line(line: str) -> Tuple[str, bool]:
    """Return ``(sanitized_line, was_changed)`` for a single line.

    Segment removal and greeting removal are applied as a **fixpoint loop**:
    removing a signature can expose a greeting that was not at the line start
    (`با تشکر پرشین وی درود بر شما سفیر سلامت شرکت پرشین وی`). Stripping the
    greeting only once, before the segments, left that remnant behind and made
    the result non-idempotent.
    """
    changed = False
    working = line

    for _ in range(5):
        progressed = False

        # 1) Inline referral / ticket / signature segments.
        for pattern in _SEGMENT_PATTERNS:
            working, count = pattern.subn(" ", working)
            if count:
                changed = True
                progressed = True

        # 2) Leading greeting / vocative, applied repeatedly so that stacked
        #    forms are all consumed (`… پرشین وی همکار گرامی پیشنهاد…`).
        while True:
            candidate = _LEADING_JUNK.sub("", working)
            match = _GREETING_PREFIX.match(candidate)
            if not match:
                break
            working = candidate[match.end():]
            changed = True
            progressed = True

        if not progressed:
            break

    if not changed:
        return line, False

    working = _WHITESPACE_RUN.sub(" ", working)
    working = _EMOJI.sub("", working)
    working = _LEADING_JUNK.sub("", working).strip()

    # 3) Nothing but punctuation/emoji is left -> the line was pure boilerplate.
    if not _HAS_CONTENT.search(working):
        return "", True

    return working, True


def _is_whole_line_boilerplate(line: str) -> bool:
    """True when the entire line is boilerplate (drop it outright).

    Blank lines are reported as ``False``: they carry no boilerplate and are
    handled by :func:`_collapse_blank_lines`, so they must not inflate the
    "lines removed" metric.
    """
    stripped = _EMOJI.sub("", line).strip()
    if not stripped:
        return False
    return any(pattern.match(stripped) for pattern in _WHOLE_LINE_BOILERPLATE)


def _collapse_blank_lines(lines: List[str]) -> str:
    """Collapse runs of blank lines into a single separator, then trim.

    A blank line is "pending" until a non-blank line follows it: keeping only
    *trailing* blanks would emit `…\n\nAnswer:` on the first pass and `…\nAnswer:`
    on the second (the dropped whole-boilerplate line leaves the blank behind),
    which would break idempotency.
    """
    out: List[str] = []
    pending_blank = False
    for line in lines:
        if line.strip():
            if pending_blank and out:
                out.append("")
            pending_blank = False
            out.append(line)
        else:
            pending_blank = True
    return "\n".join(out).strip()


def sanitize_chunk(text: str) -> Tuple[str, int]:
    """Strip Excel Q&A boilerplate from ``text``.

    Returns ``(sanitized_text, removed_line_count)``. Whole boilerplate lines are
    dropped; lines that mix boilerplate with answer content keep the content.
    Idempotent: re-running over the output is a no-op.
    """
    if not text:
        return "", 0

    out_lines: List[str] = []
    removed_lines = 0

    for raw_line in text.splitlines():
        # Blank/whitespace-only lines are forwarded so `_collapse_blank_lines`
        # can decide the separator runs; they are not boilerplate and must not
        # be counted. Dropping them here made pass 1 emit `a\n\nb` and pass 2
        # emit `a\nb` (idempotency break).
        if not raw_line.strip():
            out_lines.append("")
            continue

        new_line, changed = _strip_boilerplate_from_line(raw_line)
        # The whole-line test runs on the post-greeting remainder, so that
        # `با سلام همکار گرامی تیکت شما پاسخ داده شده است` is recognised as the
        # ticket acknowledgement it becomes once the greeting is gone. Testing
        # the raw line instead left that form behind for one extra pass and
        # broke idempotency.
        if not new_line or _is_whole_line_boilerplate(new_line):
            removed_lines += 1
            continue
        if changed:
            removed_lines += 1
        out_lines.append(new_line)

    return _collapse_blank_lines(out_lines), removed_lines


def sanitize_documents(docs: List[Any]) -> List[Any]:
    """Sanitize chunk ``page_content`` in a copy of the list (new Document objects).

    Ranking, order and metadata are preserved verbatim; only ``page_content`` is
    rewritten. Chunks left with less than :data:`MIN_CONTENT_CHARS` characters of
    content are dropped (greeting-only / referral-only rows).
    """
    if not docs:
        return []

    from langchain_core.documents import Document

    sanitized: List[Any] = []
    dropped = 0
    removed_lines_total = 0

    for doc in docs:
        original = getattr(doc, "page_content", "") or ""
        metadata = dict(doc.metadata) if isinstance(getattr(doc, "metadata", None), dict) else {}
        cleaned, removed_lines = sanitize_chunk(original)
        removed_lines_total += removed_lines

        if len(cleaned.strip()) < MIN_CONTENT_CHARS:
            dropped += 1
            continue

        sanitized.append(Document(page_content=cleaned, metadata=metadata))

    logger.info(
        "sanitizer: %d docs, %d boilerplate lines removed, %d docs dropped",
        len(docs),
        removed_lines_total,
        dropped,
    )
    return sanitized
