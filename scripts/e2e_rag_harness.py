"""End-to-end RAG harness for the Persian Way consultation bot.

Ground truth
------------
`docs/product-qa-tickets.xls` (STEP-0 DISCOVERY, measured not assumed):

  * the file is a ZIP container (magic ``50 4B 03 04``) i.e. really an XLSX that
    was renamed to ``.xls``; it is read with ``engine="openpyxl"``.
  * one sheet only, name ``پاسخ دار ها``.
  * shape ``8918 x 3``; the real column names are exactly ``Title``,
    ``Question``, ``Answer``. There is NO product / category / id column, so
    ``Title`` is the only topical label available.

What the local knowledge base actually contains (measured)
----------------------------------------------------------
Two different vector stores exist on disk and they do NOT agree:

  * ``vectordb/openrouter-openai_text-embedding-3-small-1536`` ->
    collection ``langchain``, 1444 documents, of which 1400 carry
    ``source_type="excel_qa"`` with ``row_number`` spanning 2..1405 (1400 unique
    values). Those 1400 rows are exactly pandas rows 0..1399 of the sheet, i.e.
    the XLS IS embedded -- but only its first 1400 of 8918 rows, and only in
    this store.
  * The store the *current* code resolves is a different one. With
    ``STORAGE_ROOT=/knowledgebase`` (from ``.env``) and
    ``OPENROUTER_API_KEY`` set, ``DocumentProcessor`` picks provider
    ``openrouter`` / model ``baai/bge-m3`` / dim 1024, so
    ``persist_directory`` = ``<STORAGE_ROOT>/vectordb/openrouter-baai_bge-m3-1024``
    and ``get_vector_store()`` asks for collection ``knowledge_base``.
    That directory only holds a collection named ``langchain`` (1 document), so
    ``knowledge_base`` is created empty on demand: with the committed local
    ``.env`` the XLS rows are NOT reachable through the runtime retrieval path.

Because of that gap this harness never assumes a pre-existing index. It performs
a CONTROLLED, RUN-LOCAL ingest of an explicitly bounded slice of the XLS and
tests that slice only. The domain is printed at startup and written into the
report:

    test domain = sheet ``پاسخ دار ها``, pandas rows ``[0, N)``, N = --limit

Nothing outside the run directory is written, and no remote/production endpoint
is contacted: retrieval is local Chroma + local BM25 only.

Layer separation
----------------
Each probe is executed through the *same* methods the production code uses, in
the same order, but the intermediate values are captured instead of discarded:

    1. Context     ChatService.detect_query_intent()        -> context_state / resolved_query
    2. Rewriting   KnowledgeBaseService._retrieve_context() -> rewritten_query / all_queries
    3. Retrieval     ... same call -> docs_with_scores / sources / gt_rank
    4. Gate          confidence_score vs knowledge_base_confidence_threshold
    5. Generation  KnowledgeBaseService.stream_answer_from_context()

``chat_service.py:1597`` and ``chat_service.py:1631`` do exactly steps 2..5, so
the harness exercises the real pipeline while keeping every intermediate value
available for blame attribution.

Usage
-----
    python scripts/e2e_rag_harness.py --limit 5
    python scripts/e2e_rag_harness.py --limit 150 --variants verbatim,paraphrase,typo,elliptical --multiturn
    python scripts/e2e_rag_harness.py --validate-units        # offline, no API calls
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# The Windows console here defaults to cp1252; every Persian string this harness
# prints would raise UnicodeEncodeError without this. ``errors="replace"`` keeps
# a bad glyph from killing a whole run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-tty streams
        pass

DEFAULT_XLS = os.path.join(PROJECT_ROOT, "docs", "product-qa-tickets.xls")
DEFAULT_SHEET = "پاسخ دار ها"

# --------------------------------------------------------------------------- #
# Persian text normalisation (validated against real XLS samples, see
# ``validate_dose_detector`` below).
# --------------------------------------------------------------------------- #

_FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_DIGIT_MAP: Dict[int, str] = {ord(c): str(i) for i, c in enumerate(_FA_DIGITS)}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate(_AR_DIGITS)})

# Harakat / tatweel / bidi marks are dropped so units match regardless of
# vocalisation and of the RTL/LTR control characters present in the XLS cells.
_HARAKAT_RE = re.compile(r"[\u064B-\u0652\u0640\u200e\u200f\u202a-\u202e]")


def normalize_digits(text: str) -> str:
    """Map Persian (U+06F0..) and Arabic-Indic (U+0660..) digits onto ASCII."""
    return (text or "").translate(_DIGIT_MAP)


def normalize_text(text: str) -> str:
    """Fold Persian orthography variants so lexical comparison is meaningful.

    ZWNJ is removed (not replaced by a space) which deliberately JOINS compound
    units such as ``میلی‌لیتر`` -> ``میلیلیتر``.
    """
    t = text or ""
    t = t.replace("\u200c", "")
    t = normalize_digits(t)
    t = _HARAKAT_RE.sub("", t)
    t = (t.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه")
          .replace("ۀ", "ه").replace("أ", "ا").replace("إ", "ا").replace("ٱ", "ا"))
    t = re.sub(r"[\u060C\u061B]", ",", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


# Number WORDS -> numeric. Fractions are normalised as required by the rubric
# ("نیم", "یک‌سوم"). ZWNJ is already removed by the time these are looked up, so
# "یک‌سوم" arrives here as "یکسوم".
_NUMBER_WORDS: Dict[str, float] = {
    "نیم": 0.5, "یک": 1.0, "دو": 2.0, "سه": 3.0, "چهار": 4.0, "پنج": 5.0,
    "شش": 6.0, "هفت": 7.0, "هشت": 8.0, "نه": 9.0, "ده": 10.0, "یازده": 11.0,
    "دوازده": 12.0, "پانزده": 15.0, "بیست": 20.0, "سی": 30.0, "چهل": 40.0,
    "پنجاه": 50.0, "شصت": 60.0, "صد": 100.0, "یکصد": 100.0,
}
_FRACTION_WORDS: Dict[str, float] = {
    "یکسوم": 1.0 / 3.0, "دوسوم": 2.0 / 3.0, "سهچهارم": 0.75,
    "یکچهارم": 0.25, "یکدهم": 0.1, "نیم": 0.5,
}

# NOTE: order matters -- the first alternative that matches at a position wins,
# so compound units must be listed before their head noun ("میلیلیتر" before
# "لیتر") or "2 میلیلیتر" would be misread as "2 لیتر".
_UNIT_SPECS: List[Tuple[str, str]] = [
    ("میلیلیتر", r"میلیلیتر|میللیتری|سیسی|\bcc\b|\bml\b|\bmillilit"),
    ("کیلوگرم-در-هکتار", r"کیلوگرم\s*در\s*هکتار|کیلو\s*در\s*هکتار|kg\s*/\s*ha"),
    ("کیلوگرم", r"کیلوگرم|کیلوگرمی|کیلو\b|\bkg\b|\bkilogram"),
    ("میلیگرم", r"میلیگرم|میلیگرام|\bmg\b"),
    ("گرم", r"گرم\b|\bgram|\bg\b(?![a-z])"),
    ("لیتر", r"لیتر|\blitre|\bliters?\b"),
    ("قاشق-مرباخوری", r"قاشقمرباخوری|قاشق\s*مرباخوری|مرباخوری"),
    ("قاشق-غذاخوری", r"قاشقغذاخوری|قاشق\s*غذاخوری|غذاخوری"),
    ("قاشق-چایخوری", r"قاشقچایخوری|قاشق\s*چایخوری|چایخوری"),
    ("قاشق", r"قاشق"),
    ("قطره", r"قطره|قطرات"),
    ("پیمانه", r"پیمانه"),
    ("کپسول", r"کپسول"),
    ("قرص", r"قرص"),
    ("ساشه", r"ساشه"),
    ("اسپری", r"اسپری"),
    ("درصد", r"درصد|\b%\b"),
    ("درجه", r"درجه"),
    ("واحد", r"واحد"),
    ("وعده", r"وعده"),
    ("نوبت", r"نوبت"),
    ("بار", r"بار(?![ا-ی])"),
    ("دقیقه", r"دقیقه"),
    ("ساعت", r"ساعت"),
    ("روز", r"روز(?![ا-ی])"),
    ("هفته", r"هفته"),
    ("ماه", r"ماه(?![ا-ی])"),
    ("سال", r"سال(?![ا-ی])"),
    ("هکتار", r"هکتار"),
    ("مترمربع", r"مترمربع|متر\s*مربع"),
    ("سانتیمتر", r"سانتیمتر|سانتی\s*متر|\bcm\b"),
    ("متر", r"متر\b|\bmeters?\b"),
    ("ppm", r"\bppm\b"),
]

_UNIT_RE = re.compile(
    "|".join(f"(?P<u{i}>{pat})" for i, (_, pat) in enumerate(_UNIT_SPECS)), re.I
)
_UNIT_NAME_BY_GROUP = {f"u{i}": name for i, (name, _) in enumerate(_UNIT_SPECS)}

# A quantity immediately preceding a unit, optionally a "A تا B" / "A-B" range.
_NUM_TOKEN = r"\d+(?:[.,]\d+)?"
_RANGE_TAIL_RE = re.compile(
    rf"(?P<a>{_NUM_TOKEN})(?:\s*(?:تا|-|–|_)\s*(?P<b>{_NUM_TOKEN}))?\s*$"
)
_UNIT_WINDOW = 26          # characters of left-context inspected for a quantity


def _to_float(token: str) -> Optional[float]:
    """Parse an ASCII numeric token; Arabic decimal separator is tolerated."""
    try:
        return float(token.replace(",", "."))
    except (TypeError, ValueError):
        return None


def extract_quantities(text: str) -> Dict[str, Set[float]]:
    """Extract ``{unit -> {value, ...}}`` from raw Persian text.

    Only *unit-bearing* quantities are returned on purpose: bare integers are
    overwhelmingly dates, step numbers or enumerations in this corpus and
    treating them as doses produces false S1 hits.

    Handles, in order of precedence:
      * ASCII / Persian / Arabic-Indic digits ("۱۰۰۰", "١٠٠٠", "1000"),
      * decimal separators "." and "," ("۵٫۵"),
      * ranges ("۵ تا ۱۰ دقیقه", "5-10 ml") -- both bounds are recorded,
      * fraction and number WORDS directly in front of a unit
        ("نیم قاشق", "یکسوم پیمانه").
    """
    normalized = normalize_text(text)
    if not normalized:
        return {}

    found: Dict[str, Set[float]] = defaultdict(set)

    for match in _UNIT_RE.finditer(normalized):
        unit = _UNIT_NAME_BY_GROUP.get(match.lastgroup or "")
        if not unit:
            continue
        left = normalized[max(0, match.start() - _UNIT_WINDOW):match.start()]
        tail = _RANGE_TAIL_RE.search(left)
        if tail:
            for key in ("a", "b"):
                raw = tail.group(key)
                if raw:
                    value = _to_float(raw)
                    if value is not None:
                        found[unit].add(value)
            continue

        # Word-based quantities: look at the last token(s) before the unit.
        # The candidate is stripped of markdown/punctuation first, because the
        # generated answers bold their numbers ("به مدت **یک هفته**") and a raw
        # token split would compare "**یک" instead of "یک".
        words = re.findall(r"[^\s,]+", left)
        if words:
            last = re.sub(r"[^\w\u0600-\u06FF]+", "", words[-1])
            if last in _FRACTION_WORDS:
                found[unit].add(_FRACTION_WORDS[last])
            elif last in _NUMBER_WORDS:
                found[unit].add(_NUMBER_WORDS[last])

    return {unit: values for unit, values in found.items() if values}


def format_quantities(quantities: Dict[str, Set[float]]) -> str:
    """Stable, human-readable rendering used in logs / reports."""
    if not quantities:
        return "-"
    parts = []
    for unit in sorted(quantities):
        vals = ", ".join(
            str(int(v)) if float(v).is_integer() else f"{v:g}" for v in sorted(quantities[unit])
        )
        parts.append(f"{unit}={vals}")
    return "; ".join(parts)


def find_dose_contradictions(
    gold_text: str, answer_text: str
) -> List[Dict[str, Any]]:
    """S1 core: quantities the answer asserts that the ground truth contradicts.

    A quantity is a contradiction when the answer uses a unit that the ground
    truth also uses, but with a value that never appears in the ground truth for
    that unit. Subsets are allowed (GT "5-10 دقیقه" vs answer "5 دقیقه" is NOT a
    contradiction) precisely to keep the detector conservative.
    """
    gold = extract_quantities(gold_text)
    got = extract_quantities(answer_text)
    problems: List[Dict[str, Any]] = []
    for unit, values in got.items():
        if unit not in gold:
            continue
        extra = {v for v in values if v not in gold[unit]}
        if extra:
            problems.append({
                "unit": unit,
                "gold_values": sorted(gold[unit]),
                "answer_values": sorted(values),
                "contradicting_values": sorted(extra),
            })
    return problems


def find_dropped_doses(gold_text: str, answer_text: str) -> List[Dict[str, Any]]:
    """S1 secondary: the ground truth prescribes a quantity and the answer has none.

    Reported separately from a *contradiction* because the failure mode (silent
    omission) is different from an inverted number.
    """
    gold = extract_quantities(gold_text)
    got = extract_quantities(answer_text)
    missing: List[Dict[str, Any]] = []
    for unit, values in gold.items():
        if unit not in got:
            missing.append({"unit": unit, "gold_values": sorted(values)})
    return missing


# --------------------------------------------------------------------------- #
# Refusal / style / coverage detectors
# --------------------------------------------------------------------------- #

# Verbatim from `chat_service.py:1201-1210` so the harness and production agree
# on what "the bot refused" means.
REFERRAL_INDICATORS: Tuple[str, ...] = (
    "متاسفانه اطلاعات",
    "متأسفانه اطلاعات کافی",
    "متأسفانه",
    "متاسفانه",
    "اطلاعات کافی در دسترس نیست",
    "اطلاعات کافی درباره",
    "اطلاعات کافی در مورد",
    "اطلاعات کافی برای پاسخ",
)

# Trainer greeting/signature that the LLM reproduces verbatim -> S4 style leak.
# These target the SIGN-OFF / GREETING pattern, not a bare brand mention: a
# grounded answer legitimately writes things like
#   "محصولی با نام «پرشین وی» برای درمان شپش معرفی نشده است"
# and flagging that as a style leak was a measured false positive during review.
_STYLE_GREETING = ("درود بر شما", "درود برشما", "سلام و درود", "درود بر")
_STYLE_SIGNATURE = (
    "سفیرسلامت", "سفیر سلامت",
    "با تشکر پرشین", "با تشکر پرشینوی", "با تشکر پرشین وی", "باتشکر پرشین",
    "با احترام", "ارادتمند",
)
_STYLE_CTA = (
    "برای سفارش", "جهت خرید", "برای خرید", "میتوانید سفارش", "ثبت سفارش",
    "سایت ما", "وبسایت", "اینستاگرام", "واتساپ", "پشتیبانی ما",
)
_EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF" "\U00002700-\U000027BF" "\u2764\u2600-\u26FF" "]",
    flags=re.UNICODE,
)


def is_refusal(answer: str, referral_message: str = "") -> bool:
    """True when the answer is a handoff/refusal rather than a real answer.

    An EMPTY answer is deliberately NOT a refusal: it means the generation stage
    produced nothing (the gate was open, so the referral text was never used).
    That is a different defect with a different blame, so it is reported as
    ``empty_answer`` instead of being folded into S2.
    """
    if not answer or not answer.strip():
        return False
    if referral_message and normalize_text(referral_message)[:60] in normalize_text(answer):
        return True
    return any(ind in answer for ind in REFERRAL_INDICATORS)


def detect_style_leaks(answer: str, is_refusal_answer: bool = False) -> List[str]:
    """S4: trainer greeting / signature / unsolicited CTA / emoji in the output."""
    if not answer:
        return []
    leaks: List[str] = []
    for marker in _STYLE_GREETING:
        if marker in answer:
            leaks.append(f"greeting:{marker}")
    for marker in _STYLE_SIGNATURE:
        if marker in answer:
            leaks.append(f"signature:{marker}")
    if not is_refusal_answer:
        for marker in _STYLE_CTA:
            if marker in answer:
                leaks.append(f"cta:{marker}")
    emojis = _EMOJI_RE.findall(answer)
    if emojis:
        leaks.append("emoji:" + "".join(sorted(set(emojis))[:5]))
    return leaks


# Persian function words -- excluded from key-term coverage on both sides.
_STOPWORDS = {
    "و", "در", "به", "از", "که", "این", "را", "با", "است", "برای", "آن", "یک",
    "می", "شود", "شده", "شدن", "های", "ها", "هم", "یا", "تا", "بر", "بود",
    "خود", "هر", "روی", "بین", "دو", "اگر", "همچنین", "باید", "دارد", "دارند",
    "کنید", "کنن", "کند", "کرد", "میباشد", "میباشد", "نیز", "ولی", "اما",
    "رو", "چه", "چی", "کجا", "چطور", "چگونه", "لطفا", "لطفاً", "بله", "خیر",
    "دادن", "گیرد", "کنیم", "شوند", "باشد", "هست", "نیست", "مورد", "قابل",
    "مثل", "مانند", "روزانه", "سپس", "بعد", "قبل", "بیشتر", "کمتر", "خیلی",
}


def _terms(text: str) -> Set[str]:
    tokens = re.split(r"[^\w\u0600-\u06FF]+", normalize_text(text))
    return {t for t in tokens if len(t) >= 4 and t not in _STOPWORDS and not t.isdigit()}


def build_boilerplate_terms(answers: Iterable[str], min_df_ratio: float = 0.30) -> Set[str]:
    """Data-driven boilerplate: terms appearing in >= ``min_df_ratio`` of answers.

    Every trainer answer in this corpus ends with the same greeting/signature
    ("درود بر شما سفیرسلامت شرکت پرشین وی ... با تشکر پرشین وی"), so requiring the
    model to reproduce those tokens would make coverage meaningless. The
    threshold is measured on the corpus rather than hard-coded.
    """
    answers = [a for a in answers if isinstance(a, str) and a.strip()]
    if not answers:
        return set()
    df = Counter()
    for answer in answers:
        df.update(_terms(answer))
    cutoff = max(2, int(len(answers) * min_df_ratio))
    return {term for term, count in df.items() if count >= cutoff}


def fact_coverage(gold: str, answer: str, boilerplate: Optional[Set[str]] = None) -> float:
    """Lexical coverage of the ground-truth key terms.

    HONEST LIMITATION: this is a LEXICAL metric. It has inherent false positives
    (paraphrase, morphology, spelling variants) and can only ever be a screen,
    never a verdict -- which is why ``NOTE_COVERAGE_THRESHOLD`` is deliberately
    conservative and every S3 row is listed for manual review in the report.
    """
    boilerplate = boilerplate or set()
    gold_terms = _terms(gold) - boilerplate
    if not gold_terms:
        return 1.0
    got_terms = _terms(answer)
    return len(gold_terms & got_terms) / len(gold_terms)


# Conservative on purpose: a LOWER threshold means fewer false positives from the
# purely LEXICAL coverage screen. Every S3 row is additionally listed under
# `s3_rows_for_manual_review` in the summary JSON for human inspection, because
# paraphrase and morphology can always defeat a lexical metric.
NOTE_COVERAGE_THRESHOLD = 0.35


# --------------------------------------------------------------------------- #
# Rubric (S1..S4 / PASS) and blame attribution
# --------------------------------------------------------------------------- #

# Variants whose refusal is CORRECT behaviour, not a defect: on an elliptical
# query the system legitimately lacks enough signal, and on a topic switch a
# refusal beats contaminating the answer with the previous subject. Counting
# these as S2 is exactly the false positive this control measures.
REFUSAL_OK_VARIANTS = {"elliptical"}


def classify_record(
    answer: str,
    gold_answer: str,
    refused: bool,
    gt_rank: int,
    coverage: float,
    style_leaks: Sequence[str],
    refusal_ok: bool = False,
) -> Dict[str, Any]:
    """Assign tier (S1/S2/S3/S4/Pass) plus the raw evidence behind it."""
    contradictions = find_dose_contradictions(gold_answer, answer) if answer else []
    dropped = find_dropped_doses(gold_answer, answer) if answer else []
    empty_answer = not (answer or "").strip()

    tiers: List[str] = []
    reasons: List[str] = []
    if empty_answer:
        reasons.append("empty_answer:generation produced no text")

    # S1 -- unit-aware numeric contradiction with the ground truth.
    if contradictions:
        tiers.append("S1")
        reasons.append("dose_contradiction:" + "; ".join(
            f"{c['unit']} {c['gold_values']}->{c['contradicting_values']}" for c in contradictions
        ))
    elif dropped and not refused and coverage >= NOTE_COVERAGE_THRESHOLD:
        # Answered substantively but silently lost the prescription.
        tiers.append("S1")
        reasons.append("dose_dropped:" + "; ".join(f"{d['unit']}{d['gold_values']}" for d in dropped))

    # S2 -- referral while the source row IS retrievable (and refusal not expected).
    if refused and gt_rank > 0 and not refusal_ok:
        tiers.append("S2")
        reasons.append(f"false_referral:gt_rank={gt_rank}")
    elif refused and gt_rank > 0 and refusal_ok:
        reasons.append("refusal_allowed_by_control")

    # S3 -- answered, but lost the ground truth's facts (lexical screen).
    if not refused and coverage < NOTE_COVERAGE_THRESHOLD:
        tiers.append("S3")
        reasons.append(f"low_coverage:{coverage:.2f}")
    elif not refused and gt_rank == -1:
        reasons.append("answered_from_foreign_document")

    # S4 -- style / CTA leakage.
    if style_leaks:
        tiers.append("S4")
        reasons.append("style:" + "; ".join(style_leaks))

    primary = next((t for t in ("S1", "S2", "S3", "S4") if t in tiers), "Pass")
    return {
        "tier": primary,
        "tier_all": ",".join(tiers) if tiers else "Pass",
        "tier_reasons": reasons,
        "dose_contradictions": contradictions,
        "dose_dropped": dropped,
        "coverage": round(coverage, 4),
    }


def error_overrides(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fields that replace rubric grading when a probe never produced evidence.

    A probe that raised (or hit the harness ceiling) has no answer to grade. Left
    to the normal rubric it can look benign: the referral fallback text is not a
    style leak and ``refused/gt_rank=-1`` matches no failure tier, so it lands in
    ``Pass``. That is how a hang is silently reported as healthy. Returns None
    when the probe completed.
    """
    if not record.get("error"):
        return None
    return {
        "tier": "Error",
        "tier_all": "Error",
        "tier_reasons": [record["error"]],
        "dose_contradictions": [],
        "dose_dropped": [],
        "coverage": 0.0,
        "style_leaks": [],
        "blame": "timeout" if record.get("timeout") else "error",
        "blame_note": record["error"],
    }


def attribute_blame(record: Dict[str, Any]) -> str:
    """Exactly one layer per failure, per the mission's attribution table."""
    tier = record.get("tier")
    refused = bool(record.get("refused"))
    gt_rank = int(record.get("gt_rank") or -1)
    confidence = record.get("confidence_score")
    threshold = record.get("threshold")

    # Context wins over everything else on multi-turn records: a lost or bled
    # subject is a context failure no matter what the answer looked like.
    if record.get("turn_kind") == "multiturn":
        if record.get("topic_retention") is False or record.get("context_bleed") is True:
            return "context"

    if tier == "S1" and (record.get("dose_contradictions") or record.get("dose_dropped")):
        return "generation"

    if refused:
        if gt_rank == -1:
            return "retrieval"
        if confidence is not None and threshold is not None and confidence < threshold:
            return "gate"
        return "generation"

    if gt_rank == -1:
        # Answered anyway: the text cannot have come from the right row.
        return "retrieval"

    # The right row was retrieved, the answer came back and was not a refusal --
    # so an S3 (facts lost) or S4 (style/CTA leaked) defect was produced by the
    # generation stage, not by retrieval.
    if tier in ("S3", "S4"):
        return "generation"

    return "none"


def refine_blame_with_variants(records: List[Dict[str, Any]]) -> None:
    """Second pass: re-attribute verbatim failures that only repro through rewriting.

    The mission's rule is narrow on purpose: ``rewriting`` is blamed only when a
    row was retrievable on the verbatim variant and the ranking collapsed on
    paraphrase/typo. Everything else keeps its stage-1 blame.
    """
    by_row: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_row[rec.get("row_id")].append(rec)

    for row_records in by_row.values():
        ranks = {r.get("variant"): r.get("gt_rank") for r in row_records}
        verbatim_rank = ranks.get("verbatim")
        verbatim_ok = isinstance(verbatim_rank, int) and verbatim_rank > 0
        if not verbatim_ok:
            continue
        for rec in row_records:
            variant = rec.get("variant")
            if variant not in ("paraphrase", "typo"):
                continue
            rank = rec.get("gt_rank")
            degraded = not isinstance(rank, int) or rank <= 0 or rank > verbatim_rank
            if degraded and rec.get("blame") in ("retrieval", "gate", "generation"):
                rec["blame"] = "rewriting"
                rec["blame_note"] = (
                    f"verbatim hit at rank {verbatim_rank}; {variant} degraded to "
                    f"{rank if rank else -1} -> the rewrite, not the index, lost it"
                )
                rec["tier"] = "S3" if rec.get("tier") == "Pass" else rec.get("tier")


# --------------------------------------------------------------------------- #
# Ground-truth loading
# --------------------------------------------------------------------------- #

class GroundTruthError(RuntimeError):
    """Raised when the workbook does not match the discovered schema."""


@dataclass
class GtRow:
    row_id: int              # pandas index (0-based)
    excel_row: int           # spreadsheet row (pandas index + 2)
    title: str
    question: str
    answer: str
    quantities: Dict[str, Set[float]] = field(default_factory=dict)


def load_ground_truth(xls_path: str, sheet: str, limit: int) -> Tuple[List[GtRow], Dict[str, Any]]:
    """Read the workbook and return the test domain, verifying the real schema."""
    import pandas as pd

    if not os.path.isfile(xls_path):
        raise GroundTruthError(f"ground-truth workbook not found: {xls_path}")

    xl = pd.ExcelFile(xls_path)
    sheet_names = list(xl.sheet_names)
    if sheet not in sheet_names:
        raise GroundTruthError(
            f"sheet {sheet!r} missing; workbook has {sheet_names!r}"
        )
    df = xl.parse(sheet)

    expected = ["Title", "Question", "Answer"]
    actual = [str(c) for c in df.columns]
    if actual != expected:
        raise GroundTruthError(
            f"unexpected columns: expected {expected!r}, found {actual!r}"
        )

    total_rows = len(df)
    effective_limit = total_rows if limit <= 0 else min(limit, total_rows)

    rows: List[GtRow] = []
    for idx in range(effective_limit):
        rec = df.iloc[idx]
        question = "" if pd.isna(rec["Question"]) else str(rec["Question"])
        answer = "" if pd.isna(rec["Answer"]) else str(rec["Answer"])
        title = "" if pd.isna(rec["Title"]) else str(rec["Title"])
        if not question.strip() or not answer.strip():
            continue
        rows.append(GtRow(
            row_id=idx,
            excel_row=idx + 2,
            title=title,
            question=question,
            answer=answer,
            quantities=extract_quantities(answer),
        ))

    meta = {
        "xls_path": os.path.abspath(xls_path),
        "sheet_names": sheet_names,
        "sheet_used": sheet,
        "columns": actual,
        "total_rows_in_sheet": total_rows,
        "domain_requested_rows": effective_limit,
        "domain_loaded_rows": len(rows),
        "domain_row_rule": "pandas rows [0, N) of the sheet, N = --limit (0 = all)",
    }
    return rows, meta


# --------------------------------------------------------------------------- #
# MANDATORY self-validation of the unit/number detector
# --------------------------------------------------------------------------- #
# The mission forbids shipping an untested regex. Every case below is either a
# REAL ground-truth string pulled from the workbook at runtime (cases 1-3, 7) or
# the canonical failure shape the rubric must catch (cases 4-6, 8-11).

def validate_dose_detector(rows: Sequence[GtRow]) -> List[Dict[str, Any]]:
    """Run the unit/number detector against real and canonical inputs.

    Returns one dict per check: ``{"case", "input", "observed", "expected", "ok"}``.
    ``main`` always executes this (it costs no API calls) and the results go into
    the summary JSON and the report as evidence that the detector was tested.
    """
    checks: List[Dict[str, Any]] = []

    def check(name: str, text: str, expected: Dict[str, Set[float]], got: Dict[str, Set[float]]):
        checks.append({
            "case": name,
            "input": (text[:120] + "...") if len(text) > 120 else text,
            "expected": format_quantities(expected),
            "observed": format_quantities(got),
            "ok": got == expected,
        })

    real = {r.row_id: r for r in rows}

    # --- 1-3: verbatim real answers from the workbook ------------------------ #
    if 0 in real:
        # Row 0 contains: "روزی ۲  مرباخوری", "به مدت ۵تا ۱۰ دقیقه" and
        # " .یک روز درمیان" -- the punctuation-prefixed number word is why
        # روز=1 is expected here.
        check("real row 0: روزی ۲ مرباخوری + ۵تا ۱۰ دقیقه + یک روز درمیان",
              real[0].answer,
              {"قاشق-مرباخوری": {2.0}, "دقیقه": {5.0, 10.0}, "روز": {1.0}},
              extract_quantities(real[0].answer))
    if 2 in real:
        check("real row 2: fraction word نیم ساعت",
              real[2].answer, {"ساعت": {0.5}}, extract_quantities(real[2].answer))
    if 12 in real:
        check("real row 12: number word روزی دو بار",
              real[12].answer, {"بار": {2.0}, "روز": {1.0}, "ساعت": {0.5, 1.0}},
              extract_quantities(real[12].answer))

    # --- 4-6: digit scripts, ranges, ZWNJ folding --------------------------- #
    check("Persian digits ۱۰۰۰ میلی‌لیتر",
          "روزی ۱۰۰۰ میلی‌لیتر آب", {"میلیلیتر": {1000.0}},
          extract_quantities("روزی ۱۰۰۰ میلی‌لیتر آب"))
    check("Arabic-Indic digits ١٠٠ لیتر",
          "مقدار ١٠٠ لیتر", {"لیتر": {100.0}}, extract_quantities("مقدار ١٠٠ لیتر"))
    check("range ۵ تا ۱۰ میلی‌لیتر keeps both bounds",
          "۵ تا ۱۰ میلی‌لیتر", {"میلیلیتر": {5.0, 10.0}},
          extract_quantities("۵ تا ۱۰ میلی‌لیتر"))

    # --- 7: contradiction on a REAL row ------------------------------------ #
    if 0 in real:
        mutated = real[0].answer.replace("۲  مرباخوری", "۲۰  مرباخوری")
        found = find_dose_contradictions(real[0].answer, mutated)
        checks.append({
            "case": "real row 0 with ۲ مرباخوری flipped to ۲۰ مرباخوری",
            "input": "mutated copy of row 0 answer",
            "expected": ">=1 contradiction on قاشق-مرباخوری",
            "observed": json.dumps(found, ensure_ascii=False) if found else "none",
            "ok": bool(found) and found[0]["unit"] == "قاشق-مرباخوری",
        })

    # --- 8-11: canonical shapes -------------------------------------------- #
    canon = find_dose_contradictions(
        "برای هر گلدان ۱۰۰۰ لیتر آب لازم است.",
        "برای هر گلدان ۱۰۰ لیتر آب لازم است.")
    checks.append({
        "case": "canonical 1000 لیتر vs 100 لیتر",
        "input": "GT «۱۰۰۰ لیتر» / answer «۱۰۰ لیتر»",
        "expected": ">=1 contradiction on لیتر",
        "observed": json.dumps(canon, ensure_ascii=False) if canon else "none",
        "ok": bool(canon) and canon[0]["unit"] == "لیتر",
    })

    dropped = find_dropped_doses("نیم ساعت قبل از حمام", "قبل از حمام استفاده کنید.")
    checks.append({
        "case": "dropped dose (GT نیم ساعت, answer has no quantity)",
        "input": "GT «نیم ساعت» / answer «قبل از حمام استفاده کنید.»",
        "expected": ">=1 dropped unit, 0 contradictions",
        "observed": f"dropped={len(dropped)} contradictions=0",
        "ok": bool(dropped),
    })

    subset = find_dose_contradictions("۵ تا ۱۰ دقیقه", "۵ دقیقه")
    checks.append({
        "case": "subset must NOT be a contradiction (5-10 دقیقه vs 5 دقیقه)",
        "input": "GT «۵ تا ۱۰ دقیقه» / answer «۵ دقیقه»",
        "expected": "0 contradictions",
        "observed": json.dumps(subset, ensure_ascii=False) if subset else "none",
        "ok": not subset,
    })

    # --- 12-13: style and refusal detectors --------------------------------- #
    leaks = detect_style_leaks("درود بر شما سفیرسلامت شرکت پرشین وی با تشکر پرشین وی 🌹")
    checks.append({
        "case": "style leak on real trainer boilerplate",
        "input": "درود بر شما سفیرسلامت ... با تشکر پرشین وی 🌹",
        "expected": "greeting + signature + emoji detected",
        "observed": "; ".join(leaks) or "none",
        "ok": any(x.startswith("greeting") for x in leaks)
        and any(x.startswith("signature") for x in leaks)
        and any(x.startswith("emoji") for x in leaks),
    })
    referral = "متأسفانه، اطلاعات کافی در پایگاه دانش برای پاسخ به این سؤال وجود ندارد."
    checks.append({
        "case": "refusal detected via chat_service referral indicators",
        "input": referral,
        "expected": "refused=True",
        "observed": str(is_refusal(referral)),
        "ok": is_refusal(referral),
    })
    checks.append({
        "case": "normal answer must NOT look like a refusal",
        "input": "روزی دو مرباخوری داخل آب میوه حل کنید.",
        "expected": "refused=False",
        "observed": str(is_refusal("روزی دو مرباخوری داخل آب میوه حل کنید.")),
        "ok": not is_refusal("روزی دو مرباخوری داخل آب میوه حل کنید."),
    })

    # --- 14-17: regression cases discovered by REVIEWING real harness output -- #
    # Bug 1: generated answers bold their numbers, so the token before the unit
    # was "**یک" and a real dose was reported as dropped.
    check("markdown-bolded number word (**یک هفته**) must be read as هفته=1",
          "به مدت **یک هفته** این کار انجام شود.",
          {"هفته": {1.0}}, extract_quantities("به مدت **یک هفته** این کار انجام شود."))

    # Bug 2: a legitimate brand mention was flagged as a trainer signature.
    brand = "در اسناد موجود، محصولی با نام **«پرشین وی»** برای درمان شپش معرفی نشده است."
    brand_leaks = detect_style_leaks(brand)
    checks.append({
        "case": "legit brand mention must NOT be an S4 style leak",
        "input": brand[:110],
        "expected": "no style leak",
        "observed": "; ".join(brand_leaks) or "none",
        "ok": not brand_leaks,
    })

    # Bug 3: an empty generation was classified as a false human referral (S2).
    checks.append({
        "case": "empty answer is not a refusal (it is an empty generation)",
        "input": "''",
        "expected": "refused=False",
        "observed": str(is_refusal("")),
        "ok": not is_refusal(""),
    })

    empty_tier = classify_record(
        answer="", gold_answer="نیم ساعت قبل از حمام", refused=False, gt_rank=1,
        coverage=0.0, style_leaks=[], refusal_ok=False)
    checks.append({
        "case": "empty answer with gt_rank=1 classifies as S3 (not S2)",
        "input": "answer='' gt_rank=1 refused=False",
        "expected": "tier=S3",
        "observed": empty_tier["tier"],
        "ok": empty_tier["tier"] == "S3",
    })

    # Bug 4: a probe that hit the harness ceiling was graded by the normal rubric
    # and reported as ``Pass`` -- a hang looked healthy in the summary.
    errored = {"error": "TimeoutError: probe exceeded 120s", "timeout": True,
               "answer": "برو بیرون", "refused": True, "gt_rank": -1,
               "confidence_score": None}
    plain_ok = {"answer": "روزی دو مرباخوری", "refused": False, "gt_rank": 1}
    overrides = error_overrides(errored)
    checks.append({
        "case": "timed-out probe gets tier=Error, not Pass",
        "input": "error='TimeoutError' timeout=True refused=True gt_rank=-1",
        "expected": "tier=Error blame=timeout",
        "observed": f"tier={overrides['tier']} blame={overrides['blame']}",
        "ok": overrides["tier"] == "Error" and overrides["blame"] == "timeout",
    })
    checks.append({
        "case": "successful probe is untouched by the error override",
        "input": "no error key",
        "expected": "None",
        "observed": str(error_overrides(plain_ok)),
        "ok": error_overrides(plain_ok) is None,
    })
    failure = {"error": "ValueError: boom", "answer": "x", "refused": False}
    checks.append({
        "case": "non-timeout failure is blamed on error, not timeout",
        "input": "error='ValueError: boom'",
        "expected": "blame=error",
        "observed": error_overrides(failure)["blame"],
        "ok": error_overrides(failure)["blame"] == "error",
    })

    # Bug 5: the 999 "row not retrieved" sentinel was averaged in with real rank
    # differences, so elliptical reported a meaningless delta of 378.25.
    synth = [
        {"row_id": 0, "variant": "verbatim", "gt_rank": 1, "refused": False},
        {"row_id": 0, "variant": "paraphrase", "gt_rank": 1, "refused": False},
        {"row_id": 0, "variant": "elliptical", "gt_rank": -1, "refused": False},
        {"row_id": 1, "variant": "verbatim", "gt_rank": 1, "refused": False},
        {"row_id": 1, "variant": "paraphrase", "gt_rank": 3, "refused": False},
        {"row_id": 1, "variant": "elliptical", "gt_rank": 2, "refused": False},
    ]
    rw = rewriting_metrics(synth)
    checks.append({
        "case": "lost-row sentinel is excluded from the mean rank delta",
        "input": "elliptical ranks -1 (lost) and 2 against verbatim rank 1",
        "expected": "mean=1.0 comparable=1 lost=1",
        "observed": f"mean={rw['elliptical_mean_rank_delta']} "
                    f"comparable={rw['elliptical_comparable_rows']} "
                    f"lost={rw['elliptical_lost_count']}",
        "ok": (rw["elliptical_mean_rank_delta"] == 1.0
               and rw["elliptical_comparable_rows"] == 1
               and rw["elliptical_lost_count"] == 1),
    })
    checks.append({
        "case": "paraphrase regression measured as a real rank delta",
        "input": "paraphrase ranks 1 and 3 against verbatim rank 1",
        "expected": "mean=1.0 regressed=1",
        "observed": f"mean={rw['paraphrase_mean_rank_delta']} "
                    f"regressed={rw['paraphrase_regressed_count']}",
        "ok": (rw["paraphrase_mean_rank_delta"] == 1.0
               and rw["paraphrase_regressed_count"] == 1),
    })

    return checks


def print_validation_table(checks: Sequence[Dict[str, Any]]) -> bool:
    """Print the detector self-test; return True when every case passed."""
    print("\n" + "=" * 108)
    print("UNIT / NUMBER DETECTOR SELF-VALIDATION (offline, real XLS samples)")
    print("=" * 108)
    print(f"{'#':<3} {'case':<56} {'expected':<26} {'observed':<26} ok")
    print("-" * 108)
    for i, c in enumerate(checks, 1):
        print(f"{i:<3} {c['case'][:55]:<56} {str(c['expected'])[:25]:<26} "
              f"{str(c['observed'])[:25]:<26} {'PASS' if c['ok'] else 'FAIL'}")
    print("-" * 108)
    failed = [c for c in checks if not c["ok"]]
    print(f"{len(checks) - len(failed)}/{len(checks)} detector cases passed")
    for c in failed:
        print(f"  FAILED: {c['case']}\n    expected {c['expected']}\n    observed {c['observed']}")
    return not failed


# --------------------------------------------------------------------------- #
# Isolation + runtime wiring
# --------------------------------------------------------------------------- #
# STORAGE_ROOT is read by ``document_processor`` / ``excel_processor`` at IMPORT
# time (``app/core/config.py`` ends with a module-level ``settings = Settings()``),
# so it must be set before any ``app.*`` import. That is the ONLY configuration
# this harness overrides, and it overrides it purely to redirect local storage
# into the run directory -- no threshold, weight or scoring parameter is touched.

_APP_MODULES: Dict[str, Any] = {}

# Production defects the harness walks into while running the real pipeline.
# They are collected and reported rather than patched, because the harness must
# measure the system as it is.
FINDINGS: List[str] = []

# Guards the once-per-run "generation returned nothing" finding.
_EMPTY_ANSWER_REPORTED: List[bool] = []

# Sentinel for "the ground-truth row was not retrieved at all" in the variant
# deltas. It must never be averaged with real rank differences.
LOST_RANK_DELTA = 999

# Hard ceiling on one probe. `get_llm` (chat_service.py:11) builds the provider
# client without any timeout, so a stalled OpenRouter call blocks the whole run
# forever -- observed once during testing. The guard is the harness's own; a
# timeout is recorded as an error rather than hidden.
PROBE_TIMEOUT_SECONDS = 180.0


def configure_isolation(storage_root: str) -> None:
    """Point local Chroma + docs storage at a run-private directory."""
    os.environ["STORAGE_ROOT"] = os.path.abspath(storage_root)
    os.environ["USE_REMOTE_CHROMADB"] = "False"


def app_modules() -> Dict[str, Any]:
    """Import the real services lazily, after isolation is configured."""
    if _APP_MODULES:
        return _APP_MODULES
    from app.services.knowledge_base import (  # noqa: E402
        KnowledgeBaseService,
        HYBRID_PSEUDO_DISTANCE_RANGE,
    )
    from app.services.chat_service import ChatService  # noqa: E402
    from app.services.document_processor import get_document_processor  # noqa: E402
    from app.services.excel_processor import get_excel_qa_processor  # noqa: E402

    _APP_MODULES.update({
        "KnowledgeBaseService": KnowledgeBaseService,
        "HYBRID_PSEUDO_DISTANCE_RANGE": HYBRID_PSEUDO_DISTANCE_RANGE,
        "ChatService": ChatService,
        "get_document_processor": get_document_processor,
        "get_excel_qa_processor": get_excel_qa_processor,
    })
    return _APP_MODULES


@dataclass
class HarnessConfig:
    xls: str
    sheet: str
    limit: int
    probe_rows: int
    out: str
    variants: List[str]
    multiturn: bool
    sleep: float
    storage_root: str
    run_dir: str
    stamp: str


def write_subset_workbook(rows: Sequence[GtRow], dest: str, source_df_path: str) -> str:
    """Materialise exactly the test domain as its own workbook.

    Re-using the project's own ``ExcelQAProcessor`` on this file means the ingest
    path, metadata shape and ``row_number`` semantics are the production ones --
    the harness does not re-implement chunking.
    """
    import pandas as pd

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df = pd.read_excel(source_df_path, sheet_name=DEFAULT_SHEET)
    subset = df.iloc[[r.row_id for r in rows]].reset_index(drop=True)
    subset.to_excel(dest, index=False, engine="openpyxl")
    return dest


def ingest_subset(rows: Sequence[GtRow], cfg: HarnessConfig) -> Dict[str, Any]:
    """Controlled ingest of the test domain into the run-private vector store.

    FINDING (production defect, measured, not assumed): the documents
    ``ExcelQAProcessor.process_excel_file`` builds carry
    ``{source, file_path, source_type: "excel_qa", excel_mode, title, question,
    row_number}`` -- but ``HybridRetrievalService._dense_parallel`` filters every
    dense branch on ``entry_type`` AND ``is_public``
    (hybrid_retrieval.py:324-337). Documents without those keys match NO branch,
    so an index written purely by the Excel ingest path is invisible to
    retrieval.

    The pre-existing local corpus proves which keys are expected: all 1400
    ``source_type="excel_qa"`` documents in
    ``vectordb/openrouter-openai_text-embedding-3-small-1536`` carry
    ``entry_type="user_contribution_excel"`` and ``is_public=false``. The
    harness therefore adds exactly those keys, so the controlled ingest is a
    faithful stand-in for the production corpus instead of an empty one. The
    keys are taken from measurement; nothing else is invented.
    """
    import uuid

    mods = app_modules()
    docs_dir = os.path.join(cfg.storage_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    subset_path = os.path.join(docs_dir, f"qa_subset_{cfg.stamp}.xls")
    write_subset_workbook(rows, subset_path, cfg.xls)

    processor = mods["get_excel_qa_processor"]()
    vector_store = mods["get_document_processor"]().get_vector_store()
    if vector_store is None:
        raise RuntimeError(
            "vector store unavailable: embeddings could not be initialised. "
            "Check OPENROUTER_API_KEY / OPENAI_API_KEY before running the harness."
        )

    qa_count, documents = processor.process_excel_file(subset_path)
    if not documents:
        raise RuntimeError(
            f"ExcelQAProcessor produced no documents for {subset_path} "
            f"(reported qa pairs: {qa_count})")

    missing_entry_type = sum(1 for d in documents if "entry_type" not in (d.metadata or {}))
    missing_is_public = sum(1 for d in documents if "is_public" not in (d.metadata or {}))
    for document in documents:
        meta = document.metadata if document.metadata is not None else {}
        digest = uuid.uuid5(uuid.NAMESPACE_URL,
                            f"e2e-harness:{cfg.stamp}:{meta.get('row_number')}")
        meta.setdefault("entry_type", "user_contribution_excel")
        meta.setdefault("is_public", False)
        meta.setdefault("id", str(digest))
        meta.setdefault("hash_id", str(digest))
        meta.setdefault("author_name", "Unknown")
        meta.setdefault("meta_tags", "پرسش و پاسخ")
        document.metadata = meta

    before = _collection_count(vector_store)
    batch_size = 100
    for start in range(0, len(documents), batch_size):
        vector_store.add_documents(documents[start:start + batch_size])
    after = _collection_count(vector_store)

    if after <= before:
        raise RuntimeError(
            f"controlled ingest wrote no documents (collection count {before} -> {after})")

    if missing_entry_type or missing_is_public:
        FINDINGS.append(
            f"ExcelQAProcessor.process_excel_file omits the metadata the retriever filters on: "
            f"{missing_entry_type}/{len(documents)} documents had no `entry_type` and "
            f"{missing_is_public}/{len(documents)} had no `is_public`. "
            "HybridRetrievalService._dense_parallel (hybrid_retrieval.py:324-337) filters every "
            "dense branch on both keys, so a corpus ingested ONLY through the Excel path is "
            "invisible to retrieval. The harness added "
            "entry_type='user_contribution_excel' / is_public=False, the values measured on the "
            "1400 existing excel_qa documents in the local production store.")

    info = {
        "subset_workbook": subset_path,
        "expected_documents": len(rows),
        "processor_qa_pairs": qa_count,
        "documents_written": len(documents),
        "documents_added": after - before,
        "metadata_keys_added_by_harness": {
            "entry_type_missing_on": missing_entry_type,
            "is_public_missing_on": missing_is_public,
        },
        "collection_before": before,
        "collection_after": after,
        "persist_directory": mods["get_document_processor"]().persist_directory,
        "storage_root": cfg.storage_root,
    }
    print(f"[ingest] {before} -> {after} documents ({after - before} added) in "
          f"{info['persist_directory']}")
    print(f"[ingest] harness added entry_type/is_public to {missing_entry_type} document(s) "
          f"so the retriever's dense filters can see them")
    return info


def _collection_count(vector_store) -> int:
    try:
        return int(vector_store._collection.count())
    except Exception:  # pragma: no cover - defensive, never mask the run
        return -1


# --------------------------------------------------------------------------- #
# Query variants (section B)
# --------------------------------------------------------------------------- #

VARIANT_NAMES = ("verbatim", "paraphrase", "typo", "elliptical")

# Adjacent-key character swaps used for the deterministic typo variant. This is
# deliberately *not* an LLM call: the typo variant must be reproducible byte for
# byte so a rank_delta can be re-derived from the CSV alone.
_TYPO_SWAPS = (("ا", "س"), ("ی", "ب"), ("ن", "م"), ("ک", "گ"), ("ت", "ر"), ("د", "ذ"))


def build_elliptical(question: str, max_words: int = 3) -> str:
    """2-3 key content words, no verb phrase -- the hardest rewriting probe."""
    tokens = [t for t in re.split(r"[^\w\u0600-\u06FF]+", normalize_text(question))
              if t and t not in _STOPWORDS and not t.isdigit() and len(t) >= 3]
    if not tokens:
        tokens = re.split(r"\s+", normalize_text(question))[:max_words]
    return " ".join(tokens[:max_words])


def build_typo(question: str, seed: int) -> str:
    """Deterministic character-level corruption of the longest words."""
    rng = random.Random(f"typo-{seed}")
    parts = question.split()
    candidates = sorted(
        (i for i, w in enumerate(parts) if len(w) >= 4),
        key=lambda i: (-len(parts[i]), i),
    )[:2]
    for i in candidates:
        word = parts[i]
        old, new = _TYPO_SWAPS[rng.randrange(len(_TYPO_SWAPS))]
        pos = word.find(old)
        if pos >= 0:
            parts[i] = word[:pos] + new + word[pos + 1:]
        elif len(word) >= 5:
            cut = rng.randrange(1, len(word) - 1)
            parts[i] = word[:cut] + " " + word[cut:]
    return " ".join(parts)


PARAPHRASE_PROMPT = """You rewrite Persian customer questions for a RAG search index.

RULES:
1. Keep the MEANING identical. Do not add or remove facts, products or entities.
2. Keep EVERY number and unit EXACTLY as written (numbers must not change).
3. Use natural, colloquial Persian wording (محاوره‌ای) instead of the original phrasing.
4. Output ONLY the rewritten question, on one line. No quotes, no explanation.

QUESTION:
{question}
"""


async def paraphrase_question(question: str, cache: Dict[str, Any], llm) -> Tuple[str, bool]:
    """LLM paraphrase, memoised so re-runs are byte-identical.

    Returns ``(text, from_cache)``. On any failure the original question is
    returned unchanged and flagged, rather than silently inventing a variant.
    """
    key = normalize_text(question)[:200]
    if key in cache:
        return cache[key], True
    if llm is None:
        return question, False
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = await llm.ainvoke([
            SystemMessage(content="You are a careful Persian query paraphraser."),
            HumanMessage(content=PARAPHRASE_PROMPT.format(question=question)),
        ])
        text = (getattr(response, "content", "") or "").strip().strip('"').splitlines()
        text = text[0].strip() if text else ""
        if not text:
            return question, False
        cache[key] = text
        return text, False
    except Exception as exc:  # keep the run alive, record the truth
        print(f"[variants] paraphrase failed for row question ({exc}); using original")
        return question, False


def _validate_variant(original: str, variant: str) -> Dict[str, Any]:
    """Drift check: did the variant change the numbers or lose the entities?"""
    orig_q = extract_quantities(original)
    var_q = extract_quantities(variant)
    orig_terms = _terms(original)
    var_terms = _terms(variant)
    return {
        "numbers_changed": orig_q != var_q,
        "numbers_original": format_quantities(orig_q),
        "numbers_variant": format_quantities(var_q),
        "terms_kept_ratio": round(len(orig_terms & var_terms) / len(orig_terms), 4) if orig_terms else 1.0,
        "entity_loss": not (orig_terms & var_terms) if orig_terms else False,
    }


# --------------------------------------------------------------------------- #
# Probe runner
# --------------------------------------------------------------------------- #

@dataclass
class KbHandles:
    kb: Any
    chat: Any
    threshold: float
    top_k: int
    referral_message: str
    provider: str
    model: str
    env_threshold: float = -1.0
    qa_match_threshold: float = -1.0
    similarity_threshold: float = -1.0
    temperature: float = -1.0
    gen_model: str = "?"

    @classmethod
    async def build(cls) -> "KbHandles":
        mods = app_modules()
        from app.core.config import settings as static_settings

        # Initialize embeddings BEFORE constructing KnowledgeBaseService.
        # `KnowledgeBaseService.__init__` builds `self.reranker` only when
        # `document_processor.embeddings_available` is already True
        # (knowledge_base.py:88-101), and it is never rebuilt afterwards. Getting
        # the vector store first runs the embedding probe, so the reranker is
        # actually active -- otherwise the harness would silently measure an
        # un-reranked retrieval path that production never uses.
        dp = mods["get_document_processor"]()
        dp.get_vector_store()

        kb = mods["KnowledgeBaseService"]()
        rag = await kb.config_service.get_rag_settings()

        effective = float(rag.knowledge_base_confidence_threshold)
        env_value = float(static_settings.KNOWLEDGE_BASE_CONFIDENCE_THRESHOLD)
        if effective != env_value:
            FINDINGS.append(
                f"The effective confidence threshold comes from the active MongoDB config document "
                f"(ConfigService._create_fallback_config, config_service.py:75-82 stores it and "
                f"get_rag_settings reads it back), NOT from .env: "
                f"KNOWLEDGE_BASE_CONFIDENCE_THRESHOLD={env_value} in .env is dead config while the "
                f"database value is {effective}. chat_service.py:999 and :1390 bind "
                f"KB_CONFIDENCE_THRESHOLD from rag_settings, so with {effective} the gate at "
                f"chat_service.py:1612 (`kb_confidence >= KB_CONFIDENCE_THRESHOLD`) can NEVER "
                f"block a response and the whole `gate` layer is inert.")
        if str(rag.human_referral_message).strip() != str(
                static_settings.HUMAN_REFERRAL_MESSAGE).strip():
            FINDINGS.append(
                f"rag_settings.human_referral_message is {rag.human_referral_message!r} "
                f"(from the live MongoDB config), while .env declares a different message. Every "
                f"handoff shown to a user is this string, and it is returned from "
                f"knowledge_base.py:1415 / chat_service.py:1298 / :1346.")

        llm_settings = await kb.config_service.get_llm_settings()
        gen_model = str(getattr(llm_settings, "default_model", "?"))
        if gen_model != str(static_settings.DEFAULT_MODEL):
            FINDINGS.append(
                f"llm_settings.default_model is {gen_model!r} (from the live MongoDB config "
                f"document), NOT the code default DEFAULT_MODEL={static_settings.DEFAULT_MODEL!r} "
                f"(app/core/config.py:16). get_llm (chat_service.py:66) falls back to this DB value "
                f"whenever a caller passes model_name=None, which every RAG call does "
                f"(knowledge_base.py:111).")
        if gen_model.startswith("~"):
            FINDINGS.append(
                f"The RAG generation model is the floating alias {gen_model!r} (leading '~' means "
                f"OpenRouter resolves it to whatever the newest build is), so generations are not "
                f"reproducible across runs. This alias intermittently returns a 200 OK with "
                f"zero-length content and finish_reason 'stop': 8 of 39 probes (21%) came back "
                f"empty in one reference run, the same row 2 came back empty in 3 separate runs and "
                f"answered normally when re-probed later, and a direct ainvoke returned content=''. "
                f"See the 'empty_answer' probes in e2e_records.csv.")

        return cls(
            kb=kb,
            chat=mods["ChatService"](),
            threshold=effective,
            top_k=int(getattr(rag, "top_k_results", -1)),
            referral_message=str(getattr(rag, "human_referral_message", "")),
            provider=str(getattr(dp, "_provider", "?")),
            model=str(getattr(dp, "_model_name", "?")),
            env_threshold=env_value,
            qa_match_threshold=float(getattr(rag, "qa_match_threshold", -1)),
            similarity_threshold=float(getattr(rag, "similarity_threshold", -1)),
            temperature=float(getattr(rag, "temperature", -1)),
            gen_model=gen_model,
        )


def locate_ground_truth(rank_list: Sequence[Any], row: GtRow) -> Tuple[int, List[Dict[str, Any]]]:
    """Find the ground-truth row inside the retrieved ranking.

    Matching is done on the ``question`` metadata field, which
    ``ExcelQAProcessor.process_excel_file`` copies verbatim from the workbook --
    a stronger identity than the chunk text (which the caption/sanitiser may
    rewrite). ``row_number`` is kept in the digest for cross-checking.
    """
    target = normalize_text(row.question)
    digest: List[Dict[str, Any]] = []
    rank = -1
    for position, item in enumerate(rank_list, start=1):
        doc, distance = item[0], item[1]
        meta = dict(getattr(doc, "metadata", {}) or {})
        doc_question = normalize_text(str(meta.get("question", "")))
        is_hit = doc_question == target
        if is_hit and rank == -1:
            rank = position
        digest.append({
            "rank": position,
            "row_number": meta.get("row_number"),
            "source": meta.get("source"),
            "source_type": meta.get("source_type"),
            "title": (str(meta.get("title")) or "")[:80],
            "pseudo_distance": round(float(distance), 6),
            "hybrid_score": meta.get("hybrid_score"),
            "dense_score_norm": meta.get("dense_score_norm"),
            "bm25_score_norm": meta.get("bm25_score_norm"),
            "rerank_position": meta.get("rerank_position"),
            "is_ground_truth": is_hit,
        })
    return rank, digest


async def _execute_probe(
    handles: KbHandles,
    query: str,
    conversation_history: Optional[List[Dict[str, str]]],
    skip_rewrite: bool,
    row: Optional[GtRow],
) -> Dict[str, Any]:
    """Retrieval -> gate -> generation for one query. Raises on failure."""
    retrieval = await handles.kb._retrieve_context(
        query,
        conversation_history,
        False,
        skip_rewrite=skip_rewrite,
    )
    docs_with_scores = retrieval.get("docs_with_scores") or []
    rewritten = retrieval.get("rewritten_query") or query
    confidence = float(retrieval.get("confidence_score") or 0.0)

    out: Dict[str, Any] = {
        "rewritten_query": rewritten,
        "retrieved_k": len(docs_with_scores),
        "confidence_score": round(confidence, 6),
        "top1_score": None,
        "top1_hybrid_score": None,
        "gate_blocked": False,
        "retrieved_meta": [],
        "gt_rank": -1,
    }
    if docs_with_scores:
        first_doc, first_distance = docs_with_scores[0]
        out["top1_score"] = round(float(first_distance), 6)
        out["top1_hybrid_score"] = (first_doc.metadata or {}).get("hybrid_score")
    if row is not None:
        out["gt_rank"], out["retrieved_meta"] = locate_ground_truth(docs_with_scores, row)

    # --- GATE (mirrors chat_service.py:1612) -------------------------------- #
    if confidence >= handles.threshold and docs_with_scores:
        tokens = []
        async for token in handles.kb.stream_answer_from_context(
            rewritten, retrieval.get("normalized_docs") or []
        ):
            tokens.append(token)
        out["answer"] = "".join(tokens)
        if not out["answer"].strip():
            # stream_answer_from_context swallows chunks it cannot read
            # (knowledge_base.py:1218-1219) and never raises, so a zero-length
            # completion reaches the user as a blank bubble instead of the
            # referral. Recorded once per run, not once per probe.
            if not _EMPTY_ANSWER_REPORTED:
                _EMPTY_ANSWER_REPORTED.append(True)
                FINDINGS.append(
                    f"Generation returned nothing for a probe whose ground-truth row was "
                    f"retrieved. stream_answer_from_context (knowledge_base.py:1212-1224) skips "
                    f"every chunk without readable content and logs only "
                    f"'step=response_generation_stream elapsed=...' -- 'time_to_first_chunk' is "
                    f"never logged, so the empty stream is indistinguishable from a slow one, and "
                    f"the caller streams a blank message instead of the referral. Measured "
                    f"directly against {handles.gen_model!r}: 1996 stream chunks / 0 characters, "
                    f"and a non-streaming ainvoke returning content='' with "
                    f"additional_kwargs={{'refusal': None}} on a 200 OK.")
    else:
        out["answer"] = handles.referral_message
        out["gate_blocked"] = True
    return out


async def run_probe(
    handles: KbHandles,
    cfg: HarnessConfig,
    boilerplate: Set[str],
    *,
    query: str,
    row: Optional[GtRow],
    variant: str,
    turn_index: int,
    turn_kind: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    skip_rewrite: bool = False,
    context_state: str = "",
    resolved_query: str = "",
    expect_refusal: bool = False,
) -> Dict[str, Any]:
    """Execute one probe through the production retrieval -> gate -> generation path."""
    started = time.perf_counter()
    record: Dict[str, Any] = {
        "row_id": row.row_id if row else -1,
        "excel_row": row.excel_row if row else -1,
        "title": row.title if row else "",
        "variant": variant,
        "turn_index": turn_index,
        "turn_kind": turn_kind,
        "query_sent": query,
        "context_state": context_state,
        "resolved_query": resolved_query,
        "variant_expected_refusal": expect_refusal,
        "gt_rank": -1,
        "retrieved_k": 0,
        "top1_score": None,
        "confidence_score": None,
        "threshold": handles.threshold,
        "gate_blocked": False,
        "refused": True,
        "requires_human": True,
        "answer": "",
        "rewritten_query": "",
        "topic_retention": None,
        "context_bleed": None,
        "error": "",
        "retrieved_meta": [],
    }

    try:
        outcome = await asyncio.wait_for(
            _execute_probe(handles, query, conversation_history, skip_rewrite, row),
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        record.update(outcome)
        record["refused"] = is_refusal(record["answer"], handles.referral_message)
        record["requires_human"] = record["refused"]

        log = logging.getLogger("harness.retrieval")
        log.info(
            "row=%s variant=%s turn=%s query=%r rewritten=%r docs=%d gt_rank=%s "
            "confidence=%.4f threshold=%.4f gate=%s",
            record["row_id"], variant, turn_index, query, record["rewritten_query"],
            record["retrieved_k"], record["gt_rank"], record.get("confidence_score") or 0.0,
            handles.threshold, "closed" if record["gate_blocked"] else "open",
        )
        for entry in record["retrieved_meta"]:
            log.info("  retrieved: %s", entry)

    except asyncio.TimeoutError:
        # A hung provider call must not stall the whole run. The production LLM
        # client is created without a timeout (get_llm, chat_service.py:11), so
        # this guard is the harness's own protection and the event is recorded.
        record["error"] = f"TimeoutError: probe exceeded {PROBE_TIMEOUT_SECONDS:.0f}s"
        record["timeout"] = True
        record["answer"] = handles.referral_message
        record["refused"] = True
        record["requires_human"] = True
        logging.getLogger("harness").error(
            "probe TIMEOUT row=%s variant=%s turn=%s after %.0fs",
            record["row_id"], variant, turn_index, PROBE_TIMEOUT_SECONDS)

    except Exception as exc:
        # Never swallow: the error text itself is evidence and is reported.
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["answer"] = handles.referral_message
        record["refused"] = True
        record["requires_human"] = True
        logging.getLogger("harness").exception(
            "probe error row=%s variant=%s turn=%s", record["row_id"], variant, turn_index)

    # --- RUBRIC ------------------------------------------------------------- #
    # An errored or timed-out probe has no evidence to grade; it must never be
    # summarised as a Pass. The error text is carried through to the report.
    overrides = error_overrides(record)
    if overrides is not None:
        record.update(overrides)
        record["latency_ms"] = int((time.perf_counter() - started) * 1000)
        return record

    gold_answer = row.answer if row else ""
    coverage = fact_coverage(gold_answer, record["answer"], boilerplate) if gold_answer else 0.0
    leaks = detect_style_leaks(record["answer"], record["refused"])
    record.update(classify_record(
        answer=record["answer"],
        gold_answer=gold_answer,
        refused=record["refused"],
        gt_rank=record["gt_rank"],
        coverage=coverage,
        style_leaks=leaks,
        refusal_ok=expect_refusal,
    ))
    record["style_leaks"] = leaks
    record["blame"] = attribute_blame(record)
    record["blame_note"] = ""
    record["latency_ms"] = int((time.perf_counter() - started) * 1000)
    return record


# --------------------------------------------------------------------------- #
# Multi-turn / context scenarios (section C)
# --------------------------------------------------------------------------- #

# "مقدارش چقدره؟" -- a purely referential follow-up with no subject of its own.
IMPLICIT_FOLLOWUP = "مقدارش چقدره؟"


def _distinctive_terms(a: GtRow, b: GtRow, boilerplate: Set[str]) -> Set[str]:
    """Terms unique to row A's answer -- the leak fingerprint for the topic switch."""
    return (_terms(a.answer) - _terms(b.answer)) - boilerplate


async def run_multiturn_scenarios(
    handles: KbHandles,
    cfg: HarnessConfig,
    boilerplate: Set[str],
    rows: Sequence[GtRow],
) -> List[Dict[str, Any]]:
    """Three real multi-turn scenarios; history is carried between real turns."""
    records: List[Dict[str, Any]] = []
    if len(rows) < 2:
        return records

    row_a = rows[0]
    # Topic switch needs a genuinely different subject: first row whose Title and
    # answer terms barely overlap with row A.
    row_b = next(
        (r for r in rows[1:] if r.title.strip() != row_a.title.strip()
         and len(_distinctive_terms(r, row_a, boilerplate)) >= 3),
        rows[1],
    )

    async def one_turn(
        *, query, row, history, variant, label, expect_refusal=False
    ) -> Dict[str, Any]:
        intent = {"context_state": "", "resolved_query": query, "context_decided": False,
                  "referenced_entities": []}
        try:
            intent = await handles.chat.detect_query_intent(query, history) or intent
        except Exception as exc:  # context layer failure is itself a finding
            logging.getLogger("harness").warning("detect_query_intent failed: %s", exc)
            intent = {"context_state": f"error:{type(exc).__name__}", "resolved_query": query,
                      "context_decided": False, "referenced_entities": []}

        context_decided = bool(intent.get("context_decided"))
        retrieval_query = intent.get("resolved_query") or query
        record = await run_probe(
            handles, cfg, boilerplate,
            query=retrieval_query,
            row=row, variant=variant, turn_index=len(history) // 2 + 1,
            turn_kind="multiturn",
            conversation_history=history or None,
            skip_rewrite=context_decided,
            context_state=str(intent.get("context_state") or ""),
            resolved_query=str(intent.get("resolved_query") or ""),
            expect_refusal=expect_refusal,
        )
        record["scenario"] = label
        # Same as the single-turn loop: without this the audited row keeps an
        # empty gold_answer and the S1 dropped-dose evidence cannot be checked.
        record["gold_answer"] = row.answer
        record["referenced_entities"] = intent.get("referenced_entities") or []
        record["context_decided"] = context_decided
        record["history_len"] = len(history)
        # The mission asks explicitly whether turn 2 filled the reference in.
        record["reference_filled"] = bool(
            context_decided and normalize_text(retrieval_query) != normalize_text(query)
        ) or normalize_text(record.get("rewritten_query", "")) != normalize_text(query)
        return record

    # --- Scenario 1: implicit reference ------------------------------------- #
    t1 = await one_turn(query=row_a.question, row=row_a, history=[],
                        variant="multiturn_t1", label="implicit_reference")
    history = [
        {"role": "user", "content": row_a.question},
        {"role": "assistant", "content": t1["answer"]},
    ]
    t2 = await one_turn(query=IMPLICIT_FOLLOWUP, row=row_a, history=history,
                        variant="multiturn_t2", label="implicit_reference")
    t2["topic_retention"] = t2["gt_rank"] > 0
    t2["context_bleed"] = False  # nothing to bleed from: same subject on purpose
    records.extend([t1, t2])

    # --- Scenario 2: topic change ------------------------------------------- #
    s1 = await one_turn(query=row_a.question, row=row_a, history=[],
                        variant="multiturn_t1", label="topic_change")
    history_b = [
        {"role": "user", "content": row_a.question},
        {"role": "assistant", "content": s1["answer"]},
    ]
    s2 = await one_turn(query=row_b.question, row=row_b, history=history_b,
                        variant="multiturn_t2", label="topic_change")
    fingerprint = _distinctive_terms(row_a, row_b, boilerplate)
    bled = {t for t in fingerprint if t in _terms(s2["answer"])}
    s2["bleed_terms"] = sorted(bled)
    s2["context_bleed"] = len(bled) >= 2
    s2["topic_retention"] = s2["gt_rank"] > 0
    s1["topic_retention"] = s1["gt_rank"] > 0
    s1["context_bleed"] = False
    records.extend([s1, s2])

    # --- Scenario 3: three-turn chain --------------------------------------- #
    c1 = await one_turn(query=row_a.question, row=row_a, history=[],
                        variant="multiturn_t1", label="chain3")
    c2 = await one_turn(query=IMPLICIT_FOLLOWUP, row=row_a,
                        history=[{"role": "user", "content": row_a.question},
                                 {"role": "assistant", "content": c1["answer"]}],
                        variant="multiturn_t2", label="chain3")
    c3 = await one_turn(query="بیشتر توضیح بده", row=row_a,
                        history=[{"role": "user", "content": row_a.question},
                                 {"role": "assistant", "content": c1["answer"]},
                                 {"role": "user", "content": IMPLICIT_FOLLOWUP},
                                 {"role": "assistant", "content": c2["answer"]}],
                        variant="multiturn_t3", label="chain3")
    for rec in (c1, c2, c3):
        rec["topic_retention"] = rec["gt_rank"] > 0
        rec["context_bleed"] = False
    records.extend([c1, c2, c3])

    # Stage-1 blame was computed before these fields existed; recompute.
    for rec in records:
        rec["blame"] = attribute_blame(rec)
    return records


# --------------------------------------------------------------------------- #
# Metrics + summary
# --------------------------------------------------------------------------- #

RECALL_KS = (1, 3, 5)


def retrieval_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """recall@k, MRR and rank distribution over probes that have a ground truth."""
    ranked = [r for r in records if r.get("row_id", -1) >= 0]
    ranks = [r.get("gt_rank", -1) for r in ranked]
    hits = [x for x in ranks if isinstance(x, int) and x > 0]
    n = len(ranks)
    if n == 0:
        return {"probes": 0, "note": "not computed: no retrievable probes in this run"}

    metrics: Dict[str, Any] = {
        "probes": n,
        "retrieved": len(hits),
        "missed": n - len(hits),
        "mrr": round(sum(1.0 / x for x in hits) / n, 4) if n else 0.0,
        "rank_distribution": {str(k): v for k, v in sorted(Counter(ranks).items())},
    }
    for k in RECALL_KS:
        metrics[f"recall@{k}"] = round(sum(1 for x in hits if x <= k) / n, 4)
    confidences = [r["confidence_score"] for r in ranked if r.get("confidence_score") is not None]
    if confidences:
        metrics["mean_confidence"] = round(statistics.fmean(confidences), 4)
        metrics["median_confidence"] = round(statistics.median(confidences), 4)
        metrics["below_threshold"] = sum(
            1 for r in ranked
            if r.get("confidence_score") is not None
            and r.get("threshold") is not None
            and r["confidence_score"] < r["threshold"]
        )
    return metrics


def rewriting_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-row rank_delta of each variant against the verbatim baseline."""
    by_row: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        if rec.get("row_id", -1) >= 0 and rec.get("variant") in VARIANT_NAMES:
            by_row[rec["row_id"]][rec["variant"]] = rec

    deltas: List[Dict[str, Any]] = []
    worst: List[Dict[str, Any]] = []
    for row_id in sorted(by_row):
        variants = by_row[row_id]
        base = variants.get("verbatim")
        if not base:
            continue
        base_rank = base.get("gt_rank", -1)
        entry: Dict[str, Any] = {"row_id": row_id, "verbatim_rank": base_rank, "deltas": {}}
        for name in ("paraphrase", "typo", "elliptical"):
            rec = variants.get(name)
            if not rec:
                continue
            rank = rec.get("gt_rank", -1)
            delta = None
            if isinstance(base_rank, int) and base_rank > 0 and isinstance(rank, int) and rank > 0:
                delta = rank - base_rank
            elif isinstance(base_rank, int) and base_rank > 0 and (not rank or rank < 0):
                delta = LOST_RANK_DELTA  # sentinel: row not retrieved at all
            entry["deltas"][name] = delta
            entry[f"{name}_refused"] = bool(rec.get("refused"))
            entry[f"{name}_numbers_changed"] = rec.get("numbers_changed")
        deltas.append(entry)
        if any(v is not None and v > 0 for v in entry["deltas"].values()):
            worst.append(entry)

    if not deltas:
        return {"rows": 0, "note": "not computed: variants were disabled"}

    summary: Dict[str, Any] = {"rows": len(deltas), "regressed_rows": len(worst)}
    for name in ("paraphrase", "typo", "elliptical"):
        raw = [e["deltas"][name] for e in deltas if e["deltas"].get(name) is not None]
        # 999 is a "lost entirely" sentinel, not a rank difference. Averaging it
        # in with real deltas produces a number that measures nothing (elliptical
        # scored 378.25 before this split); losses are reported as a rate instead.
        finite = [v for v in raw if v != LOST_RANK_DELTA]
        summary[f"{name}_mean_rank_delta"] = (
            round(statistics.fmean(finite), 3) if finite else None)
        summary[f"{name}_comparable_rows"] = len(finite)
        summary[f"{name}_lost_count"] = sum(1 for v in raw if v == LOST_RANK_DELTA)
        summary[f"{name}_regressed_count"] = sum(1 for v in raw if v > 0)
        summary[f"{name}_refusal_count"] = sum(1 for e in deltas if e.get(f"{name}_refused"))
        summary[f"{name}_numbers_changed_count"] = sum(
            1 for e in deltas if e.get(f"{name}_numbers_changed"))
    summary["regressed_row_ids"] = [e["row_id"] for e in worst]
    summary["details"] = deltas
    return summary


def multiturn_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """topic_retention / context_bleed / reference fill-in per scenario."""
    turns = [r for r in records if r.get("turn_kind") == "multiturn"]
    if not turns:
        return {"scenarios": 0, "note": "not computed: --multiturn was not requested"}
    out: Dict[str, Any] = {}
    for scenario in sorted({str(t.get("scenario")) for t in turns}):
        group = [t for t in turns if str(t.get("scenario")) == scenario]
        followups = [t for t in group if (t.get("turn_index") or 1) > 1]
        out[scenario] = {
            "turns": len(group),
            "followup_turns": len(followups),
            "topic_retention_rate": (
                round(sum(1 for t in followups if t.get("topic_retention")) / len(followups), 4)
                if followups else None
            ),
            "context_bleed_count": sum(1 for t in followups if t.get("context_bleed")),
            "reference_filled_count": sum(1 for t in followups if t.get("reference_filled")),
            "context_state_seen": sorted(
                {str(t.get("context_state")) for t in group if t.get("context_state")}),
            "refused_followups": sum(1 for t in followups if t.get("refused")),
        }
    return out
def _tier_blame_matrix(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in records:
        matrix[rec.get("tier", "Pass")][rec.get("blame", "none")] += 1
    return {tier: dict(blames) for tier, blames in sorted(matrix.items())}


def build_summary(
    records: Sequence[Dict[str, Any]],
    meta: Dict[str, Any],
    cfg: HarnessConfig,
    handles_info: Dict[str, Any],
    validation: Sequence[Dict[str, Any]],
    ingest: Dict[str, Any],
) -> Dict[str, Any]:
    """Everything the mission requires in ``reports/*.summary.json``."""
    tiers = Counter(r.get("tier", "Pass") for r in records)
    blames = Counter(r.get("blame", "none") for r in records)

    error_rows: Dict[str, List[int]] = defaultdict(list)
    for r in records:
        if r.get("blame") not in (None, "none"):
            error_rows[r["blame"]].append(r.get("row_id"))
        if r.get("tier") not in (None, "Pass"):
            error_rows[f"tier_{r['tier']}"].append(r.get("row_id"))

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": cfg.run_dir,
        "configuration": {
            "xls": cfg.xls,
            "sheet": cfg.sheet,
            "limit": cfg.limit,
            "probe_rows": cfg.probe_rows,
            "variants": cfg.variants,
            "multiturn": cfg.multiturn,
            "sleep_seconds": cfg.sleep,
            "storage_root": cfg.storage_root,
        },
        "environment": handles_info,
        "ingest": ingest,
        "findings": list(FINDINGS),
        "layer_contract": {
            "context": "ChatService.detect_query_intent(message, conversation_history=None, *, llm=None) -> context_state/resolved_query",
            "rewrite_helper": "KnowledgeBaseService.expand_query_with_context(query, conversation_history=None, max_history=4, skip_rewrite=False)",
            "retrieval_and_rewrite": "KnowledgeBaseService._retrieve_context(query, conversation_history=None, is_public=False, include_web_search=True, skip_rewrite=False)",
            "confidence": "KnowledgeBaseService._calculate_confidence_score(docs_with_scores, top_n=3, score_scale=HYBRID_PSEUDO_DISTANCE_RANGE=1.0)",
            "gate": "retrieval.confidence_score >= rag_settings.knowledge_base_confidence_threshold (chat_service.py:1612)",
            "generation": "KnowledgeBaseService.stream_answer_from_context(query, normalized_docs) (chat_service.py:1631)",
        },
        "thresholds": {
            "knowledge_base_confidence_threshold": handles_info.get("threshold"),
            "top_k_results": handles_info.get("top_k"),
            "note_coverage_threshold": NOTE_COVERAGE_THRESHOLD,
        },
        "detector_validation": {
            "cases": len(validation),
            "passed": sum(1 for c in validation if c["ok"]),
            "all_passed": all(c["ok"] for c in validation),
            "details": list(validation),
        },
        "counts": {
            "probe_records": len(records),
            "distinct_rows": len({r["row_id"] for r in records if r.get("row_id", -1) >= 0}),
            "by_variant": dict(Counter(r.get("variant", "") for r in records)),
            "errors": sum(1 for r in records if r.get("error")),
            "gate_blocked": sum(1 for r in records if r.get("gate_blocked")),
            "refused": sum(1 for r in records if r.get("refused")),
        },
        "tier_distribution": dict(tiers),
        "blame_distribution": dict(blames),
        "tier_x_blame": _tier_blame_matrix(records),
        "retrieval": retrieval_metrics(records),
        "rewriting": rewriting_metrics(records),
        "multiturn": multiturn_metrics(records),
        "error_row_ids": {k: sorted({x for x in v if x is not None and x >= 0})
                          for k, v in sorted(error_rows.items())},
        "s3_rows_for_manual_review": [
            {
                "row_id": r["row_id"], "variant": r["variant"], "coverage": r.get("coverage"),
                "gt_rank": r.get("gt_rank"), "answer_head": (r.get("answer") or "")[:160],
            }
            for r in records if r.get("tier") == "S3"
        ],
        "ground_truth": meta,
    }


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

CSV_FIELDS = [
    "row_id", "excel_row", "variant", "turn_index", "turn_kind", "scenario", "title",
    "query_sent", "context_state", "resolved_query", "reference_filled", "rewritten_query",
    "gt_rank", "retrieved_k", "top1_score", "top1_hybrid_score",
    "confidence_score", "threshold", "gate_blocked", "refused", "requires_human",
    "tier", "tier_all", "note", "blame", "blame_note", "coverage",
    "dose_contradictions", "dose_dropped", "style_leaks",
    "numbers_changed", "terms_kept_ratio", "entity_loss",
    "topic_retention", "context_bleed", "bleed_terms", "latency_ms", "error", "timeout",
    "answer", "gold_answer",
]


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def record_to_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    row = {field: _cell(rec.get(field)) for field in CSV_FIELDS}
    parts = list(rec.get("tier_reasons") or [])
    if not parts:
        parts = [rec.get("error") or "pass"]
    row["note"] = " | ".join(str(p) for p in parts)
    row["gold_answer"] = (rec.get("gold_answer") or "")
    return row


def write_csv(path: str, records: Sequence[Dict[str, Any]]) -> None:
    # utf-8-sig so Excel opens the Persian columns correctly.
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(record_to_row(rec))


class CsvSink:
    """Append-as-you-go CSV writer.

    A full run costs ~40 s per probe, so results must survive an interruption:
    every probe is flushed to disk immediately instead of only at the end.
    """

    def __init__(self, path: str):
        self.path = path
        self._handle = open(path, "w", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._handle, fieldnames=CSV_FIELDS,
                                      extrasaction="ignore")
        self._writer.writeheader()
        self._handle.flush()

    def add(self, record: Dict[str, Any]) -> None:
        self._writer.writerow(record_to_row(record))
        self._handle.flush()

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()

    def __enter__(self) -> "CsvSink":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def write_ground_truth_csv(path: str, rows: Sequence[GtRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "excel_row", "title", "question", "gold_answer", "gold_quantities"])
        for row in rows:
            writer.writerow([row.row_id, row.excel_row, row.title, row.question,
                             row.answer, format_quantities(row.quantities)])


def setup_logging(run_dir: str) -> str:
    """Full per-call log: final query sent to the retriever + retrieved metadata."""
    log_path = os.path.join(run_dir, "harness.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s | %(message)s"))
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s | %(message)s"))
    root.addHandler(stream_handler)
    return log_path


def create_run_dir(out: str, stamp: str) -> Tuple[str, str]:
    """``--out`` is the reports root; each run gets its own timestamped folder."""
    run_dir = os.path.abspath(os.path.join(out, stamp))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, os.path.join(run_dir, "storage")


# --------------------------------------------------------------------------- #
# Console report
# --------------------------------------------------------------------------- #
# Each blame maps to the exact source location that owns it plus one concrete
# change. These references are the ones discovered in STEP-0, not guesses.

BOTTLENECK_FIXES: Dict[str, Dict[str, str]] = {
    "retrieval": {
        "source": "app/services/knowledge_base.py:1073-1132 (_retrieve_context) and "
                  "app/services/hybrid_retrieval.py:549-640 (hybrid_retrieve)",
        "fix": "The ground-truth row never reaches the top-k. hybrid_retrieve uses a "
               "hard-coded k=15 / prefilter_k=20 (hybrid_retrieval.py:554-555) regardless of "
               "rag_settings.top_k_results, and the 0.6*dense + 0.4*bm25 blend "
               "(hybrid_retrieval.py:598-603) is dense-dominated. Raise prefilter_k, make the "
               "branch weights configurable, then re-measure recall@5 on "
               "error_row_ids.retrieval.",
    },
    "gate": {
        "source": "app/services/knowledge_base.py:1145-1151 (confidence call) vs "
                  "app/services/chat_service.py:1612 (the comparison)",
        "fix": "The row IS in the top-k but confidence fell under "
               "knowledge_base_confidence_threshold, so a retrievable answer is refused. The "
               "score comes from the curve at knowledge_base.py:352-367 over "
               "HYBRID_PSEUDO_DISTANCE_RANGE=1.0, and the structural multiplier caps it at "
               "best_confidence (knowledge_base.py:328-333). Fix the ranking so the best "
               "document scores higher, or re-calibrate the curve on the labelled rows -- "
               "do not just lower the threshold.",
    },
    "generation": {
        "source": "app/services/knowledge_base.py:1184-1224 (stream_answer_from_context) and "
                  "app/services/chat_service.py:1631-1640",
        "fix": "Retrieval and the gate were both fine; the defect is in the produced text "
               "(wrong number, dropped dose, style leakage). The prompt is assembled at "
               "knowledge_base.py:117-125 from rag_settings.system_prompt + prompt_template. "
               "Instruct the model to copy every number and unit verbatim from the context and "
               "never emit the trainer greeting/signature, and validate the streamed answer "
               "against the extracted quantities before emitting it.",
    },
    "rewriting": {
        "source": "app/services/knowledge_base.py:670-895 (expand_query_with_context)",
        "fix": "The row is retrievable verbatim but the paraphrase/typo variant loses it, so the "
               "rewrite changed meaning. The rewriter is told to stay short "
               "(knowledge_base.py:754) but number preservation is never checked, even though a "
               "contamination guard already exists (_looks_contaminated, "
               "knowledge_base.py:945-960). Extend that guard with a number/entity preservation "
               "check and fall back to the raw query when it fails.",
    },
    "context": {
        "source": "app/services/chat_service.py:520-963 (detect_query_intent: context_state / "
                  "resolved_query) consumed at app/services/chat_service.py:1597-1600",
        "fix": "The follow-up turn lost or bled the subject. detect_query_intent makes the "
               "context decision in one LLM call and _retrieve_context is then called with "
               "skip_rewrite=context_decided, so a wrong decision is never corrected "
               "downstream. Tighten the referential-cue list in the classifier prompt and add "
               "a deterministic post-check: for a follow-up, resolved_query must contain an "
               "entity taken from the previous user turn.",
    },
    "none": {"source": "-", "fix": "-"},
}


def print_console_report(summary: Dict[str, Any], records: Sequence[Dict[str, Any]]) -> None:
    """tier x blame table, then the top bottlenecks with source refs and fixes."""
    print("\n" + "=" * 108)
    print("E2E RAG HARNESS -- RESULT SUMMARY")
    print("=" * 108)
    gt = summary["ground_truth"]
    print(f"ground truth : {gt['xls_path']}")
    print(f"sheet        : {gt['sheet_used']}  columns={gt['columns']}  "
          f"rows in sheet={gt['total_rows_in_sheet']}")
    print(f"test domain  : {gt['domain_row_rule']}  -> loaded {gt['domain_loaded_rows']} rows")
    conf = summary.get("configuration", {})
    print(f"probe scope  : corpus={gt['domain_loaded_rows']} rows, "
          f"probed={conf.get('probe_rows') or gt['domain_loaded_rows']} rows "
          f"(probe_rows={conf.get('probe_rows')})")
    env = summary["environment"]
    print(f"embedding    : {env.get('provider')}/{env.get('model')}   "
          f"threshold={env.get('threshold')}   top_k={env.get('top_k')}")

    dv = summary["detector_validation"]
    print(f"detector self-validation: {dv['passed']}/{dv['cases']} cases passed "
          f"(all_passed={dv['all_passed']})")

    if summary.get("findings"):
        print("\nPRODUCTION FINDINGS ENCOUNTERED WHILE RUNNING THE PIPELINE")
        print("-" * 108)
        for finding in summary["findings"]:
            print(f"  * {finding}")

    print("\nTIER x BLAME")
    print("-" * 108)
    blames = sorted({b for row in summary["tier_x_blame"].values() for b in row})
    print("tier  " + "".join(f"{b:>12}" for b in blames) + f"{'total':>8}")
    for tier, row in summary["tier_x_blame"].items():
        total = sum(row.values())
        print(f"{tier:<6}" + "".join(f"{row.get(b, 0):>12}" for b in blames) + f"{total:>8}")

    print("\nDISTRIBUTIONS")
    print("-" * 108)
    print(f"tier : {summary['tier_distribution']}")
    print(f"blame: {summary['blame_distribution']}")
    ret = summary["retrieval"]
    if "recall@5" in ret:
        print(f"retrieval: recall@1={ret['recall@1']} recall@3={ret['recall@3']} "
              f"recall@5={ret['recall@5']} MRR={ret['mrr']} "
              f"mean_conf={ret.get('mean_confidence')} median_conf={ret.get('median_confidence')}")
        print(f"rank distribution: {ret['rank_distribution']}")
        print(f"below threshold: {ret.get('below_threshold')}/{ret['probes']}")

    mt = summary["multiturn"]
    # The computed payload is a mapping of scenario -> stats; the "not computed"
    # payload is the only one carrying a "note" key. Testing for "scenarios" here
    # (which the computed payload never has) printed "not computed" even after a
    # full multi-turn run.
    if mt and "note" not in mt:
        print("\nMULTI-TURN")
        print("-" * 108)
        for scenario, stats in mt.items():
            print(f"  {scenario}: {stats}")
    else:
        print(f"\nMULTI-TURN: {mt.get('note', 'not computed')}")

    print("\nTOP BOTTLENECKS (ranked by number of probes)")
    print("-" * 108)
    ranked = [(b, c) for b, c in summary["blame_distribution"].items() if b != "none"]
    ranked.sort(key=lambda kv: (-kv[1], kv[0]))
    if not ranked:
        print("  none -- every probe passed.")
    for rank, (blame, count) in enumerate(ranked[:3], start=1):
        info = BOTTLENECK_FIXES.get(blame, {"source": "?", "fix": "?"})
        rows = summary["error_row_ids"].get(blame, [])
        print(f"\n#{rank}  blame={blame}   probes={count}   affected rows={rows[:20]}"
              f"{' ...' if len(rows) > 20 else ''}")
        print(f"    source : {info['source']}")
        print(f"    fix    : {info['fix']}")
    print("\n" + "=" * 108)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="e2e_rag_harness.py",
        description="End-to-end RAG harness for the Persian Way bot (local KB only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Test domain: sheet `%s`, pandas rows [0, N) where N = --limit (0 = all rows).\n"
            "The harness re-ingests exactly that slice into a run-private vector store; it\n"
            "never reads or writes the project's production vector store." % DEFAULT_SHEET
        ),
    )
    parser.add_argument("--xls", default=DEFAULT_XLS, help="ground-truth workbook path")
    parser.add_argument("--sheet", default=DEFAULT_SHEET, help="worksheet name")
    parser.add_argument("--limit", type=int, default=150,
                        help="number of rows in the test domain (default 150, 0 = all)")
    parser.add_argument("--probe-rows", type=int, default=0,
                        help="probe only the first N rows of the domain while still INGESTING "
                             "the whole domain (0 = probe all). Needed because retrieval is "
                             "meaningless when the corpus is smaller than top_k_results.")
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "reports"),
                        help="reports root; a timestamped run dir is created inside it")
    parser.add_argument("--variants", default=",".join(VARIANT_NAMES),
                        help="comma-separated subset of: " + ",".join(VARIANT_NAMES))
    parser.add_argument("--multiturn", action="store_true",
                        help="also run the three multi-turn context scenarios")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="seconds to sleep between pipeline calls (throttling)")
    parser.add_argument("--call-timeout", type=float, default=180.0,
                        help="per-probe ceiling in seconds (default 180); a timeout is "
                             "recorded as an error instead of stalling the run")
    parser.add_argument("--reuse-storage", default=None,
                        help="reuse an existing run storage dir and skip the ingest")
    parser.add_argument("--strict-errors", action="store_true",
                        help="abort on the first probe error instead of recording it")
    parser.add_argument("--validate-units", action="store_true",
                        help="only run the detector self-validation (offline, no API calls)")
    return parser


def _parse_variants(raw: str) -> List[str]:
    requested = [v.strip() for v in (raw or "").split(",") if v.strip()]
    unknown = [v for v in requested if v not in VARIANT_NAMES]
    if unknown:
        raise SystemExit(
            f"unknown variant(s) {unknown!r}; valid choices are {list(VARIANT_NAMES)}")
    if not requested:
        raise SystemExit("--variants must name at least one variant")
    return requested


async def _run(cfg: HarnessConfig, args: argparse.Namespace) -> int:
    # Imported after isolation, and *before* KbHandles.build(): knowledge_base.py:9
    # pulls get_llm from chat_service while chat_service.py:108 pulls
    # get_knowledge_base_service back, so whichever of the two is imported first
    # decides whether the pair resolves. Importing chat_service first is the order
    # that works; going straight to knowledge_base raises
    # "cannot import name 'get_knowledge_base_service' from partially initialized module".
    from app.services.chat_service import get_llm

    rows, meta = load_ground_truth(cfg.xls, cfg.sheet, cfg.limit)
    if not rows:
        raise SystemExit("test domain is empty -- nothing to evaluate")

    checks = validate_dose_detector(rows)
    detector_ok = print_validation_table(checks)
    if args.validate_units:
        return 0 if detector_ok else 1

    log_path = setup_logging(cfg.run_dir)
    print(f"[run] {cfg.run_dir}\n[log] {log_path}")
    print(f"[domain] sheet='{cfg.sheet}' rows={meta['domain_loaded_rows']} "
          f"of {meta['total_rows_in_sheet']} (limit={cfg.limit})")

    if args.reuse_storage:
        ingest = {"reused": True, "storage_root": cfg.storage_root}
        print(f"[ingest] reused existing storage at {cfg.storage_root}")
    else:
        ingest = ingest_subset(rows, cfg)

    handles = await KbHandles.build()
    handles_info = {
        "provider": handles.provider,
        "embedding_model": handles.model,
        "generation_model": handles.gen_model,
        "threshold": handles.threshold,
        "env_threshold_ignored": handles.env_threshold,
        "top_k": handles.top_k,
        "qa_match_threshold": handles.qa_match_threshold,
        "similarity_threshold": handles.similarity_threshold,
        "rag_temperature": handles.temperature,
        "human_referral_message": handles.referral_message,
        "embedding_storage": ingest.get("persist_directory") or cfg.storage_root,
    }
    print(f"[env] provider={handles.provider} embedding={handles.model} "
          f"generation={handles.gen_model} threshold={handles.threshold} "
          f"(env says {handles.env_threshold}) top_k={handles.top_k}")
    print(f"[env] human_referral_message={handles.referral_message!r}")

    # Boilerplate is measured on the ingested domain, not hard-coded.
    boilerplate = build_boilerplate_terms([r.answer for r in rows])
    print(f"[coverage] {len(boilerplate)} boilerplate terms filtered "
          f"(document frequency >= 30% of the domain)")

    cache_path = os.path.join(args.out, "variants_cache.json")
    cache: Dict[str, Any] = {}
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as handle:
                cache = json.load(handle)
        except Exception as exc:
            print(f"[variants] cache unreadable ({exc}); starting fresh")

    paraphrase_llm = None
    if "paraphrase" in cfg.variants:
        paraphrase_llm = await get_llm(model_name="qwen/qwen3-32b", temperature=0.0,
                                       max_tokens=400)

    records: List[Dict[str, Any]] = []
    probed_rows = rows if cfg.probe_rows <= 0 else rows[:min(cfg.probe_rows, len(rows))]
    total = len(probed_rows) * len(cfg.variants)
    done = 0
    csv_path = os.path.join(cfg.run_dir, "e2e_records.csv")
    print(f"[probes] corpus={len(rows)} rows, probing {len(probed_rows)} rows x "
          f"{len(cfg.variants)} variants = {total} probes")

    # Incremental sink: every probe is on disk before the next one starts, so a
    # long run or an interruption still leaves a usable CSV behind. The file is
    # rewritten once more at the end so blame refinement is reflected.
    sink = CsvSink(csv_path)
    try:
        for row in probed_rows:
            queries: Dict[str, str] = {"verbatim": row.question}
            if "paraphrase" in cfg.variants:
                text, _from_cache = await paraphrase_question(row.question, cache, paraphrase_llm)
                queries["paraphrase"] = text
            if "typo" in cfg.variants:
                queries["typo"] = build_typo(row.question, row.row_id)
            if "elliptical" in cfg.variants:
                queries["elliptical"] = build_elliptical(row.question)

            for name in cfg.variants:
                query = queries[name]
                record = await run_probe(
                    handles, cfg, boilerplate,
                    query=query, row=row, variant=name, turn_index=1, turn_kind="single",
                    # Elliptical refusal is CORRECT behaviour -> the S2 FP control.
                    expect_refusal=(name == "elliptical"),
                )
                record["gold_answer"] = row.answer
                record.update(_validate_variant(row.question, query))
                records.append(record)
                sink.add(record)
                done += 1
                print(f"  [{done}/{total}] row={row.row_id} {name} rank={record['gt_rank']} "
                      f"conf={record['confidence_score']} tier={record['tier']} "
                      f"blame={record['blame']} {record['latency_ms']}ms", flush=True)
                if cfg.sleep:
                    await asyncio.sleep(cfg.sleep)

        if cfg.multiturn:
            print("[multiturn] running implicit_reference / topic_change / chain3", flush=True)
            for record in await run_multiturn_scenarios(handles, cfg, boilerplate, probed_rows):
                records.append(record)
                sink.add(record)
    finally:
        sink.close()

    refine_blame_with_variants(records)
    if cache and not args.reuse_storage:
        write_json(cache_path, cache)

    summary = build_summary(records, meta, cfg, handles_info, checks, ingest)
    gt_path = os.path.join(cfg.run_dir, "ground_truth_domain.csv")
    json_path = os.path.join(cfg.run_dir, "e2e_summary.json")
    # Authoritative rewrite: the incremental file above held pre-refinement blame.
    write_csv(csv_path, records)
    write_ground_truth_csv(gt_path, rows)
    write_json(json_path, summary)

    print_console_report(summary, records)
    print("[artifacts]")
    for path in (csv_path, json_path, log_path, gt_path):
        print(f"  {path}")

    if args.strict_errors and summary["counts"]["errors"]:
        print(f"[strict] {summary['counts']['errors']} probe(s) raised -- failing the run")
        return 2
    return 0 if detector_ok else 3


def main() -> int:
    global PROBE_TIMEOUT_SECONDS
    args = build_parser().parse_args()
    PROBE_TIMEOUT_SECONDS = max(30.0, float(args.call_timeout))
    variants = _parse_variants(args.variants)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir, storage_root = create_run_dir(args.out, stamp)
    if args.reuse_storage:
        storage_root = os.path.abspath(args.reuse_storage)
        os.makedirs(storage_root, exist_ok=True)
    configure_isolation(storage_root)

    cfg = HarnessConfig(
        xls=os.path.abspath(args.xls),
        sheet=args.sheet,
        limit=args.limit,
        probe_rows=int(args.probe_rows),
        out=os.path.abspath(args.out),
        variants=variants,
        multiturn=bool(args.multiturn),
        sleep=float(args.sleep),
        storage_root=storage_root,
        run_dir=run_dir,
        stamp=stamp,
    )
    try:
        return asyncio.run(_run(cfg, args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

