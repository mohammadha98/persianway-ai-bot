"""Unit tests for the query-time Excel Q&A boilerplate sanitizer.

The retrieval path must not hand the LLM a trainer greeting or a CTA closer: the
model reproduces the most salient opening of its context, so those lines leak
verbatim into user-visible answers. `sanitize_chunk` / `sanitize_documents` strip
them at query time (no re-ingestion) and are wired into
`KnowledgeBaseService._normalize_documents_for_context`.

The variants below are taken from the live corpus (`docs/*.xls`, 8918 rows), not
invented: `برشما`/`سفیرسلامت` spellings, the `شرکت پرشین وی` truncation, trailing
emoji, appended `همکار گرامی`, and answers whose body shares the greeting's line.
"""
import os
import sys

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from langchain_core.documents import Document

from app.services.text_sanitizer import (
    MIN_CONTENT_CHARS,
    sanitize_chunk,
    sanitize_documents,
)

# --- Real greeting-only lines (must be removed entirely) ---------------------

GREETING_ONLY_LINES = [
    'درود بر شما سفیر سلامت شرکت پرشین وی',
    'درود برشما سفیر سلامت شرکت پرشین وی',
    'درود برشما سفیر سلامت شرکت پرشین وی🌹',
    'درود بر شما سفیرسلامت شرکت پرشین وی',
    'درود بر شما سفیرسلامت شرکت پرشین وی همکار گرامی',
    'درود بر شما سفیرسلامت شرکت پرشین وی همکار محترم',
    'درود برشما سفیر سلامت پرشین وی',
    'درود بر سفیر سلامت شرکت پرشین وی',
    'درود بر شما سفیر سلامت شرکت پرشین وی.',
    'درود بر شما سفیر سلامت شرکت پرشین',
    # Leading punctuation / harakat noise seen in the corpus.
    '"درود بر شما سفیر سلامت شرکت پرشین وی',
    'ٰدرود بر شما سفیر سلامت شرکت پرشین وی🌹',
    '. درود بر شما سفیر سلامت شرکت پرشین وی',
    'درود بر شما',
    # Other greeting families.
    'با سلام',
    'با سلام و احترام',
    'سلام و درود',
    'سلام ودرود',
    'سلام وقت بخیر',
    'سلام وقت بخیر، همکار گرامی',
    'همکار گرامی',
    'همکار محترم',
]


@pytest.mark.parametrize('line', GREETING_ONLY_LINES)
def test_greeting_only_line_is_removed(line):
    cleaned, removed = sanitize_chunk(line)
    assert cleaned == ''
    assert removed == 1


# --- Boilerplate that must be removed but whose content must survive ---------

MERGE_CASES = [
    # greeting glued to the answer body
    (
        'درود بر شما سفیرسلامت شرکت پرشین وی همکار محترم  پیشنهاد میشه از حجم کم شروع کنید.',
        'پیشنهاد میشه از حجم کم شروع کنید.',
    ),
    (
        'درود بر شما سفیرسلامت شرکت پرشین وی، برای دست می توانید استفاده کنید.',
        'برای دست می توانید استفاده کنید.',
    ),
    (
        'درود بر شما سفیر سلامت شرکت پرشین وی. همکار گرامی به زودی شارژ میشود. با تشکر پرشین وی',
        'به زودی شارژ میشود.',
    ),
    # signature inside the line
    (
        'درود بر شما سفیرسلامت شرکت پرشین وی مطالعه کنید. باتشکر پرشین وی',
        'مطالعه کنید.',
    ),
]


@pytest.mark.parametrize('raw,expected', MERGE_CASES)
def test_greeting_prefix_stripped_but_body_preserved(raw, expected):
    cleaned, removed = sanitize_chunk(raw)
    assert cleaned == expected
    assert removed == 1


# --- Whole-line closers / annotations ---------------------------------------

WHOLE_LINE_BOILERPLATE = [
    # CTA closers (exact shape, with and without the trailing emoji).
    'اگر خواستید، می\u200cتونم بر اساس شرایط دقیق فرد (سن، داروها، مزاج، وزن) برنامه منظم\u200cتری هم براتون تنظیم کنم 🌱',
    'اگر خواستید، می\u200cتونم دقیق\u200cتر بر اساس علائم (یبوست، اسهال، نفخ شدید و...) راهنمایی\u200cتون کنم 🌱',
    'اگر بخواید میتونم براتون برنامه بدهم.',
    'اگر تمایل داشته باشید میتونم راهنماییتون کنم.',
    'اگر مایل بودید بگید تا بیشتر توضیح بدم.',
    # Referral fallbacks.
    'درخواست شما به واحد مربوطه ارجاع داده شد و به زودی پاسختون رو دریافت می\u200cکنید.',
    'درخواست شما به واحد مربوطه ارجاع شد.',
    'تیکت شما به واحد مربوطه انتقال داده می شود.',
    'همکار گرامی تیکت شما با شماره تیکت 430964 به واحد مربوطه انتقال داده شده است .',
    'اطلاعات کافی در پایگاه دانش برای این سوال یافت نشد.',
    'متاسفانه اطلاعات کافی در این مورد وجود ندارد.',
    'اطلاعات دقیقی در دسترس نیست.',
    # Ticket acknowledgements.
    'تیکت شما پاسخ داده شده است.',
    'تیکت شما دوبار ارسال شده است پاسخ در تیکت قبلی داده خواهد شد.',
    # Trainer annotations.
    'نیاز به نگارش این متن نمی باشد',
    'لازم به نگارش نیست',
]


@pytest.mark.parametrize('line', WHOLE_LINE_BOILERPLATE)
def test_whole_line_boilerplate_is_removed(line):
    cleaned, removed = sanitize_chunk(line)
    assert cleaned == '', f'unexpected remainder: {cleaned!r}'
    assert removed == 1


def test_greeting_plus_ticket_ack_drops_entire_line():
    """A greeting in front of a ticket ack must be recognised in the same pass."""
    raw = 'با سلام همکار گرامی، تیکت شما دوبار ارسال شده است پاسخ در تیکت قبلی داده خواهد شد. باتشکر پرشین وی'
    cleaned, removed = sanitize_chunk(raw)
    assert cleaned == ''
    assert removed == 1


def test_signature_before_greeting_drops_entire_line():
    """`با تشکر پرشین وی درود بر شما ...` exists in the corpus verbatim."""
    raw = 'با تشکر پرشین وی درود بر شما سفیر سلامت شرکت پرشین وی'
    cleaned, removed = sanitize_chunk(raw)
    assert cleaned == ''
    assert removed == 1


# --- False positives: real content must never be touched --------------------

MUST_PRESERVE = [
    # `با سلام` must not bite into the inside of `سلامتی`.
    'با سلامتی کامل میتونید مصرف کنید.',
    'برای بهبود سلامتی این موارد توصیه می‌شود.',
    'سلامت پوست به تغذیه وابسته است.',
    # `درود` in a non-greeting position (mid-line).
    'کود اوره برای رشد رویشی گیاه استفاده می‌شود. درود بر شما مهمان عزیز',
    # A real 400+ char answer that a broader `نیاز به ارسال` rule deleted.
    'نیاز به ارسال آزمایشات نیست.راهکاری که در فایل بیشتر بدانیم در مورد مشکلات گوارشی مطرح شده',
    # Real guidance mentioning a referral mid-sentence.
    'پیشنهاد میکنم با مطب دکتر فهیمی که متخصص طب ایرانی هستند تماس بگیرید.',
    # Ordinary answers.
    'مالت خرما یک قاشق مرباخوری صبح و شب میل شود.',
    'استفاده از کامبوچا برای بهبود گوارش مفید است.',
    'درود بر سفیران سلامت و همکاران محترم فروش',
]


@pytest.mark.parametrize('text', MUST_PRESERVE)
def test_real_content_is_preserved(text):
    cleaned, removed = sanitize_chunk(text)
    assert cleaned == text
    assert removed == 0


# --- Structure, idempotency and document handling ---------------------------

CHUNK_WITH_GREETING = (
    'Question:\n'
    'کاربرد قهوه سبز چیست؟\n'
    '\n'
    'Answer:\n'
    'درود بر شما سفیرسلامت شرکت پرشین وی\n'
    '\n'
    'قهوه سبز سرشار از آنتی اکسیدان هست.\n'
    'اگر خواستید، می\u200cتونم برنامه منظم\u200cتری هم براتون تنظیم کنم 🌱'
)


def test_chunk_greeting_and_cta_removed_body_and_structure_survive():
    cleaned, removed = sanitize_chunk(CHUNK_WITH_GREETING)
    # The greeting-only line is dropped; the blank separator that followed it is
    # preserved (collapsing it away would differ between the first and second
    # pass and break idempotency).
    assert cleaned == (
        'Question:\n'
        'کاربرد قهوه سبز چیست؟\n'
        '\n'
        'Answer:\n'
        '\n'
        'قهوه سبز سرشار از آنتی اکسیدان هست.'
    )
    assert removed == 2
    assert 'درود' not in cleaned
    assert 'اگر خواستید' not in cleaned


def test_sanitize_is_idempotent():
    for text in [CHUNK_WITH_GREETING] + GREETING_ONLY_LINES + MUST_PRESERVE:
        first, _ = sanitize_chunk(text)
        second, removed_second = sanitize_chunk(first)
        assert second == first
        assert removed_second == 0


def test_sanitize_chunk_handles_empty_and_none_like_input():
    assert sanitize_chunk('') == ('', 0)
    assert sanitize_chunk('   \n  \n ') == ('', 0)


def test_sanitize_documents_preserves_metadata_and_order():
    docs = [
        Document(page_content=CHUNK_WITH_GREETING, metadata={'source': 'a.xls', 'page': 1, 'is_public': True}),
        Document(page_content='مالت خرما یک قاشق مرباخوری صبح و شب میل شود.', metadata={'source': 'b.xls'}),
    ]
    out = sanitize_documents(docs)
    assert len(out) == 2
    assert [d.metadata['source'] for d in out] == ['a.xls', 'b.xls']
    assert out[0].metadata == {'source': 'a.xls', 'page': 1, 'is_public': True}
    assert out[1].page_content == docs[1].page_content
    assert sanitize_chunk(out[0].page_content)[0] == out[0].page_content


def test_sanitize_documents_returns_new_objects_and_keeps_scores_out():
    docs = [Document(page_content=CHUNK_WITH_GREETING, metadata={'source': 'a.xls'})]
    out = sanitize_documents(docs)
    assert out[0] is not docs[0]
    assert docs[0].page_content == CHUNK_WITH_GREETING  # input untouched


def test_sanitize_documents_drops_greeting_only_chunk():
    docs = [
        Document(page_content='درود بر شما سفیر سلامت شرکت پرشین وی', metadata={'source': 'greeting.xls'}),
        Document(page_content='مالت خرما یک قاشق مرباخوری صبح و شب میل شود.', metadata={'source': 'b.xls'}),
    ]
    out = sanitize_documents(docs)
    assert len(out) == 1
    assert out[0].metadata['source'] == 'b.xls'


def test_sanitize_documents_drops_chunks_below_min_content():
    docs = [Document(page_content='درود بر شما' + ' ' + 'a' * (MIN_CONTENT_CHARS - 3), metadata={})]
    assert sanitize_documents(docs) == []


def test_sanitize_documents_empty_input():
    assert sanitize_documents([]) == []
    assert sanitize_documents(None) == []


# --- Integration: the KB normalisation chokepoint really sanitizes ----------

def test_knowledge_base_normalization_strips_boilerplate():
    """`_normalize_documents_for_context` is where both answer paths get context.

    `_retrieve_context` returns these docs, the streaming path feeds them to
    `stream_answer_from_context`, and the non-streaming path uses them as sources
    -- so asserting on the helper covers both consumers.
    """
    from unittest.mock import patch

    with patch('app.services.chat_service.get_llm'):
        with patch('app.services.document_processor.get_document_processor'):
            with patch('app.services.excel_processor.get_excel_qa_processor'):
                from app.services.knowledge_base import KnowledgeBaseService

    service = KnowledgeBaseService()
    docs = [
        Document(
            page_content=(
                'Question:\nکاربرد قهوه سبز چیست؟\n\nAnswer:\n'
                'درود بر شما سفیرسلامت شرکت پرشین وی\n'
                'قهوه سبز سرشار از آنتی اکسیدان هست.'
            ),
            metadata={'source': 'qa.xls', 'source_type': 'excel_qa', 'is_public': True},
        ),
    ]

    normalized = service._normalize_documents_for_context(docs)

    assert len(normalized) == 1
    content = normalized[0].page_content
    assert 'درود بر' not in content
    assert 'سفیرسلامت' not in content
    assert 'قهوه سبز سرشار از آنتی اکسیدان هست.' in content
    # metadata survives the sanitizer untouched
    assert normalized[0].metadata == {'source': 'qa.xls', 'source_type': 'excel_qa', 'is_public': True}


def test_knowledge_base_normalization_drops_greeting_only_doc():
    from unittest.mock import patch

    with patch('app.services.chat_service.get_llm'):
        with patch('app.services.document_processor.get_document_processor'):
            with patch('app.services.excel_processor.get_excel_qa_processor'):
                from app.services.knowledge_base import KnowledgeBaseService

    service = KnowledgeBaseService()
    docs = [
        Document(
            page_content='درود بر شما سفیر سلامت شرکت پرشین وی',
            metadata={'source': 'empty.xls', 'source_type': 'excel_qa'},
        ),
        Document(
            page_content='مالت خرما یک قاشق مرباخوری صبح و شب میل شود.',
            metadata={'source': 'real.pdf', 'source_type': 'pdf'},
        ),
    ]

    normalized = service._normalize_documents_for_context(docs)

    assert len(normalized) == 1
    assert normalized[0].metadata['source'] == 'real.pdf'
