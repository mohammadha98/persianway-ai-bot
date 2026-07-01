import importlib
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def processor():
    """Create ExcelQAProcessor with document_processor import safely stubbed."""
    fake_doc_processor_module = types.ModuleType("app.services.document_processor")
    fake_doc_processor_module.get_document_processor = lambda: MagicMock()

    with patch.dict(
        sys.modules, {"app.services.document_processor": fake_doc_processor_module}
    ):
        excel_module = importlib.import_module("app.services.excel_processor")
        excel_module = importlib.reload(excel_module)
        return excel_module.ExcelQAProcessor()


def _mock_read_excel(df: pd.DataFrame):
    return patch("pandas.read_excel", return_value=df)


# Test 1: Existing QA Excel still works


def test_existing_qa_excel_still_works(processor):
    df = pd.DataFrame(
        {
            "Question": ["What is NPK?"],
            "Answer": ["Nitrogen, Phosphorus, Potassium"],
            "Title": ["Fertilizer Basics"],
        }
    )

    with _mock_read_excel(df):
        count, docs = processor.process_excel_file("sample.xlsx")

    assert count == 1
    assert docs[0].metadata["source_type"] == "excel_qa"
    assert docs[0].metadata["excel_mode"] == "qa"
    assert docs[0].metadata["title"] == "Fertilizer Basics"
    assert docs[0].metadata["question"] == "What is NPK?"
    assert docs[0].metadata["row_number"] == 2
    assert "Question:" in docs[0].page_content
    assert "Answer:" in docs[0].page_content


# Test 2: Persian QA Excel works


def test_persian_qa_excel_works(processor):
    df = pd.DataFrame(
        {
            "سوال": ["کود مناسب گندم چیست؟"],
            "پاسخ": ["کود ازته و فسفاته"],
            "عنوان": ["تغذیه گندم"],
        }
    )

    with _mock_read_excel(df):
        count, docs = processor.process_excel_file("persian.xlsx")

    assert count == 1
    assert docs[0].metadata["source_type"] == "excel_qa"
    assert docs[0].metadata["excel_mode"] == "qa"
    assert docs[0].metadata["title"] == "تغذیه گندم"
    assert docs[0].metadata["question"] == "کود مناسب گندم چیست؟"


# Test 3: Arbitrary organizational Excel works


def test_arbitrary_organizational_excel_fallback(processor):
    df = pd.DataFrame(
        {
            "Customer Name": ["Ali"],
            "Contract Number": [12345],
            "Status": ["Active"],
            "Notes": ["Important client"],
        }
    )

    with _mock_read_excel(df):
        count, docs = processor.process_excel_file("org.xlsx")

    assert count == 1
    doc = docs[0]
    assert doc.metadata["source_type"] == "excel_structured"
    assert doc.metadata["excel_mode"] == "structured_fallback"
    assert doc.metadata["title"]
    assert "Customer Name:\nAli" in doc.page_content
    assert "Contract Number:\n12345" in doc.page_content
    assert "Status:\nActive" in doc.page_content
    assert "Notes:\nImportant client" in doc.page_content


# Test 4: Missing QA columns should not fail


def test_missing_qa_columns_do_not_fail(processor):
    df = pd.DataFrame(
        {
            "Product": ["Seed"],
            "Price": [200],
            "Category": ["Input"],
        }
    )

    assert processor.validate_excel_structure(df) is True

    with _mock_read_excel(df):
        count, docs = processor.process_excel_file("product.xlsx")

    assert count == 1
    assert docs[0].metadata["excel_mode"] == "structured_fallback"


# Test 5: Empty rows are skipped


def test_empty_rows_are_skipped(processor):
    df = pd.DataFrame(
        {
            "Product": ["Seed", None, "Fertilizer"],
            "Price": [200, None, None],
            "Category": ["Input", None, "Soil"],
        }
    )

    with _mock_read_excel(df):
        count, docs = processor.process_excel_file("rows.xlsx")

    assert count == 2
    assert len(docs) == 2
    assert docs[0].metadata["row_number"] == 2
    assert docs[1].metadata["row_number"] == 4


# Test 6: Title generation


def test_title_generation_priority(processor):
    df = pd.DataFrame(
        {
            "Name": [None, "X" * 120, None],
            "Details": ["First row details", "Second details", None],
        }
    )

    with _mock_read_excel(df):
        count, docs = processor.process_excel_file("titlegen.xlsx")

    assert count == 2

    # Row 1: title_col exists but empty -> first non-empty meaningful cell fallback
    assert docs[0].metadata["title"] == "First row details"

    # Row 2: title col used and truncated to 80 chars
    assert len(docs[1].metadata["title"]) == 80


def test_title_fallback_format_when_no_meaningful_cell(processor):
    # Direct helper check for deterministic final fallback
    row = pd.Series({"A": None, "B": "   "})
    generated = processor._generate_title(
        row, row_number=7, file_name="file.xlsx", title_col=None
    )
    assert generated == "file.xlsx - Row 7"


# Test 7: No text-length heuristic


def test_no_text_length_heuristic(processor):
    long_text = "L" * 5000
    df = pd.DataFrame(
        {
            "Address": [long_text],
            "Logs": ["normal"],
            "Status": ["Open"],
        }
    )

    with _mock_read_excel(df):
        count, docs = processor.process_excel_file("longtext.xlsx")

    assert count == 1
    assert docs[0].metadata["excel_mode"] == "structured_fallback"
    assert docs[0].metadata["source_type"] == "excel_structured"


# Test 8: Backward compatibility metadata


def test_backward_compatibility_metadata_keys(processor):
    df = pd.DataFrame(
        {
            "Question": ["Q1"],
            "Answer": ["A1"],
            "Title": ["T1"],
        }
    )

    with _mock_read_excel(df):
        _, docs = processor.process_excel_file("compat.xlsx")

    md = docs[0].metadata
    # Existing keys kept
    assert "source" in md
    assert "file_path" in md
    assert "source_type" in md
    assert "title" in md
    assert "question" in md
    assert "row_number" in md
    # New mode indicator added
    assert md["excel_mode"] == "qa"


def test_explicit_mapping_overrides_synonyms(processor):
    df = pd.DataFrame(
        {
            "PromptX": ["What is pH?"],
            "ReplyX": ["Acidity scale"],
            "HeadingX": ["Chemistry"],
        }
    )

    with _mock_read_excel(df):
        count, docs = processor.process_excel_file(
            "mapped.xlsx",
            column_mapping={
                "question": "PromptX",
                "answer": "ReplyX",
                "title": "HeadingX",
            },
        )

    assert count == 1
    assert docs[0].metadata["excel_mode"] == "qa"
    assert docs[0].metadata["title"] == "Chemistry"
    assert docs[0].metadata["question"] == "What is pH?"
