import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain.schema import Document

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from app.core.config import settings
from app.services.document_processor import get_document_processor


class ExcelQAProcessor:
    """Service for processing Excel QA files and creating vector embeddings.

    This service handles loading Excel files with QA pairs, creating embeddings,
    and storing them in the vector database alongside PDF document chunks.
    """

    TITLE_SYNONYMS = {
        "title",
        "subject",
        "topic",
        "name",
        "عنوان",
        "موضوع",
        "نام",
    }
    QUESTION_SYNONYMS = {
        "question",
        "query",
        "prompt",
        "پرسش",
        "سوال",
        "سؤال",
    }
    ANSWER_SYNONYMS = {
        "answer",
        "response",
        "reply",
        "content",
        "description",
        "text",
        "body",
        "پاسخ",
        "جواب",
        "توضیحات",
        "شرح",
        "متن",
        "محتوا",
    }

    def __init__(self):
        """Initialize the Excel QA processor."""
        # Get the document processor to access embeddings and vector store
        self.document_processor = get_document_processor()

        # Set up the Excel directory path using STORAGE_ROOT
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        _storage_root = (
            settings.STORAGE_ROOT.strip() if settings.STORAGE_ROOT else _project_root
        )
        self.excel_dir = os.path.join(_storage_root, "docs")

    def _normalize_text(self, value: Any) -> str:
        """Normalize Persian/English text for reliable column matching."""
        text = str(value).strip()
        text = text.replace("ي", "ی").replace("ك", "ک")
        text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        text = text.replace("ؤ", "و").replace("ئ", "ی").replace("ة", "ه")
        return text.lower()

    def _is_empty_value(self, value: Any) -> bool:
        """Return True for empty/null/whitespace-like values."""
        if pd.isna(value):
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        return False

    def _safe_cell_to_string(self, value: Any) -> str:
        """Convert a cell to user-friendly string value."""
        if self._is_empty_value(value):
            return ""
        return str(value).strip()

    def _normalized_column_index(self, df: pd.DataFrame) -> Dict[str, str]:
        """Map normalized column names to actual DataFrame column names."""
        index: Dict[str, str] = {}
        for col in df.columns:
            index[self._normalize_text(col)] = col
        return index

    def _resolve_mapped_column(
        self, mapping_value: Optional[str], normalized_index: Dict[str, str]
    ) -> Optional[str]:
        """Resolve an explicit mapping column name with normalization support."""
        if not mapping_value:
            return None
        normalized = self._normalize_text(mapping_value)
        return normalized_index.get(normalized)

    def _find_by_synonyms(
        self, normalized_index: Dict[str, str], synonyms: set
    ) -> Optional[str]:
        """Find the first matching column for a synonym set."""
        for synonym in synonyms:
            col = normalized_index.get(self._normalize_text(synonym))
            if col is not None:
                return col
        return None

    def detect_excel_columns(
        self, df: pd.DataFrame, column_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Optional[str]]:
        """Detect Excel processing mode and key columns.

        Priority:
          1) Explicit mapping from caller
          2) Synonym-based detection
          3) Structured fallback mode
        """
        column_mapping = column_mapping or {}
        normalized_index = self._normalized_column_index(df)

        title_col = self._resolve_mapped_column(
            column_mapping.get("title"), normalized_index
        )
        question_col = self._resolve_mapped_column(
            column_mapping.get("question"), normalized_index
        )
        answer_col = self._resolve_mapped_column(
            column_mapping.get("answer") or column_mapping.get("content"),
            normalized_index,
        )

        if title_col is None:
            title_col = self._find_by_synonyms(normalized_index, self.TITLE_SYNONYMS)
        if question_col is None:
            question_col = self._find_by_synonyms(
                normalized_index, self.QUESTION_SYNONYMS
            )
        if answer_col is None:
            answer_col = self._find_by_synonyms(normalized_index, self.ANSWER_SYNONYMS)

        if answer_col and (question_col or title_col):
            return {
                "mode": "qa",
                "title_col": title_col,
                "question_col": question_col,
                "answer_col": answer_col,
            }

        return {
            "mode": "structured_fallback",
            "title_col": title_col,
            "question_col": None,
            "answer_col": None,
        }

    def _generate_title(
        self,
        row: pd.Series,
        row_number: int,
        file_name: str,
        title_col: Optional[str] = None,
    ) -> str:
        """Generate a deterministic non-empty title for each row."""
        if title_col and title_col in row.index:
            title_value = self._safe_cell_to_string(row[title_col])
            if title_value:
                return title_value[:80]

        for col in row.index:
            value = self._safe_cell_to_string(row[col])
            if value:
                return value[:80]

        return f"{file_name} - Row {row_number}"

    def _row_has_data(self, row: pd.Series) -> bool:
        """Check whether a row has at least one non-empty cell."""
        for value in row.values:
            if not self._is_empty_value(value):
                return True
        return False

    def validate_excel_structure(
        self, df: pd.DataFrame, column_mapping: Optional[Dict[str, str]] = None
    ) -> bool:
        """Validate basic Excel readability/structure.

        Validation fails only when:
          - dataframe is empty
          - dataframe has no columns
          - all rows are effectively empty
        """
        if df is None:
            print("Error: Could not read Excel data")
            return False

        if len(df.columns) == 0:
            print("Error: Excel file has no columns")
            return False

        if df.empty:
            print("Error: Excel file is empty")
            return False

        has_any_data = any(self._row_has_data(row) for _, row in df.iterrows())
        if not has_any_data:
            print("Error: Excel file contains no non-empty rows")
            return False

        return True

    def process_excel_file(
        self, file_path: str, column_mapping: Optional[Dict[str, str]] = None
    ) -> Tuple[int, List[Document]]:
        """Process a single Excel file and convert rows to LangChain documents.

        Supports two modes:
          - qa: when answer + (question/title) columns are detected by mapping/synonyms.
          - structured_fallback: for arbitrary column names.
        """
        try:
            df = pd.read_excel(file_path, engine="openpyxl")

            if not self.validate_excel_structure(df, column_mapping=column_mapping):
                return 0, []

            file_name = os.path.basename(file_path)
            detected = self.detect_excel_columns(df, column_mapping=column_mapping)
            mode = detected["mode"]
            title_col = detected.get("title_col")
            question_col = detected.get("question_col")
            answer_col = detected.get("answer_col")

            logging.info(
                "Excel processing mode=%s file=%s title_col=%s question_col=%s answer_col=%s",
                mode,
                file_name,
                title_col,
                question_col,
                answer_col,
            )

            documents: List[Document] = []
            skipped_rows = 0

            for idx, row in df.iterrows():
                row_number = (
                    idx + 2
                )  # +2 because pandas is 0-indexed and Excel has a header row

                if not self._row_has_data(row):
                    skipped_rows += 1
                    continue

                if mode == "qa":
                    question = ""
                    answer = ""

                    if question_col and question_col in row.index:
                        question = self._safe_cell_to_string(row[question_col])
                    if not question and title_col and title_col in row.index:
                        question = self._safe_cell_to_string(row[title_col])

                    if answer_col and answer_col in row.index:
                        answer = self._safe_cell_to_string(row[answer_col])

                    if not answer or not question:
                        skipped_rows += 1
                        continue

                    title = self._generate_title(
                        row, row_number, file_name, title_col=title_col
                    )
                    content = f"Question:\n{question}\n\nAnswer:\n{answer}"

                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": file_name,
                            "file_path": file_path,
                            "source_type": "excel_qa",
                            "excel_mode": "qa",
                            "title": title,
                            "question": question,
                            "row_number": row_number,
                        },
                    )
                    documents.append(doc)
                else:
                    structured_parts: List[str] = []
                    for col in df.columns:
                        cell_value = self._safe_cell_to_string(row[col])
                        if not cell_value:
                            continue
                        structured_parts.append(f"{str(col).strip()}:\n{cell_value}")

                    if not structured_parts:
                        skipped_rows += 1
                        continue

                    content = "\n\n".join(structured_parts)
                    title = self._generate_title(
                        row, row_number, file_name, title_col=title_col
                    )

                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": file_name,
                            "file_path": file_path,
                            "source_type": "excel_structured",
                            "excel_mode": "structured_fallback",
                            "title": title,
                            "row_number": row_number,
                        },
                    )
                    documents.append(doc)

            logging.info(
                "Excel processed file=%s generated_docs=%s skipped_rows=%s mode=%s",
                file_name,
                len(documents),
                skipped_rows,
                mode,
            )

            return len(documents), documents
        except Exception as e:
            print(f"Error processing Excel file {file_path}: {str(e)}")
            return 0, []

    def process_all_excel_files(self) -> int:
        """Process all Excel files in the docs directory and add them to the vector store.

        Returns:
            Number of QA pairs processed
        """
        all_docs = []
        total_qa_pairs = 0

        # Get all Excel files in the docs directory
        excel_files = [
            os.path.join(self.excel_dir, f)
            for f in os.listdir(self.excel_dir)
            if f.lower().endswith((".xlsx", ".xls"))
        ]

        # Process each Excel file
        for excel_file in excel_files:
            qa_count, docs = self.process_excel_file(excel_file)
            total_qa_pairs += qa_count
            all_docs.extend(docs)

        # Add documents to vector store in batches to avoid token limit issues
        if all_docs:
            vector_store = self.document_processor.get_vector_store()

            # Process in batches of 100 documents to stay well under the 300k token limit
            batch_size = 100
            for i in range(0, len(all_docs), batch_size):
                batch = all_docs[i : i + batch_size]
                vector_store.add_documents(batch)
                print(
                    f"Processed batch {i // batch_size + 1}/{(len(all_docs) + batch_size - 1) // batch_size} with {len(batch)} documents"
                )

            vector_store.persist()

        return total_qa_pairs


# Singleton instance
_excel_qa_processor = None


def get_excel_qa_processor():
    """Get the Excel QA processor instance.

    Returns:
        A singleton instance of the ExcelQAProcessor
    """
    global _excel_qa_processor
    if _excel_qa_processor is None:
        _excel_qa_processor = ExcelQAProcessor()
    return _excel_qa_processor
