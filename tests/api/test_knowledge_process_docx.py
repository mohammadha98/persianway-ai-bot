"""`POST /api/knowledge/process-docx` ingests the DOCX files in the docs directory.

The endpoint mirrors `/api/knowledge/process` (PDF) and
`/api/knowledge/process-excel`: the response is returned immediately and the
ingestion itself is scheduled as a background task. These tests pin the parts the
callers depend on -- the response shape, the fact that the ingest is offloaded off
the event loop, and the metadata keys the produced chunks must carry so that
`HybridRetrievalService` can actually find them (its dense/BM25 branches filter on
`entry_type` and `is_public`, so a DOCX ingested without them is invisible).
"""

import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.routes import knowledge_base as kb_routes
from app.services.document_processor import (
    DOCX_SUPPORT,
    DocumentProcessor,
    get_document_processor,
)
from app.services.knowledge_base import KnowledgeBaseService

from main import app

client = TestClient(app)
PROCESS_DOCX_URL = "/api/knowledge/process-docx"


class TestProcessDocxRoute:
    """HTTP contract of `POST /api/knowledge/process-docx`."""

    def test_returns_processing_status_and_schedules_background_task(self):
        """The route answers immediately and hands the ingest to a background task."""
        service = MagicMock()
        service.process_docx_files = AsyncMock(return_value=7)

        with patch.object(kb_routes, "get_knowledge_base_service", return_value=service):
            response = client.post(PROCESS_DOCX_URL)

        assert response.status_code == 200
        assert response.json() == {
            "message": "DOCX processing started in the background",
            "status": "processing",
        }
        # TestClient drains background tasks before returning, so the scheduled
        # coroutine has already been awaited at this point.
        service.process_docx_files.assert_awaited_once()

    def test_route_keeps_the_knowledge_prefix_of_its_siblings(self):
        """`/process-docx` sits next to `/process` and `/process-excel`.

        Asserted through the OpenAPI schema rather than `app.routes`: with
        FastAPI 0.138 included routers are wrapped in `_IncludedRouter`, which
        has no `path` and does not expose the nested routes directly.
        """
        paths = client.get("/openapi.json").json()["paths"]
        assert PROCESS_DOCX_URL in paths
        assert "post" in paths[PROCESS_DOCX_URL]
        assert "/api/knowledge/process" in paths
        assert "/api/knowledge/process-excel" in paths


class TestProcessDocxService:
    """`KnowledgeBaseService.process_docx_files` must not block the event loop."""

    @pytest.mark.asyncio
    async def test_runs_the_ingest_off_the_event_loop(self):
        # `__new__` instead of `__init__`: this test needs a single collaborator,
        # while the real constructor also builds the vector store/reranker.
        service = KnowledgeBaseService.__new__(KnowledgeBaseService)
        processor = MagicMock()
        caller_thread = threading.get_ident()
        seen = {}

        def _process_all_docx():
            seen["thread"] = threading.get_ident()
            return 3

        processor.process_all_docx.side_effect = _process_all_docx
        service.document_processor = processor

        result = await service.process_docx_files()

        assert result == 3
        processor.process_all_docx.assert_called_once_with()
        assert seen["thread"] != caller_thread, "the ingest must run in a worker thread"


def _write_docx(path, text: str) -> None:
    """Write a one-paragraph DOCX at `path` (real python-docx, no mocks)."""
    from docx import Document as RealDocxDocument

    document = RealDocxDocument()
    document.add_paragraph(text)
    document.save(str(path))


class TestProcessedDocxMetadata:
    """The produced chunks must satisfy the retrieval branches' filters."""

    @pytest.mark.skipif(not DOCX_SUPPORT, reason="python-docx not installed")
    def test_chunks_are_visible_to_the_docx_retrieval_branch(self, tmp_path):
        docx_path = tmp_path / "kood.docx"
        _write_docx(docx_path, "کود نیتروژن برای رشد برگ گیاه مفید است.")

        chunks = DocumentProcessor().process_docx(str(docx_path))

        assert chunks, "a non-empty DOCX must yield at least one chunk"
        metadata = chunks[0].metadata
        assert metadata["file_type"] == "docx"
        assert metadata["source_type"] == "docx"
        # These two are exactly what `hybrid_retrieval.py` filters on.
        assert metadata["entry_type"] == "user_contribution_docx"
        assert metadata["is_public"] is False

    @pytest.mark.skipif(not DOCX_SUPPORT, reason="python-docx not installed")
    def test_process_all_docx_pushes_retrievable_chunks_to_the_vector_store(self, tmp_path):
        _write_docx(tmp_path / "malat.docx", "مالت خرما یک قاشق مرباخوری صبح و شب میل شود.")

        processor = DocumentProcessor()
        processor.docs_dir = str(tmp_path)
        vector_store = MagicMock()

        with patch.object(processor, "get_vector_store", return_value=vector_store):
            count = processor.process_all_docx()

        assert count > 0
        added = [doc for call in vector_store.add_documents.call_args_list for doc in call.args[0]]
        assert added, "the chunks must reach the vector store"
        assert all(doc.metadata["entry_type"] == "user_contribution_docx" for doc in added)
        assert all(doc.metadata["is_public"] is False for doc in added)

    def test_singleton_is_still_shared(self):
        """The module-level accessor stays a singleton (regression guard)."""
        assert get_document_processor() is get_document_processor()


class TestProcessDocxEndToEnd:
    """Route -> service -> DocumentProcessor -> (fake) vector store."""

    @pytest.mark.skipif(not DOCX_SUPPORT, reason="python-docx not installed")
    def test_post_ingests_the_docs_directory_into_the_vector_store(self, tmp_path):
        _write_docx(tmp_path / "kood.docx", "کود نیتروژن برای رشد برگ گیاه مفید است.")

        processor = DocumentProcessor()
        processor.docs_dir = str(tmp_path)
        vector_store = MagicMock()

        # Real service object (`__new__` avoids the constructor's side effects)
        # wired to the real processor, so the whole delegation chain runs.
        service = KnowledgeBaseService.__new__(KnowledgeBaseService)
        service.document_processor = processor

        with patch.object(processor, "get_vector_store", return_value=vector_store), \
                patch.object(kb_routes, "get_knowledge_base_service", return_value=service):
            response = client.post(PROCESS_DOCX_URL)

        assert response.status_code == 200
        added = [doc for call in vector_store.add_documents.call_args_list for doc in call.args[0]]
        assert added, "the DOCX in the docs directory must reach the vector store"
        assert {doc.metadata["entry_type"] for doc in added} == {"user_contribution_docx"}
        assert {doc.metadata["is_public"] for doc in added} == {False}
