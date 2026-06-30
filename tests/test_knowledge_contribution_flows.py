"""
Tests for the two knowledge contribution approaches:

Approach 1 – sync_mongodb_to_vectordb
    Reads existing documents from MongoDB and writes them to the vectordb ONLY.
    Must NOT create new MongoDB records (no duplicates) and must preserve the
    original hash_id that already exists in the database.

Approach 2 – /contribute (add_knowledge_contribution + background task)
    add_knowledge_contribution  -> inserts into MongoDB, queues a task.
                                   Must NOT write to vectordb at this stage.
    process_knowledge_contribution_background
                                -> writes to vectordb, updates sync flag.
                                   Must NOT insert a second MongoDB document.
"""

import pytest
import sys
import os
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# anyio is the async test runner available in this project
pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Shared lightweight mocks
# ---------------------------------------------------------------------------

class MockDocument:
    """Mimics a langchain_core.documents.Document."""

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = dict(metadata)

    def __repr__(self):
        return f"MockDocument(title={self.metadata.get('title')!r})"


class MockTextSplitter:
    """Splits text into one chunk per input (sufficient for unit tests)."""

    def create_documents(self, texts, metadatas):
        return [MockDocument(text, meta) for text, meta in zip(texts, metadatas)]


class MockVectorStore:
    """Records every add_documents call for later assertion."""

    def __init__(self):
        self.added_batches: list[list[MockDocument]] = []

    def add_documents(self, docs):
        self.added_batches.append(list(docs))
        return [str(uuid.uuid4()) for _ in docs]

    @property
    def total_added(self) -> int:
        return sum(len(b) for b in self.added_batches)


class MockDocumentProcessorForKB:
    """Minimal DocumentProcessor stand-in used inside KnowledgeBaseService tests."""

    def __init__(self, vector_store=None):
        self.vector_store = vector_store or MockVectorStore()
        self.text_splitter = MockTextSplitter()

    def get_vector_store(self):
        return self.vector_store


# ---------------------------------------------------------------------------
# Helpers to build the module-under-test with external deps patched out
# ---------------------------------------------------------------------------

def _make_kb_service(mock_doc_processor=None):
    """Return a KnowledgeBaseService with all heavy dependencies mocked."""
    with patch("app.services.chat_service.get_llm"):
        with patch("app.services.document_processor.get_document_processor"):
            with patch("app.services.excel_processor.get_excel_qa_processor"):
                from app.services.knowledge_base import KnowledgeBaseService

    service = KnowledgeBaseService.__new__(KnowledgeBaseService)
    service.document_processor = mock_doc_processor or MockDocumentProcessorForKB()
    service.excel_processor = MagicMock()
    service.config_service = MagicMock()
    service._qa_chain = None
    return service


# ---------------------------------------------------------------------------
# APPROACH 2 – add_knowledge_contribution (the synchronous part)
# ---------------------------------------------------------------------------

class TestAddKnowledgeContribution:
    """
    add_knowledge_contribution should:
      ✓ Create a task record
      ✓ Insert exactly ONE document into MongoDB
      ✗ NOT write anything to the vectordb (background task owns that)
    """

    async def test_inserts_into_mongodb_exactly_once(self):
        vector_store = MockVectorStore()
        service = _make_kb_service(MockDocumentProcessorForKB(vector_store))

        mock_task_service = AsyncMock()
        mock_task_service.create_task = AsyncMock(return_value="task-123")

        mock_db_service = AsyncMock()
        mock_db_service.insert_knowledge_document = AsyncMock(return_value="db-id-1")

        with patch(
            "app.services.task_service.get_task_service",
            return_value=mock_task_service,
        ):
            with patch(
                "app.services.database.get_database_service",
                return_value=mock_db_service,
            ):
                result = await service.add_knowledge_contribution(
                    title="Test Title",
                    content="Test content in Persian",
                    meta_tags=["tag1", "tag2"],
                    source="unit-test",
                    author_name="Tester",
                    additional_references="https://example.com",
                    is_public=False,
                )

        # MongoDB insert called exactly once
        mock_db_service.insert_knowledge_document.assert_called_once()

        # Task was created
        mock_task_service.create_task.assert_called_once()

        # Response has required fields
        assert result["id"]
        assert result["task_id"] == "task-123"
        assert result["db_id"] == "db-id-1"
        assert result["status"] == "queued"

    async def test_does_not_write_to_vectordb(self):
        """The premature vectordb write was removed; only the background task writes."""
        vector_store = MockVectorStore()
        service = _make_kb_service(MockDocumentProcessorForKB(vector_store))

        mock_task_service = AsyncMock()
        mock_task_service.create_task = AsyncMock(return_value="task-456")
        mock_db_service = AsyncMock()
        mock_db_service.insert_knowledge_document = AsyncMock(return_value="db-id-2")

        with patch(
            "app.services.task_service.get_task_service",
            return_value=mock_task_service,
        ):
            with patch(
                "app.services.database.get_database_service",
                return_value=mock_db_service,
            ):
                await service.add_knowledge_contribution(
                    title="No VectorDB Write",
                    content="This should not land in vectordb yet",
                    meta_tags=["test"],
                )

        assert vector_store.total_added == 0, (
            f"vectordb should not be written during add_knowledge_contribution, "
            f"but {vector_store.total_added} document(s) were added"
        )

    async def test_uploaded_file_path_stored_in_db(self):
        """File metadata should be persisted in the MongoDB document."""
        service = _make_kb_service()

        mock_task_service = AsyncMock()
        mock_task_service.create_task = AsyncMock(return_value="task-789")
        mock_db_service = AsyncMock()
        mock_db_service.insert_knowledge_document = AsyncMock(return_value="db-id-3")

        with patch(
            "app.services.task_service.get_task_service",
            return_value=mock_task_service,
        ):
            with patch(
                "app.services.database.get_database_service",
                return_value=mock_db_service,
            ):
                await service.add_knowledge_contribution(
                    title="File Upload Test",
                    content="Content here",
                    meta_tags=["file"],
                    uploaded_file_path="/docs/report.pdf",
                )

        saved_doc = mock_db_service.insert_knowledge_document.call_args[0][0]
        assert saved_doc.get("file_path") == "/docs/report.pdf"
        assert saved_doc.get("file_type") == "pdf"
        assert saved_doc.get("file_name") == "report.pdf"


# ---------------------------------------------------------------------------
# APPROACH 2 – process_knowledge_contribution_background
# ---------------------------------------------------------------------------

class TestProcessKnowledgeContributionBackground:
    """
    process_knowledge_contribution_background should:
      ✓ Write to the vectordb (text-split documents)
      ✗ NOT insert a new MongoDB document (only update sync flag)
    """

    async def test_writes_to_vectordb_exactly_once(self):
        vector_store = MockVectorStore()
        service = _make_kb_service(MockDocumentProcessorForKB(vector_store))

        hash_id = str(uuid.uuid4())

        mock_task_service = AsyncMock()
        mock_task_service.update_task_status = AsyncMock()
        mock_db_service = AsyncMock()
        mock_db_service.insert_knowledge_document = AsyncMock()
        mock_db_service.update_knowledge_document_sync_status = AsyncMock()

        with patch(
            "app.services.task_service.get_task_service",
            return_value=mock_task_service,
        ):
            with patch(
                "app.services.database.get_database_service",
                return_value=mock_db_service,
            ):
                await service.process_knowledge_contribution_background(
                    task_id="task-bg-1",
                    knowledge_hash_id=hash_id,
                    metadata={
                        "title": "Background Task Title",
                        "content": "Content to be embedded",
                        "source": "unit-test",
                        "author_name": "Tester",
                        "additional_references": None,
                        "uploaded_file_path": None,
                        "is_public": False,
                        "meta_tags": ["agri", "soil"],
                    },
                )

        # Vectordb received at least one batch
        assert len(vector_store.added_batches) >= 1, "Background task must write to vectordb"
        assert vector_store.total_added >= 1

        # Metadata on written docs should carry the original hash_id
        all_docs = [d for batch in vector_store.added_batches for d in batch]
        assert all(
            d.metadata.get("hash_id") == hash_id for d in all_docs
        ), "All vectordb docs should carry the original hash_id"

    async def test_does_not_insert_into_mongodb(self):
        """The background task must update the sync flag but never insert a new doc."""
        vector_store = MockVectorStore()
        service = _make_kb_service(MockDocumentProcessorForKB(vector_store))

        mock_task_service = AsyncMock()
        mock_task_service.update_task_status = AsyncMock()
        mock_db_service = AsyncMock()
        mock_db_service.insert_knowledge_document = AsyncMock()
        mock_db_service.update_knowledge_document_sync_status = AsyncMock()

        with patch(
            "app.services.task_service.get_task_service",
            return_value=mock_task_service,
        ):
            with patch(
                "app.services.database.get_database_service",
                return_value=mock_db_service,
            ):
                await service.process_knowledge_contribution_background(
                    task_id="task-bg-2",
                    knowledge_hash_id=str(uuid.uuid4()),
                    metadata={
                        "title": "No DB Insert",
                        "content": "Should not create new MongoDB record",
                        "source": None,
                        "author_name": None,
                        "additional_references": None,
                        "uploaded_file_path": None,
                        "is_public": False,
                        "meta_tags": [],
                    },
                )

        mock_db_service.insert_knowledge_document.assert_not_called()
        mock_db_service.update_knowledge_document_sync_status.assert_called_once()

    async def test_no_duplicate_vectordb_writes_across_both_steps(self):
        """
        Simulate the full /contribute flow:
          1. add_knowledge_contribution  (must not touch vectordb)
          2. process_knowledge_contribution_background (must write vectordb exactly once)

        Total vectordb write count must be exactly 1 (from background task only).
        """
        vector_store = MockVectorStore()
        service = _make_kb_service(MockDocumentProcessorForKB(vector_store))

        hash_id = str(uuid.uuid4())

        mock_task_service = AsyncMock()
        mock_task_service.create_task = AsyncMock(return_value="task-combined")
        mock_task_service.update_task_status = AsyncMock()
        mock_db_service = AsyncMock()
        mock_db_service.insert_knowledge_document = AsyncMock(return_value="db-combined")
        mock_db_service.update_knowledge_document_sync_status = AsyncMock()

        with patch(
            "app.services.task_service.get_task_service",
            return_value=mock_task_service,
        ):
            with patch(
                "app.services.database.get_database_service",
                return_value=mock_db_service,
            ):
                # Step 1 – synchronous part of /contribute
                contribution = await service.add_knowledge_contribution(
                    title="Full Flow Test",
                    content="Full flow content",
                    meta_tags=["full", "flow"],
                )

                # At this point nothing should be in the vectordb
                assert vector_store.total_added == 0, (
                    "add_knowledge_contribution must not write to vectordb"
                )

                # Step 2 – background task
                await service.process_knowledge_contribution_background(
                    task_id=contribution["task_id"],
                    knowledge_hash_id=contribution["id"],
                    metadata={
                        "title": "Full Flow Test",
                        "content": "Full flow content",
                        "source": None,
                        "author_name": None,
                        "additional_references": None,
                        "uploaded_file_path": None,
                        "is_public": False,
                        "meta_tags": ["full", "flow"],
                    },
                )

        # Exactly one vectordb write (from background task)
        assert vector_store.total_added >= 1
        # MongoDB was inserted exactly once (not a second time in the background task)
        mock_db_service.insert_knowledge_document.assert_called_once()


# ---------------------------------------------------------------------------
# APPROACH 1 – sync_mongodb_to_vectordb
# ---------------------------------------------------------------------------

class AsyncMockCursor:
    """Async cursor that returns a fixed list of documents."""

    def __init__(self, documents: list):
        self._documents = documents

    async def to_list(self, length=None):
        return list(self._documents)


class AsyncMockCollection:
    """Async MongoDB collection mock used in sync tests."""

    def __init__(self, documents: list):
        self._documents = documents

    def find(self, query):
        return AsyncMockCursor(self._documents)


class MockDatabaseServiceForSync:
    """Stand-in for DatabaseService used inside sync_mongodb_to_vectordb."""

    def __init__(self, documents: list):
        self._collection = AsyncMockCollection(documents)
        self.insert_calls: list = []

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    def get_knowledgebase_collection(self):
        return self._collection

    async def insert_knowledge_document(self, doc):
        """If this is called the test will fail – sync should never insert."""
        self.insert_calls.append(doc)
        return "unexpected-insert"


def _make_dp_for_sync(vector_store=None):
    """
    Build a DocumentProcessor instance with __init__ bypassed and the
    minimum attributes needed by sync_mongodb_to_vectordb set manually.
    """
    with patch("app.services.document_processor.settings"):
        from app.services.document_processor import DocumentProcessor

    dp = object.__new__(DocumentProcessor)
    dp.text_splitter = MockTextSplitter()
    dp._vector_store = vector_store or MockVectorStore()
    dp.embeddings_available = True
    dp.embeddings = MagicMock()
    dp.chroma_client = MagicMock()
    return dp


class TestSyncMongoDBToVectorDB:
    """
    sync_mongodb_to_vectordb should:
      ✓ Add documents to vectordb (with text splitting)
      ✓ Preserve the original hash_id from MongoDB
      ✓ Handle additional_references stored as a list (not crash)
      ✗ NOT insert new documents into MongoDB
      ✗ NOT create new hash_ids
    """

    def _build_mongo_doc(self, **overrides):
        base = {
            "_id": "mongo-" + str(uuid.uuid4()),
            "hash_id": "orig-hash-" + str(uuid.uuid4()),
            "title": "Sync Test Document",
            "content": "Agricultural knowledge content",
            "meta_tags": ["wheat", "irrigation"],
            "source": "field-research",
            "author_name": "Researcher",
            "additional_references": ["https://ref1.com", "https://ref2.com"],
            "is_public": True,
            "submission_timestamp": datetime.now().isoformat(),
            "synced": True,
            "entry_type": "user_contribution",
        }
        base.update(overrides)
        return base

    async def test_does_not_insert_into_mongodb(self):
        """Syncing must write to vectordb but must never insert a new DB document."""
        mongo_doc = self._build_mongo_doc()
        mock_db = MockDatabaseServiceForSync([mongo_doc])
        vector_store = MockVectorStore()
        dp = _make_dp_for_sync(vector_store)

        with patch("app.services.document_processor.DatabaseService", return_value=mock_db):
            with patch.object(dp, "get_vector_store", return_value=vector_store):
                result = await dp.sync_mongodb_to_vectordb(knowledge_base_service=None)

        assert mock_db.insert_calls == [], (
            f"sync must not insert into MongoDB, but {len(mock_db.insert_calls)} insert(s) occurred"
        )
        assert result["processed_count"] == 1
        assert result["error_count"] == 0

    async def test_preserves_original_hash_id(self):
        """Vectordb chunks must carry the hash_id from MongoDB, not a freshly generated one."""
        original_hash = "fixed-hash-abc-123"
        mongo_doc = self._build_mongo_doc(hash_id=original_hash)
        vector_store = MockVectorStore()
        mock_db = MockDatabaseServiceForSync([mongo_doc])
        dp = _make_dp_for_sync(vector_store)

        with patch("app.services.document_processor.DatabaseService", return_value=mock_db):
            with patch.object(dp, "get_vector_store", return_value=vector_store):
                await dp.sync_mongodb_to_vectordb(knowledge_base_service=None)

        all_docs = [d for batch in vector_store.added_batches for d in batch]
        assert all_docs, "Expected at least one document in vectordb"
        for doc in all_docs:
            assert doc.metadata["hash_id"] == original_hash, (
                f"Expected hash_id={original_hash!r}, got {doc.metadata['hash_id']!r}"
            )

    async def test_writes_to_vectordb(self):
        """Sync must add chunks to the vectordb."""
        mongo_doc = self._build_mongo_doc()
        vector_store = MockVectorStore()
        mock_db = MockDatabaseServiceForSync([mongo_doc])
        dp = _make_dp_for_sync(vector_store)

        with patch("app.services.document_processor.DatabaseService", return_value=mock_db):
            with patch.object(dp, "get_vector_store", return_value=vector_store):
                await dp.sync_mongodb_to_vectordb(knowledge_base_service=None)

        assert vector_store.total_added >= 1, "sync must add at least one chunk to vectordb"

    async def test_handles_additional_references_as_list(self):
        """
        MongoDB stores additional_references as a list.
        The sync must join it to a string before embedding it in metadata
        (Chroma metadata values must be scalar).
        """
        mongo_doc = self._build_mongo_doc(
            additional_references=["https://ref1.com", "https://ref2.com"]
        )
        vector_store = MockVectorStore()
        mock_db = MockDatabaseServiceForSync([mongo_doc])
        dp = _make_dp_for_sync(vector_store)

        with patch("app.services.document_processor.DatabaseService", return_value=mock_db):
            with patch.object(dp, "get_vector_store", return_value=vector_store):
                result = await dp.sync_mongodb_to_vectordb(knowledge_base_service=None)

        assert result["error_count"] == 0, "No errors expected when additional_references is a list"
        all_docs = [d for batch in vector_store.added_batches for d in batch]
        for doc in all_docs:
            ref_val = doc.metadata.get("additional_references", "")
            assert isinstance(ref_val, str), (
                f"additional_references in metadata must be a string, got {type(ref_val)}"
            )

    async def test_handles_additional_references_as_string(self):
        """additional_references stored as a plain string must also work."""
        mongo_doc = self._build_mongo_doc(additional_references="https://ref1.com")
        vector_store = MockVectorStore()
        mock_db = MockDatabaseServiceForSync([mongo_doc])
        dp = _make_dp_for_sync(vector_store)

        with patch("app.services.document_processor.DatabaseService", return_value=mock_db):
            with patch.object(dp, "get_vector_store", return_value=vector_store):
                result = await dp.sync_mongodb_to_vectordb(knowledge_base_service=None)

        assert result["error_count"] == 0

    async def test_skips_documents_with_missing_title_or_content(self):
        """Documents without a title or content must be counted as skipped, not errored."""
        docs = [
            self._build_mongo_doc(title=""),          # missing title
            self._build_mongo_doc(content=""),         # missing content
            self._build_mongo_doc(),                   # valid
        ]
        vector_store = MockVectorStore()
        mock_db = MockDatabaseServiceForSync(docs)
        dp = _make_dp_for_sync(vector_store)

        with patch("app.services.document_processor.DatabaseService", return_value=mock_db):
            with patch.object(dp, "get_vector_store", return_value=vector_store):
                result = await dp.sync_mongodb_to_vectordb(knowledge_base_service=None)

        assert result["skipped_count"] == 2, (
            f"Expected 2 skipped, got {result['skipped_count']}"
        )
        assert result["processed_count"] == 1
        assert result["error_count"] == 0

    async def test_multiple_documents_all_go_to_vectordb_not_mongodb(self):
        """Every document in MongoDB must produce vectordb chunks but zero new DB inserts."""
        docs = [self._build_mongo_doc(title=f"Doc {i}") for i in range(5)]
        mock_db = MockDatabaseServiceForSync(docs)
        vector_store = MockVectorStore()
        dp = _make_dp_for_sync(vector_store)

        with patch("app.services.document_processor.DatabaseService", return_value=mock_db):
            with patch.object(dp, "get_vector_store", return_value=vector_store):
                result = await dp.sync_mongodb_to_vectordb(knowledge_base_service=None)

        assert result["processed_count"] == 5
        assert result["error_count"] == 0
        assert mock_db.insert_calls == [], "No new MongoDB inserts expected during sync"
        assert vector_store.total_added >= 5

    async def test_stats_include_timing(self):
        """Result dict must include start_time, end_time, and total_time_seconds."""
        mongo_doc = self._build_mongo_doc()
        mock_db = MockDatabaseServiceForSync([mongo_doc])
        vector_store = MockVectorStore()
        dp = _make_dp_for_sync(vector_store)

        with patch("app.services.document_processor.DatabaseService", return_value=mock_db):
            with patch.object(dp, "get_vector_store", return_value=vector_store):
                result = await dp.sync_mongodb_to_vectordb(knowledge_base_service=None)

        assert "start_time" in result
        assert "end_time" in result
        assert "total_time_seconds" in result
        assert result["total_time_seconds"] >= 0
