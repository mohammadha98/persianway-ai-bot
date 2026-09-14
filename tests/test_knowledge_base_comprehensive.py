"""
Integration tests for knowledge base removal handling.

The add/verify/performance scenarios that used to live here tested an expired
contract: they assumed ``add_knowledge_contribution`` wrote to the vectordb
synchronously, whereas that write now happens exclusively in
``process_knowledge_contribution_background``. Those scenarios were removed and
are superseded by ``tests/test_knowledge_contribution_flows.py``.

What remains here:

1. Test Setup: Initialize the knowledge base service with a clean test environment
2. Removal Handling: ``remove_knowledge_contribution`` tolerates an unknown hash_id
3. Reporting: Detailed logging and performance metrics

Author: AI Assistant
Created: 2024
"""

import pytest
import asyncio
import logging
import time
import uuid
import os
import sys
from typing import Dict, Any, List
from unittest.mock import patch, AsyncMock, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging for test reporting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_knowledge_base_comprehensive.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestMetrics:
    """Class to track performance metrics during testing."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        logger.info(f"Starting operation: {operation}")
    
    def end_timer(self, operation: str):
        """End timing an operation and record the duration."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation] = duration
            logger.info(f"Completed operation: {operation} in {duration:.3f} seconds")
            del self.start_times[operation]
            return duration
        return 0
    
    def get_report(self) -> Dict[str, Any]:
        """Generate a performance report."""
        total_time = sum(self.metrics.values())
        return {
            "total_execution_time": total_time,
            "operation_metrics": self.metrics,
            "average_operation_time": total_time / len(self.metrics) if self.metrics else 0
        }


class MockVectorStore:
    """Mock vector store for testing."""
    
    def __init__(self):
        self.documents = []
        self.collection = MockCollection()
        self._collection = self.collection  # For compatibility
        self.persisted = False
    
    def add_documents(self, documents):
        """Add documents to the mock vector store."""
        self.documents.extend(documents)
        # Add to collection as well
        for doc in documents:
            self.collection.add_document(doc)
        return [str(uuid.uuid4()) for _ in documents]
    
    def similarity_search_with_score(self, query: str, k: int = 4):
        """Mock similarity search with scores."""
        # Return matching documents based on content similarity
        # Only search through documents that are still in the collection
        results = []
        for doc in self.collection.documents:
            if query.lower() in doc.page_content.lower() or query.lower() in doc.metadata.get('title', '').lower():
                results.append((doc, 0.1))  # Low score indicates high similarity
        return results[:k]
    
    def persist(self):
        """Mock persist operation."""
        self.persisted = True
        logger.info("Mock vector store persisted")
    
    def delete(self, ids=None):
        """Mock delete method."""
        if ids:
            initial_count = len(self.documents)
            self.documents = [doc for doc in self.documents if doc.metadata.get('hash_id') not in ids]
            # Also remove from collection
            self.collection.delete(ids=ids)
            removed_count = initial_count - len(self.documents)
            logger.info(f"Removed {removed_count} documents from mock vector store")
            return removed_count
        return 0


class MockCollection:
    """Mock ChromaDB collection for testing."""
    
    def __init__(self):
        self.documents = []
        self.ids = []
        self.metadatas = []
    
    def add_document(self, document):
        """Add a document to the collection."""
        doc_id = str(uuid.uuid4())
        self.documents.append(document)
        self.ids.append(doc_id)
        self.metadatas.append(document.metadata)
        return doc_id
    
    def get(self, where=None, include=None):
        """Mock get method that filters by metadata."""
        if where:
            # Filter documents by metadata
            filtered_ids = []
            filtered_metadatas = []
            
            for i, metadata in enumerate(self.metadatas):
                match = True
                for key, value in where.items():
                    if metadata.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_ids.append(self.ids[i])
                    if include and 'metadatas' in include:
                        filtered_metadatas.append(metadata)
            
            result = {'ids': filtered_ids}
            if include and 'metadatas' in include:
                result['metadatas'] = filtered_metadatas
            
            return result
        else:
            result = {'ids': self.ids}
            if include and 'metadatas' in include:
                result['metadatas'] = self.metadatas
            return result
    
    def delete(self, where=None, ids=None):
        """Mock delete method."""
        removed_count = 0
        
        if where:
            # Delete by metadata
            indices_to_remove = []
            for i, metadata in enumerate(self.metadatas):
                match = True
                for key, value in where.items():
                    if metadata.get(key) != value:
                        match = False
                        break
                if match:
                    indices_to_remove.append(i)
            
            # Remove in reverse order to maintain indices
            for i in reversed(indices_to_remove):
                self.documents.pop(i)
                self.ids.pop(i)
                self.metadatas.pop(i)
                removed_count += 1
                
        elif ids:
            # Delete by IDs
            indices_to_remove = []
            for i, doc_id in enumerate(self.ids):
                if doc_id in ids:
                    indices_to_remove.append(i)
            
            # Remove in reverse order to maintain indices
            for i in reversed(indices_to_remove):
                self.documents.pop(i)
                self.ids.pop(i)
                self.metadatas.pop(i)
                removed_count += 1
        
        logger.info(f"MockCollection deleted {removed_count} documents")
        return removed_count


class MockDocument:
    """Mock Langchain Document."""
    
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata


class MockTextSplitter:
    """Mock text splitter."""
    
    def create_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """Create mock documents."""
        documents = []
        for text, metadata in zip(texts, metadatas):
            documents.append(MockDocument(text, metadata))
        return documents


class MockDocumentProcessor:
    """Mock document processor."""
    
    def __init__(self):
        self.vector_store = MockVectorStore()
        self.text_splitter = MockTextSplitter()
    
    def get_vector_store(self):
        """Return the mock vector store."""
        return self.vector_store


class MockDatabaseService:
    """Mock database service."""
    
    def __init__(self):
        self.documents = {}
        self.next_id = 1
    
    async def insert_knowledge_document(self, document: Dict[str, Any]) -> int:
        """Mock insert operation."""
        doc_id = self.next_id
        self.next_id += 1
        self.documents[doc_id] = document
        logger.info(f"Inserted document with ID: {doc_id}")
        return doc_id
    
    def get_knowledgebase_collection(self):
        """Return a mock collection object."""
        return MockMongoCollection(self.documents)
    
    async def delete_knowledge_document_by_hash_id(self, hash_id: str) -> bool:
        """Mock delete operation by hash_id."""
        for doc_id, doc in list(self.documents.items()):
            if doc.get('hash_id') == hash_id:
                del self.documents[doc_id]
                logger.info(f"Deleted document with hash_id: {hash_id}")
                return True
        return False
    
    async def update_knowledge_document_sync_status(self, hash_id: str, synced: bool) -> bool:
        """Mock update sync status operation."""
        # First, ensure all documents have a synced field with default True
        for doc in self.documents.values():
            if 'synced' not in doc:
                doc['synced'] = True
        
        # Update the specific document's sync status
        for doc in self.documents.values():
            if doc.get('hash_id') == hash_id:
                doc['synced'] = synced
                logger.info(f"Updated sync status for hash_id {hash_id} to {synced}")
                return True
        
        logger.warning(f"Document with hash_id {hash_id} not found for sync status update")
        return False


class MockMongoCollection:
    """Mock MongoDB collection."""
    
    def __init__(self, documents):
        self.documents = documents
    
    async def delete_one(self, filter_dict):
        """Mock delete_one operation."""
        hash_id = filter_dict.get('hash_id')
        for doc_id, doc in list(self.documents.items()):
            if doc.get('hash_id') == hash_id:
                del self.documents[doc_id]
                logger.info(f"MockMongoCollection deleted document with hash_id: {hash_id}")
                return MockDeleteResult(1)  # deleted_count = 1
        return MockDeleteResult(0)  # deleted_count = 0


class MockDeleteResult:
    """Mock delete result."""
    
    def __init__(self, deleted_count):
        self.deleted_count = deleted_count


@pytest.fixture
def test_metrics():
    """Fixture to provide test metrics tracking."""
    return TestMetrics()


@pytest.fixture
def mock_knowledge_base_service():
    """Fixture to provide a mocked knowledge base service."""
    # Mock the problematic imports
    with patch('app.services.chat_service.get_llm'):
        with patch('app.services.document_processor.get_document_processor') as mock_doc_proc:
            with patch('app.services.excel_processor.get_excel_qa_processor'):
                with patch('app.services.config_service.ConfigService'):
                    # Import after mocking
                    from app.services.knowledge_base import KnowledgeBaseService
                    
                    # Create service instance
                    service = KnowledgeBaseService()
                    
                    # Replace with mock processors
                    service.document_processor = MockDocumentProcessor()
                    
                    # Mock the config service
                    mock_config = AsyncMock()
                    mock_config.get_rag_settings.return_value = MagicMock(
                        top_k_results=5,
                        qa_match_threshold=0.8,
                        knowledge_base_confidence_threshold=0.7,
                        human_referral_message="متأسفانه، پاسخ مناسبی یافت نشد."
                    )
                    service.config_service = mock_config
                    
                    return service


@pytest.mark.asyncio
class TestKnowledgeBaseComprehensive:
    """Comprehensive test class for knowledge base operations."""
    
    async def test_remove_nonexistent_contribution(self, mock_knowledge_base_service, test_metrics):
        """Test removing a non-existent knowledge contribution."""
        logger.info("Testing removal of non-existent knowledge contribution")
        
        service = mock_knowledge_base_service
        mock_db_service = MockDatabaseService()
        
        test_metrics.start_timer("nonexistent_removal_test")
        
        with patch('app.services.database.get_database_service', return_value=mock_db_service):
            
            # Generate a random hash_id that doesn't exist
            nonexistent_hash_id = str(uuid.uuid4())
            
            try:
                result = await service.remove_knowledge_contribution(nonexistent_hash_id)
                
                # Should handle gracefully without errors
                assert result is not None
                assert result["hash_id"] == nonexistent_hash_id
                assert "success" in result
                assert "removed_from_vector_store" in result
                assert "removed_from_database" in result
                assert result["documents_removed_count"] == 0
                
                test_duration = test_metrics.end_timer("nonexistent_removal_test")
                logger.info(f"✓ Non-existent removal test completed in {test_duration:.3f} seconds")
                logger.info(f"✓ Gracefully handled removal of non-existent hash_id: {nonexistent_hash_id}")
                
            except Exception as e:
                logger.error(f"✗ Non-existent removal test failed: {str(e)}")
                pytest.fail(f"Non-existent removal test failed: {str(e)}")
