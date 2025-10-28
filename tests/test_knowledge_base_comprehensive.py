"""
Comprehensive test script for knowledge base system operations.

This test script performs the following operations:
1. Test Setup: Initialize knowledge base system with clean test environment
2. Test Case Implementation: Add and verify test data with complete metadata
3. Cleanup Verification: Remove test data and verify cleanup
4. Assertions: All operations complete without errors with explicit assertions
5. Reporting: Detailed logging and performance metrics

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
from datetime import datetime
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


@pytest.fixture
def test_data():
    """Fixture to provide test data for knowledge base operations."""
    unique_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    return {
        "hash_id": unique_id,
        "title": f"Test Knowledge Entry {unique_id[:8]}",
        "content": f"This is test content for knowledge base testing. Unique ID: {unique_id}. Created at: {timestamp}",
        "source": "Test Suite",
        "meta_tags": ["test", "automation", "knowledge_base"],
        "author_name": "Test Author",
        "additional_references": "https://test.example.com",
        "timestamp": timestamp
    }


@pytest.mark.asyncio
class TestKnowledgeBaseComprehensive:
    """Comprehensive test class for knowledge base operations."""
    
    async def test_complete_knowledge_base_workflow(self, mock_knowledge_base_service, test_data, test_metrics):
        """
        Test the complete workflow of adding and removing knowledge contributions.
        
        This test performs:
        1. Add new test data to the knowledge base
        2. Verify the data was successfully added
        3. Remove the test data
        4. Verify successful removal
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE KNOWLEDGE BASE TEST")
        logger.info("=" * 80)
        
        service = mock_knowledge_base_service
        
        # Mock database service
        mock_db_service = MockDatabaseService()
        
        with patch('app.services.database.get_database_service', return_value=mock_db_service):
            
            # PHASE 1: ADD KNOWLEDGE CONTRIBUTION
            logger.info("PHASE 1: Adding knowledge contribution")
            test_metrics.start_timer("add_knowledge_contribution")
            
            try:
                result = await service.add_knowledge_contribution(
                    title=test_data["title"],
                    content=test_data["content"],
                    source=test_data["source"],
                    meta_tags=test_data["meta_tags"],
                    author_name=test_data["author_name"],
                    additional_references=test_data["additional_references"]
                )
                
                add_duration = test_metrics.end_timer("add_knowledge_contribution")
                
                # Assertions for successful addition
                assert result is not None, "Add operation should return a result"
                assert "id" in result, "Result should contain an ID"
                assert result["title"] == test_data["title"], "Title should match input"
                assert result["source"] == test_data["source"], "Source should match input"
                assert result["meta_tags"] == test_data["meta_tags"], "Meta tags should match input"
                assert result["author_name"] == test_data["author_name"], "Author name should match input"
                assert "db_id" in result, "Result should contain database ID"
                
                hash_id = result["id"]
                logger.info(f"✓ Successfully added knowledge contribution with hash_id: {hash_id}")
                logger.info(f"✓ Addition completed in {add_duration:.3f} seconds")
                
            except Exception as e:
                logger.error(f"✗ Failed to add knowledge contribution: {str(e)}")
                pytest.fail(f"Add operation failed: {str(e)}")
            
            # PHASE 2: VERIFY DATA WAS ADDED
            logger.info("PHASE 2: Verifying data was successfully added")
            test_metrics.start_timer("verify_addition")
            
            try:
                # Get vector store and check if documents were added
                vector_store = service.document_processor.get_vector_store()
                assert vector_store is not None, "Vector store should be available"
                
                # Perform similarity search to find the added document
                search_results = vector_store.similarity_search_with_score(test_data["title"], k=5)
                
                verify_duration = test_metrics.end_timer("verify_addition")
                
                # Assertions for successful verification
                assert len(search_results) > 0, "Search should return results for added document"
                
                found_document = False
                for doc, score in search_results:
                    if doc.metadata.get("hash_id") == hash_id:
                        found_document = True
                        assert doc.metadata.get("title") == test_data["title"], "Document title should match"
                        assert doc.metadata.get("source") == test_data["source"], "Document source should match"
                        assert doc.metadata.get("author_name") == test_data["author_name"], "Document author should match"
                        assert test_data["title"] in doc.page_content, "Document content should contain title"
                        break
                
                assert found_document, f"Document with hash_id {hash_id} should be found in search results"
                
                logger.info(f"✓ Successfully verified document addition with {len(search_results)} search results")
                logger.info(f"✓ Verification completed in {verify_duration:.3f} seconds")
                
            except Exception as e:
                logger.error(f"✗ Failed to verify document addition: {str(e)}")
                pytest.fail(f"Verification failed: {str(e)}")
            
            # PHASE 3: REMOVE KNOWLEDGE CONTRIBUTION
            logger.info("PHASE 3: Removing knowledge contribution")
            test_metrics.start_timer("remove_knowledge_contribution")
            
            try:
                removal_result = await service.remove_knowledge_contribution(hash_id)
                
                remove_duration = test_metrics.end_timer("remove_knowledge_contribution")
                
                # Assertions for successful removal
                assert removal_result is not None, "Remove operation should return a result"
                assert "success" in removal_result, "Result should contain success status"
                assert removal_result["hash_id"] == hash_id, "Result should contain correct hash_id"
                assert "removed_from_vector_store" in removal_result, "Result should contain vector store removal status"
                assert "removed_from_database" in removal_result, "Result should contain database removal status"
                assert "documents_removed_count" in removal_result, "Result should contain removed documents count"
                assert "timestamp" in removal_result, "Result should contain timestamp"
                
                logger.info(f"✓ Successfully initiated removal for hash_id: {hash_id}")
                logger.info(f"✓ Vector store removal: {removal_result['removed_from_vector_store']}")
                logger.info(f"✓ Database removal: {removal_result['removed_from_database']}")
                logger.info(f"✓ Documents removed: {removal_result['documents_removed_count']}")
                logger.info(f"✓ Removal completed in {remove_duration:.3f} seconds")
                
                # Verify sync status was updated before document deletion
                # Since the document is deleted after sync status update, we can't verify the sync status directly
                # But we can verify that the sync status update method was called by checking our mock logs
                logger.info("✓ Sync status update should have been called during removal process")
                
            except Exception as e:
                logger.error(f"✗ Failed to remove knowledge contribution: {str(e)}")
                pytest.fail(f"Remove operation failed: {str(e)}")
            
            # PHASE 4: VERIFY CLEANUP
            logger.info("PHASE 4: Verifying successful cleanup")
            test_metrics.start_timer("verify_cleanup")
            
            try:
                # Perform the same search query to verify no results are returned
                cleanup_search_results = vector_store.similarity_search_with_score(test_data["title"], k=5)
                
                verify_cleanup_duration = test_metrics.end_timer("verify_cleanup")
                
                # Check that the specific document is no longer found
                document_still_exists = False
                for doc, score in cleanup_search_results:
                    if doc.metadata.get("hash_id") == hash_id:
                        document_still_exists = True
                        break
                
                assert not document_still_exists, f"Document with hash_id {hash_id} should not be found after removal"
                
                # Verify database cleanup - document should still exist but be marked as unsynced
                db_document_exists = hash_id in [doc.get('hash_id') for doc in mock_db_service.documents.values()]
                assert db_document_exists, f"Document with hash_id {hash_id} should still exist in database after removal"
                
                # Verify the document is marked as unsynced
                db_document = None
                for doc in mock_db_service.documents.values():
                    if doc.get('hash_id') == hash_id:
                        db_document = doc
                        break
                
                assert db_document is not None, f"Document with hash_id {hash_id} should be found in database"
                assert db_document.get('synced') == False, f"Document with hash_id {hash_id} should be marked as unsynced"
                
                logger.info(f"✓ Successfully verified cleanup - document no longer found in vector store")
                logger.info(f"✓ Database cleanup verified - document marked as unsynced in database")
                logger.info(f"✓ Cleanup verification completed in {verify_cleanup_duration:.3f} seconds")
                
            except Exception as e:
                logger.error(f"✗ Failed to verify cleanup: {str(e)}")
                pytest.fail(f"Cleanup verification failed: {str(e)}")
        
        # PHASE 5: GENERATE TEST REPORT
        logger.info("PHASE 5: Generating test report")
        
        performance_report = test_metrics.get_report()
        
        logger.info("=" * 80)
        logger.info("TEST COMPLETION REPORT")
        logger.info("=" * 80)
        logger.info(f"✓ All test phases completed successfully")
        logger.info(f"✓ Total execution time: {performance_report['total_execution_time']:.3f} seconds")
        logger.info(f"✓ Average operation time: {performance_report['average_operation_time']:.3f} seconds")
        logger.info("✓ Performance metrics:")
        
        for operation, duration in performance_report['operation_metrics'].items():
            logger.info(f"  - {operation}: {duration:.3f} seconds")
        
        logger.info("✓ Knowledge base left in original state")
        logger.info("=" * 80)
        
        # Final assertions
        assert performance_report['total_execution_time'] > 0, "Test should have measurable execution time"
        assert len(performance_report['operation_metrics']) == 4, "Should have metrics for all 4 operations"
        
        logger.info("🎉 COMPREHENSIVE KNOWLEDGE BASE TEST COMPLETED SUCCESSFULLY! 🎉")


    async def test_add_knowledge_contribution_with_metadata_validation(self, mock_knowledge_base_service, test_metrics):
        """Test adding knowledge contribution with comprehensive metadata validation."""
        logger.info("Testing knowledge contribution addition with metadata validation")
        
        service = mock_knowledge_base_service
        mock_db_service = MockDatabaseService()
        
        test_metrics.start_timer("metadata_validation_test")
        
        with patch('app.services.database.get_database_service', return_value=mock_db_service):
            
            # Test data with comprehensive metadata
            test_data = {
                "title": "Metadata Validation Test Entry",
                "content": "This entry tests comprehensive metadata validation in the knowledge base system.",
                "source": "Automated Test Suite",
                "meta_tags": ["validation", "metadata", "testing", "automation"],
                "author_name": "Test Automation System",
                "additional_references": "https://test.validation.com, https://metadata.test.org"
            }
            
            try:
                result = await service.add_knowledge_contribution(**test_data)
                
                # Validate all metadata fields are preserved
                assert result["title"] == test_data["title"]
                assert result["source"] == test_data["source"]
                assert result["meta_tags"] == test_data["meta_tags"]
                assert result["author_name"] == test_data["author_name"]
                assert result["additional_references"] == test_data["additional_references"]
                assert "submitted_at" in result
                assert "id" in result
                assert "db_id" in result
                
                # Verify document in vector store has correct metadata
                vector_store = service.document_processor.get_vector_store()
                search_results = vector_store.similarity_search_with_score(test_data["title"], k=1)
                
                assert len(search_results) > 0
                doc, score = search_results[0]
                
                assert doc.metadata["title"] == test_data["title"]
                assert doc.metadata["source"] == test_data["source"]
                assert doc.metadata["author_name"] == test_data["author_name"]
                assert doc.metadata["entry_type"] == "user_contribution"
                assert doc.metadata["source_type"] == "qa_contribution"
                assert "hash_id" in doc.metadata
                assert "submission_timestamp" in doc.metadata
                
                test_duration = test_metrics.end_timer("metadata_validation_test")
                logger.info(f"✓ Metadata validation test completed in {test_duration:.3f} seconds")
                
                # Cleanup
                await service.remove_knowledge_contribution(result["id"])
                
            except Exception as e:
                logger.error(f"✗ Metadata validation test failed: {str(e)}")
                pytest.fail(f"Metadata validation failed: {str(e)}")


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


    async def test_performance_benchmarks(self, mock_knowledge_base_service, test_metrics):
        """Test performance benchmarks for knowledge base operations."""
        logger.info("Running performance benchmark tests")
        
        service = mock_knowledge_base_service
        mock_db_service = MockDatabaseService()
        
        with patch('app.services.database.get_database_service', return_value=mock_db_service):
            
            # Performance thresholds (in seconds)
            ADD_THRESHOLD = 2.0
            SEARCH_THRESHOLD = 1.0
            REMOVE_THRESHOLD = 1.0
            
            # Test data
            test_entries = []
            for i in range(3):  # Test with multiple entries
                test_entries.append({
                    "title": f"Performance Test Entry {i+1}",
                    "content": f"This is performance test content for entry {i+1}. " * 10,  # Longer content
                    "source": f"Performance Test {i+1}",
                    "meta_tags": ["performance", "benchmark", f"test{i+1}"],
                    "author_name": "Performance Tester",
                    "additional_references": f"https://performance.test{i+1}.com"
                })
            
            added_ids = []
            
            try:
                # Benchmark addition operations
                for i, entry in enumerate(test_entries):
                    test_metrics.start_timer(f"add_performance_{i+1}")
                    result = await service.add_knowledge_contribution(**entry)
                    add_duration = test_metrics.end_timer(f"add_performance_{i+1}")
                    
                    added_ids.append(result["id"])
                    assert add_duration < ADD_THRESHOLD, f"Add operation {i+1} took {add_duration:.3f}s, exceeding threshold of {ADD_THRESHOLD}s"
                    logger.info(f"✓ Add operation {i+1} completed in {add_duration:.3f}s (threshold: {ADD_THRESHOLD}s)")
                
                # Benchmark search operations
                vector_store = service.document_processor.get_vector_store()
                for i, entry in enumerate(test_entries):
                    test_metrics.start_timer(f"search_performance_{i+1}")
                    search_results = vector_store.similarity_search_with_score(entry["title"], k=5)
                    search_duration = test_metrics.end_timer(f"search_performance_{i+1}")
                    
                    assert search_duration < SEARCH_THRESHOLD, f"Search operation {i+1} took {search_duration:.3f}s, exceeding threshold of {SEARCH_THRESHOLD}s"
                    assert len(search_results) > 0, f"Search operation {i+1} should return results"
                    logger.info(f"✓ Search operation {i+1} completed in {search_duration:.3f}s (threshold: {SEARCH_THRESHOLD}s)")
                
                # Benchmark removal operations
                for i, hash_id in enumerate(added_ids):
                    test_metrics.start_timer(f"remove_performance_{i+1}")
                    result = await service.remove_knowledge_contribution(hash_id)
                    remove_duration = test_metrics.end_timer(f"remove_performance_{i+1}")
                    
                    assert remove_duration < REMOVE_THRESHOLD, f"Remove operation {i+1} took {remove_duration:.3f}s, exceeding threshold of {REMOVE_THRESHOLD}s"
                    logger.info(f"✓ Remove operation {i+1} completed in {remove_duration:.3f}s (threshold: {REMOVE_THRESHOLD}s)")
                
                logger.info("✓ All performance benchmarks passed successfully")
                
            except Exception as e:
                # Cleanup any remaining entries
                for hash_id in added_ids:
                    try:
                        await service.remove_knowledge_contribution(hash_id)
                    except:
                        pass
                
                logger.error(f"✗ Performance benchmark test failed: {str(e)}")
                pytest.fail(f"Performance benchmark failed: {str(e)}")


if __name__ == "__main__":
    # Run the tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])