"""
Unit tests for DOCX processing functionality.

Tests the document processor's ability to handle Word documents.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.services.document_processor import (
    DocumentProcessor,
    get_document_processor,
    DOCX_SUPPORT
)


class TestDocxProcessing:
    """Test DOCX processing capabilities."""
    
    @pytest.fixture
    def processor(self):
        """Create a document processor instance."""
        return DocumentProcessor()
    
    def test_docx_support_available(self):
        """Test that DOCX support flag is set correctly."""
        # DOCX support should be available if python-docx is installed
        assert isinstance(DOCX_SUPPORT, bool)
    
    @pytest.mark.skipif(not DOCX_SUPPORT, reason="python-docx not installed")
    def test_extract_tables_from_docx_no_file(self, processor):
        """Test table extraction with non-existent file."""
        result = processor.extract_tables_from_docx("nonexistent.docx")
        assert isinstance(result, list)
    
    @pytest.mark.skipif(not DOCX_SUPPORT, reason="python-docx not installed")
    def test_process_docx_no_file(self, processor):
        """Test processing non-existent DOCX file."""
        result = processor.process_docx("nonexistent.docx")
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_process_all_docx_empty_dir(self, processor):
        """Test processing DOCX files with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor.docs_dir = temp_dir
            count = processor.process_all_docx()
            assert count == 0
    
    def test_clean_persian_text(self, processor):
        """Test Persian text cleaning functionality."""
        # Test Arabic to Persian numeral conversion
        text = "١٢٣٤٥٦٧٨٩٠"
        cleaned = processor.clean_persian_text(text)
        assert "۱۲۳۴۵۶۷۸۹۰" == cleaned
        
        # Test whitespace normalization
        text = "text  with   spaces"
        cleaned = processor.clean_persian_text(text)
        assert "text with spaces" == cleaned
        
        # Test empty string
        assert processor.clean_persian_text("") == ""
        assert processor.clean_persian_text(None) == ""
    
    def test_detect_rtl_headers(self, processor):
        """Test RTL header detection."""
        # Test colon ending (header pattern)
        text = "عنوان مهم:"
        result = processor.detect_rtl_headers(text)
        assert result.startswith("##")
        
        # Test numbered section
        text = "۱. عنوان اول"
        result = processor.detect_rtl_headers(text)
        assert result.startswith("###")
    
    @pytest.mark.skipif(not DOCX_SUPPORT, reason="python-docx not installed")
    @patch('app.services.document_processor.DocxDocument')
    def test_load_docx_as_markdown_mocked(self, mock_docx, processor):
        """Test DOCX to markdown conversion with mocked document."""
        # Create mock document
        mock_doc = MagicMock()
        mock_doc.element.body = []
        mock_doc.paragraphs = []
        mock_docx.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            result = processor.load_docx_as_markdown(tmp_path)
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_process_all_documents(self, processor):
        """Test processing all documents (PDF + DOCX)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor.docs_dir = temp_dir
            
            result = processor.process_all_documents()
            
            assert isinstance(result, dict)
            assert 'pdf_documents' in result
            assert 'docx_documents' in result
            assert 'total_documents' in result
            assert result['total_documents'] == result['pdf_documents'] + result['docx_documents']
    
    @pytest.mark.skipif(not DOCX_SUPPORT, reason="python-docx not installed")
    def test_batch_process_mixed_directory_empty(self, processor):
        """Test batch processing with empty directory."""
        with tempfile.TemporaryDirectory() as input_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                result = processor.batch_process_mixed_directory(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    create_vectors=False
                )
                
                assert isinstance(result, dict)
                assert result['total_files'] == 0
                assert result['successful'] == 0
                assert result['failed'] == 0
    
    def test_singleton_instance(self):
        """Test that get_document_processor returns singleton."""
        processor1 = get_document_processor()
        processor2 = get_document_processor()
        assert processor1 is processor2
    
    def test_persian_config(self, processor):
        """Test Persian configuration is set correctly."""
        assert 'chunk_size' in processor.persian_config
        assert 'chunk_overlap' in processor.persian_config
        assert 'persian_separators' in processor.persian_config
        assert 'table_extraction' in processor.persian_config
        
        # Verify Persian separators include expected characters
        separators = processor.persian_config['persian_separators']
        assert '،' in separators  # Persian comma
        assert '؛' in separators  # Persian semicolon


class TestDocxIntegration:
    """Integration tests for DOCX processing."""
    
    @pytest.mark.skipif(not DOCX_SUPPORT, reason="python-docx not installed")
    def test_full_docx_workflow(self):
        """Test complete DOCX processing workflow."""
        processor = get_document_processor()
        
        # This test requires a sample DOCX file
        # In a real environment, you would create a test DOCX file
        # For now, we just verify the methods exist and are callable
        
        assert callable(processor.extract_tables_from_docx)
        assert callable(processor.load_docx_as_markdown)
        assert callable(processor.process_docx)
        assert callable(processor.process_all_docx)
        assert callable(processor.batch_process_mixed_directory)
    
    def test_error_handling_without_docx_support(self):
        """Test graceful degradation when python-docx not available."""
        processor = get_document_processor()
        
        # Even without DOCX support, methods should exist and handle errors
        if not DOCX_SUPPORT:
            result = processor.process_all_docx()
            assert result == 0


class TestDocxMetadata:
    """Test metadata handling for DOCX files."""
    
    @pytest.mark.skipif(not DOCX_SUPPORT, reason="python-docx not installed")
    @patch('app.services.document_processor.DocxDocument')
    def test_docx_metadata_structure(self, mock_docx):
        """Test that DOCX processing creates correct metadata."""
        processor = get_document_processor()
        
        # Mock document with simple content
        mock_doc = MagicMock()
        mock_doc.element.body = []
        mock_doc.paragraphs = [MagicMock()]
        mock_doc.paragraphs[0].text = "Test content"
        mock_doc.paragraphs[0].style.name = "Normal"
        mock_doc.paragraphs[0]._element = MagicMock()
        mock_docx.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            documents = processor.process_docx(tmp_path)
            
            if documents:
                # Check metadata structure
                metadata = documents[0].metadata
                assert 'source' in metadata
                assert 'file_path' in metadata
                assert 'chunk_index' in metadata
                assert 'file_type' in metadata
                assert metadata['file_type'] == 'docx'
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "skipif: skip test if condition is true"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

