# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-11-10

### Added - Word Document (DOCX) Processing

#### Core Features
- **DOCX Processing Support**: Complete implementation for processing Word documents (.docx files)
- **API DOCX Upload Support**: Knowledge contribution API now accepts DOCX files
  - `extract_tables_from_docx()`: Extract tables from Word documents
  - `load_docx_as_markdown()`: Convert DOCX files to Markdown format
  - `process_docx()`: Process single DOCX files and create embeddings
  - `process_all_docx()`: Batch process all DOCX files in a directory
  - `process_all_documents()`: Unified processing for both PDF and DOCX files
  - `batch_process_mixed_directory()`: Advanced batch processing for mixed file types

#### Documentation
- **DOCX Processing Guide**: Comprehensive documentation (`docs/DOCX_PROCESSING.md`)
  - Feature descriptions and usage examples
  - Persian/RTL text support details
  - API integration guide
  - Configuration options
  - Troubleshooting section
  - Performance considerations
  - Best practices

- **Quick Start Guide**: Easy-to-follow guide (`docs/QUICK_START_DOCX.md`)
  - 5-minute setup instructions
  - Common task examples
  - Command-line reference
  - Troubleshooting quick fixes

- **Feature Summary**: Complete implementation overview (`DOCX_FEATURE_SUMMARY.md`)
  - All changes documented
  - Technical details
  - Usage examples
  - Known limitations
  - Future enhancements

- **Updated README**: Added document processing section to main README
  - Feature highlights
  - Quick start examples
  - Installation instructions

#### Examples and Tools
- **Example Scripts**: Comprehensive examples (`examples/docx_processing_example.py`)
  - 7 different usage scenarios
  - Single file processing
  - Batch processing
  - Table extraction
  - Document search
  - Ready-to-run demonstrations

- **API Examples**: API integration examples (`examples/api_docx_contribution_example.py`)
  - Python requests examples
  - CURL command examples
  - Multiple file upload examples
  - Mixed file type contributions
  - Error handling examples

- **CLI Utility**: Command-line tool (`scripts/process_documents.py`)
  - Process single files
  - Process directories
  - Extract tables
  - Search documents
  - Convert to markdown
  - User-friendly output with progress indicators

#### Testing
- **Unit Tests**: Comprehensive test suite (`tests/test_docx_processing.py`)
  - Core functionality tests
  - Integration tests
  - Error handling tests
  - Metadata verification
  - Graceful degradation tests

#### Dependencies
- **New Requirements**: Added to `requirements.txt`
  - `python-docx>=0.8.11`: Word document processing
  - `pymupdf>=1.23.0`: Enhanced PDF processing
  - `tabulate>=0.9.0`: Markdown table formatting
  - `arabic-reshaper>=3.0.0`: Persian text reshaping
  - `python-bidi>=0.4.2`: Bidirectional text support

### Enhanced

#### Persian/RTL Support
- Enhanced Persian text processing for DOCX files
- Arabic to Persian numeral conversion
- Zero-width character removal
- Proper header detection for RTL text
- Bidirectional text handling

#### Document Processing Pipeline
- Unified pipeline for PDF and DOCX files
- Consistent metadata structure across file types
- Improved error handling and reporting
- Progress tracking for batch operations
- Detailed processing statistics

#### Vector Store Integration
- Seamless integration of DOCX content into vector store
- Consistent chunking strategy across file types
- Enhanced metadata for better search results
- Batch embedding generation for efficiency

### Technical Details

#### Architecture
- Graceful degradation when `python-docx` not installed
- `DOCX_SUPPORT` flag for feature detection
- Singleton pattern for processor instance
- Consistent API across PDF and DOCX processing

#### Performance
- Batch processing in chunks of 100 documents
- Memory-efficient streaming for large files
- Progress tracking for long operations
- Optimized vector store updates

#### Output Structure
```
processed/
├── markdown/              # Human-readable markdown files
├── structured_json/       # Structured data with metadata
├── vector_db/            # Vector embeddings (ChromaDB)
└── batch_processing_stats.json  # Processing statistics
```

#### API Integration
- **Knowledge Contribution Endpoint**: Enhanced `/api/knowledge/contribute`
  - Now accepts DOCX files in addition to PDF and Excel
  - Automatic DOCX processing and vectorization
  - Rich metadata attachment to document chunks
  - Batch processing for file documents
  - Updated validation to accept `.docx` extension
  - Enhanced API documentation with file format details

- **Service Layer**: Updated `add_knowledge_contribution` method
  - DOCX file processing integration
  - Consistent metadata structure across file types
  - Entry type: "user_contribution_docx"
  - Logging for DOCX processing events

### File Changes Summary

#### Modified Files
- `app/services/document_processor.py` - Added 442 lines of DOCX processing code
- `app/services/knowledge_base.py` - Added DOCX support to contribution method (22 lines)
- `app/api/routes/knowledge_base.py` - Updated file validation and documentation (8 lines)
- `requirements.txt` - Added 6 new dependencies
- `README.md` - Added document processing section (79 lines)

#### New Files
- `docs/DOCX_PROCESSING.md` - Complete documentation
- `docs/QUICK_START_DOCX.md` - Quick start guide
- `docs/API_DOCX_CONTRIBUTION.md` - API integration guide
- `examples/docx_processing_example.py` - Usage examples
- `examples/api_docx_contribution_example.py` - API usage examples
- `scripts/process_documents.py` - CLI utility
- `tests/test_docx_processing.py` - Test suite
- `DOCX_FEATURE_SUMMARY.md` - Implementation summary
- `CHANGELOG.md` - This file

### Backward Compatibility

✅ **No Breaking Changes**
- All existing PDF processing functionality preserved
- Existing API endpoints continue to work
- No changes to existing method signatures
- Optional DOCX support with graceful degradation

### Known Limitations

1. Only supports `.docx` format (not older `.doc` format)
2. Images are not extracted (text and tables only)
3. Some advanced Word formatting may be lost in conversion
4. Complex merged table cells handled on best-effort basis
5. Requires `python-docx` library for DOCX support

### Future Enhancements

Planned for future releases:
- [ ] Support for older `.doc` format
- [ ] Image extraction and OCR integration
- [ ] Advanced formatting preservation
- [ ] Document comparison features
- [ ] Streaming processing for very large files
- [ ] Concurrent batch processing
- [ ] Custom chunking strategies per file type

### Migration Guide

No migration needed! The new features are additive:

1. **Install new dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start using DOCX processing**:
   ```python
   processor = get_document_processor()
   documents = processor.process_docx("file.docx")
   ```

3. **Or continue using PDF only**:
   - No changes required to existing code
   - PDF processing works exactly as before

### Contributors

- Development Team

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## [1.0.0] - Previous Release

### Initial Release
- PDF document processing
- Vector store integration
- Persian text support
- Knowledge base system
- Chat API with LangChain
- FastAPI server with OpenAPI documentation
- MongoDB integration
- User authentication
- Excel file processing

---

For older changes, see version control history.

