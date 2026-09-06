"""
Example script demonstrating DOCX processing capabilities.

This script shows how to:
1. Process single DOCX files
2. Process multiple DOCX files in a directory
3. Extract tables from DOCX files
4. Convert DOCX to markdown
5. Process both PDF and DOCX files together
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.document_processor import get_document_processor


def example_process_single_docx():
    """Example: Process a single DOCX file."""
    print("\n=== Example 1: Process Single DOCX File ===")
    
    processor = get_document_processor()
    
    # Path to your DOCX file
    docx_path = "docs/sample.docx"
    
    if not os.path.exists(docx_path):
        print(f"File not found: {docx_path}")
        print("Please place a DOCX file in the docs directory")
        return
    
    # Process the DOCX file
    documents = processor.process_docx(docx_path)
    
    print(f"Processed {len(documents)} document chunks from {docx_path}")
    
    # Show first chunk as example
    if documents:
        print("\nFirst chunk preview:")
        print(documents[0].page_content[:200] + "...")
        print(f"\nMetadata: {documents[0].metadata}")


def example_extract_tables_from_docx():
    """Example: Extract tables from a DOCX file."""
    print("\n=== Example 2: Extract Tables from DOCX ===")
    
    processor = get_document_processor()
    
    docx_path = "docs/sample.docx"
    
    if not os.path.exists(docx_path):
        print(f"File not found: {docx_path}")
        return
    
    # Extract tables
    tables = processor.extract_tables_from_docx(docx_path)
    
    print(f"Found {len(tables)} tables in the document")
    
    for table in tables:
        print(f"\nTable {table['table_index']}:")
        print(f"  Rows: {table['rows']}, Columns: {table['columns']}")
        print("\nMarkdown representation:")
        print(table['markdown'][:300] + "..." if len(table['markdown']) > 300 else table['markdown'])


def example_convert_docx_to_markdown():
    """Example: Convert DOCX to markdown format."""
    print("\n=== Example 3: Convert DOCX to Markdown ===")
    
    processor = get_document_processor()
    
    docx_path = "docs/sample.docx"
    output_dir = "processed_docs"
    
    if not os.path.exists(docx_path):
        print(f"File not found: {docx_path}")
        return
    
    # Convert to markdown
    markdown_content = processor.load_docx_as_markdown(docx_path, output_dir)
    
    print(f"Converted DOCX to markdown ({len(markdown_content)} characters)")
    print(f"Markdown saved to: {output_dir}/markdown/")
    
    # Show preview
    print("\nMarkdown preview:")
    print(markdown_content[:500] + "...")


def example_process_all_docx_files():
    """Example: Process all DOCX files in docs directory."""
    print("\n=== Example 4: Process All DOCX Files ===")
    
    processor = get_document_processor()
    
    # Process all DOCX files in the docs directory
    count = processor.process_all_docx()
    
    print(f"Processed {count} document chunks from all DOCX files")
    print("Documents added to vector store for semantic search")


def example_process_mixed_documents():
    """Example: Process both PDF and DOCX files together."""
    print("\n=== Example 5: Process Mixed PDF and DOCX Files ===")
    
    processor = get_document_processor()
    
    input_dir = "docs"
    output_dir = "processed_docs"
    
    # Process all PDF and DOCX files
    stats = processor.batch_process_mixed_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        create_vectors=True
    )
    
    print("\nProcessing Statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  PDF files: {stats['pdf_files']}")
    print(f"  DOCX files: {stats['docx_files']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total pages: {stats['total_pages']}")
    print(f"  Total tables: {stats['total_tables']}")
    print(f"  Processing time: {stats['processing_time']:.2f} seconds")
    
    if stats.get('avg_processing_time_per_file'):
        print(f"  Avg time per file: {stats['avg_processing_time_per_file']:.2f} seconds")


def example_process_all_documents():
    """Example: Process all documents (PDF + DOCX) with summary."""
    print("\n=== Example 6: Process All Documents with Summary ===")
    
    processor = get_document_processor()
    
    # Process all documents
    result = processor.process_all_documents()
    
    print("\nDocument Processing Summary:")
    print(f"  PDF documents: {result['pdf_documents']} chunks")
    print(f"  DOCX documents: {result['docx_documents']} chunks")
    print(f"  Total documents: {result['total_documents']} chunks")
    print("\nAll documents added to vector store for retrieval")


def example_search_documents():
    """Example: Search processed documents."""
    print("\n=== Example 7: Search Processed Documents ===")
    
    processor = get_document_processor()
    
    # Example search queries
    queries = [
        "What are the main requirements?",
        "چگونه می‌توان ثبت‌نام کرد؟",  # Persian: How to register?
        "Show me tables with pricing information"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = processor.search_documents(query, k=3)
        
        print(f"Found {len(results)} relevant documents:")
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Source: {result['metadata'].get('filename', 'unknown')}")
            print(f"    File type: {result['metadata'].get('file_type', 'unknown')}")
            print(f"    Content preview: {result['content'][:150]}...")


def main():
    """Run all examples."""
    print("=".ljust(60, "="))
    print("DOCX Processing Examples")
    print("=".ljust(60, "="))
    
    examples = [
        ("Process Single DOCX File", example_process_single_docx),
        ("Extract Tables from DOCX", example_extract_tables_from_docx),
        ("Convert DOCX to Markdown", example_convert_docx_to_markdown),
        ("Process All DOCX Files", example_process_all_docx_files),
        ("Process Mixed PDF and DOCX Files", example_process_mixed_documents),
        ("Process All Documents with Summary", example_process_all_documents),
        ("Search Processed Documents", example_search_documents),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning Example 6 (Process All Documents with Summary)...")
    print("To run other examples, modify the main() function or run them individually.")
    
    try:
        example_process_all_documents()
    except Exception as e:
        print(f"\nError running example: {e}")
        print("\nMake sure you have:")
        print("  1. Installed required dependencies: pip install python-docx pymupdf")
        print("  2. Placed some DOCX files in the 'docs' directory")
        print("  3. Configured your OpenAI API key for embeddings")


if __name__ == "__main__":
    main()

