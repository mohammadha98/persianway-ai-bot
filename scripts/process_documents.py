#!/usr/bin/env python3
"""
Command-line utility for processing PDF and DOCX documents.

Usage:
    python scripts/process_documents.py --help
    python scripts/process_documents.py process-single file.docx
    python scripts/process_documents.py process-directory docs/ --output processed/
    python scripts/process_documents.py extract-tables file.docx
    python scripts/process_documents.py search "your query" --top-k 5
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.document_processor import (
    get_document_processor,
    DOCX_SUPPORT,
    PERSIAN_SUPPORT
)


def print_banner():
    """Print application banner."""
    print("=" * 70)
    print("Document Processing Utility")
    print("PDF and DOCX Document Processor with Vector Store Integration")
    print("=" * 70)
    print(f"DOCX Support: {'✓ Available' if DOCX_SUPPORT else '✗ Not Available'}")
    print(f"Persian Support: {'✓ Available' if PERSIAN_SUPPORT else '✗ Not Available'}")
    print("=" * 70)


def process_single_file(file_path: str, output_dir: Optional[str] = None):
    """Process a single PDF or DOCX file."""
    print(f"\n📄 Processing: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return 1
    
    processor = get_document_processor()
    file_ext = Path(file_path).suffix.lower()
    
    try:
        if file_ext == '.pdf':
            documents = processor.process_pdf(file_path)
            print(f"✓ Processed {len(documents)} chunks from PDF")
        elif file_ext == '.docx':
            if not DOCX_SUPPORT:
                print("❌ Error: DOCX support not available. Install python-docx.")
                return 1
            documents = processor.process_docx(file_path)
            print(f"✓ Processed {len(documents)} chunks from DOCX")
        else:
            print(f"❌ Error: Unsupported file type: {file_ext}")
            return 1
        
        # Optionally convert to markdown
        if output_dir:
            print(f"\n📝 Converting to Markdown...")
            if file_ext == '.pdf':
                markdown = processor.load_persian_pdf_as_markdown(file_path, output_dir)
            else:
                markdown = processor.load_docx_as_markdown(file_path, output_dir)
            print(f"✓ Markdown saved to: {output_dir}/markdown/")
            print(f"   Length: {len(markdown)} characters")
        
        # Show preview
        if documents:
            print(f"\n📋 First Chunk Preview:")
            print(f"   {documents[0].page_content[:200]}...")
            print(f"\n📊 Metadata:")
            for key, value in documents[0].metadata.items():
                print(f"   {key}: {value}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return 1


def process_directory(input_dir: str, output_dir: str = "processed", create_vectors: bool = True):
    """Process all PDF and DOCX files in a directory."""
    print(f"\n📁 Processing directory: {input_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Create vectors: {create_vectors}")
    
    if not os.path.exists(input_dir):
        print(f"❌ Error: Directory not found: {input_dir}")
        return 1
    
    processor = get_document_processor()
    
    try:
        print("\n⏳ Processing files...")
        stats = processor.batch_process_mixed_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            create_vectors=create_vectors
        )
        
        print("\n✓ Processing Complete!")
        print(f"\n📊 Statistics:")
        print(f"   Total files: {stats['total_files']}")
        print(f"   PDF files: {stats['pdf_files']}")
        print(f"   DOCX files: {stats['docx_files']}")
        print(f"   Successful: {stats['successful']}")
        print(f"   Failed: {stats['failed']}")
        print(f"   Total pages: {stats['total_pages']}")
        print(f"   Total tables: {stats['total_tables']}")
        print(f"   Processing time: {stats['processing_time']:.2f} seconds")
        
        if stats.get('avg_processing_time_per_file'):
            print(f"   Avg time per file: {stats['avg_processing_time_per_file']:.2f} seconds")
        
        if stats['failed_files']:
            print(f"\n⚠️  Failed Files:")
            for failed in stats['failed_files']:
                print(f"   - {failed['filename']}: {failed['error']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error processing directory: {e}")
        return 1


def extract_tables(file_path: str, output_file: Optional[str] = None):
    """Extract tables from a PDF or DOCX file."""
    print(f"\n📊 Extracting tables from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return 1
    
    processor = get_document_processor()
    file_ext = Path(file_path).suffix.lower()
    
    try:
        if file_ext == '.pdf':
            tables = processor.extract_tables_from_pdf(file_path)
        elif file_ext == '.docx':
            if not DOCX_SUPPORT:
                print("❌ Error: DOCX support not available. Install python-docx.")
                return 1
            tables = processor.extract_tables_from_docx(file_path)
        else:
            print(f"❌ Error: Unsupported file type: {file_ext}")
            return 1
        
        print(f"✓ Found {len(tables)} tables")
        
        for table in tables:
            print(f"\n📋 Table {table['table_index']}:")
            print(f"   Rows: {table['rows']}, Columns: {table['columns']}")
            if file_ext == '.pdf' and 'page_number' in table:
                print(f"   Page: {table['page_number']}")
            print(f"\n{table['markdown']}\n")
        
        # Save to file if specified
        if output_file and tables:
            output_data = {
                'source_file': file_path,
                'total_tables': len(tables),
                'tables': [{
                    'table_index': t['table_index'],
                    'rows': t['rows'],
                    'columns': t['columns'],
                    'page_number': t.get('page_number'),
                    'markdown': t['markdown']
                } for t in tables]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"✓ Tables saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error extracting tables: {e}")
        return 1


def search_documents(query: str, top_k: int = 5):
    """Search processed documents."""
    print(f"\n🔍 Searching for: {query}")
    print(f"   Top results: {top_k}")
    
    processor = get_document_processor()
    
    try:
        results = processor.search_documents(query, k=top_k)
        
        print(f"\n✓ Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"   Source: {result['metadata'].get('filename', 'unknown')}")
            print(f"   Type: {result['metadata'].get('file_type', 'unknown')}")
            if 'chunk_index' in result['metadata']:
                print(f"   Chunk: {result['metadata']['chunk_index']}")
            print(f"   Content: {result['content'][:200]}...")
            print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error searching documents: {e}")
        return 1


def convert_to_markdown(file_path: str, output_dir: str = "markdown_output"):
    """Convert a document to Markdown format."""
    print(f"\n📝 Converting to Markdown: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return 1
    
    processor = get_document_processor()
    file_ext = Path(file_path).suffix.lower()
    
    try:
        if file_ext == '.pdf':
            markdown = processor.load_persian_pdf_as_markdown(file_path, output_dir)
        elif file_ext == '.docx':
            if not DOCX_SUPPORT:
                print("❌ Error: DOCX support not available. Install python-docx.")
                return 1
            markdown = processor.load_docx_as_markdown(file_path, output_dir)
        else:
            print(f"❌ Error: Unsupported file type: {file_ext}")
            return 1
        
        print(f"✓ Conversion complete")
        print(f"   Length: {len(markdown)} characters")
        print(f"   Output: {output_dir}/markdown/")
        
        # Show preview
        print(f"\n📄 Preview:")
        print(markdown[:500])
        if len(markdown) > 500:
            print("...")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error converting to markdown: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process PDF and DOCX documents with vector store integration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Process single file
    single_parser = subparsers.add_parser('process-single', help='Process a single file')
    single_parser.add_argument('file', help='Path to PDF or DOCX file')
    single_parser.add_argument('--output', '-o', help='Output directory for markdown')
    
    # Process directory
    dir_parser = subparsers.add_parser('process-directory', help='Process all files in directory')
    dir_parser.add_argument('input_dir', help='Input directory containing files')
    dir_parser.add_argument('--output', '-o', default='processed', help='Output directory')
    dir_parser.add_argument('--no-vectors', action='store_true', help='Skip vector creation')
    
    # Extract tables
    table_parser = subparsers.add_parser('extract-tables', help='Extract tables from file')
    table_parser.add_argument('file', help='Path to PDF or DOCX file')
    table_parser.add_argument('--output', '-o', help='Output JSON file')
    
    # Search
    search_parser = subparsers.add_parser('search', help='Search processed documents')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results')
    
    # Convert to markdown
    md_parser = subparsers.add_parser('to-markdown', help='Convert to markdown')
    md_parser.add_argument('file', help='Path to PDF or DOCX file')
    md_parser.add_argument('--output', '-o', default='markdown_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Execute command
    if args.command == 'process-single':
        return process_single_file(args.file, args.output)
    
    elif args.command == 'process-directory':
        return process_directory(
            args.input_dir,
            args.output,
            create_vectors=not args.no_vectors
        )
    
    elif args.command == 'extract-tables':
        return extract_tables(args.file, args.output)
    
    elif args.command == 'search':
        return search_documents(args.query, args.top_k)
    
    elif args.command == 'to-markdown':
        return convert_to_markdown(args.file, args.output)
    
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

