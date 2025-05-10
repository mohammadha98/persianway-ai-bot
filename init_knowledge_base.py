import os
import sys
import asyncio
from app.services.document_processor import get_document_processor


async def main():
    """Initialize the knowledge base by processing all PDF documents."""
    print("Starting Persian Agriculture Knowledge Base initialization...")
    print("Processing PDF documents in the 'docs' directory...")
    
    # Get the document processor
    doc_processor = get_document_processor()
    
    # Process all PDFs
    doc_count = doc_processor.process_all_pdfs()
    
    print(f"Successfully processed {doc_count} document chunks.")
    print("Knowledge base initialization complete!")
    print("\nYou can now query the knowledge base using the API:")
    print("  POST /api/knowledge/query")
    print("\nExample:")
    print("  curl -X POST \"http://localhost:8000/api/knowledge/query\" \\")
    print("       -H \"Content-Type: application/json\" \\")
    print("       -d '{\"question\":\"انواع کودهای شیمیایی کدامند؟\"}'")


if __name__ == "__main__":
    asyncio.run(main())