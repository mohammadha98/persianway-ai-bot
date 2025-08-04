import os
import sys
import asyncio
from app.services.document_processor import get_document_processor
from app.services.knowledge_base import get_knowledge_base_service


async def main():
    """Initialize the knowledge base by processing all PDF documents and Excel QA files."""
    print("Starting Persian Agriculture Knowledge Base initialization...")
    print("Processing PDF documents in the 'docs' directory...")
    
    # Get the document processor and knowledge base service
    doc_processor = get_document_processor()
    kb_service = get_knowledge_base_service()
    
    # Process all PDFs
    pdf_count = doc_processor.process_all_pdfs()
    print(f"Successfully processed {pdf_count} PDF document chunks.")
    
    # Process all Excel QA files
    print("\nProcessing Excel QA files in the 'docs' directory...")
    qa_count = kb_service.process_excel_files()
    print(f"Successfully processed {qa_count} QA pairs from Excel files.")
    
    print("\nKnowledge base initialization complete!")
    print("\nYou can now query the knowledge base using the API:")
    print("  POST /api/knowledge/query")
    print("\nExample:")
    print("  curl -X POST \"http://localhost:8000/api/knowledge/query\" \\")
    print("       -H \"Content-Type: application/json\" \\")
    print("       -d '{\"question\":\"انواع کودهای شیمیایی کدامند؟\"}'")
    
    print("\nTo process additional Excel QA files:")
    print("  POST /api/knowledge/process-excel")


if __name__ == "__main__":
    asyncio.run(main())