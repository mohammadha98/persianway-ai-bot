from typing import List, Dict, Any, Optional
import os
import tempfile
import logging
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from app.core.config import settings


class DocumentProcessor:
    """Service for processing PDF documents and creating vector embeddings.
    
    This service handles loading PDF files, splitting them into chunks,
    creating embeddings, and storing them in a vector database.
    """
    
    def __init__(self):
        """Initialize the document processor."""
        self.docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs")
        self.persist_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vectordb")
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embeddings
        # Use OpenAI API key for embeddings (OpenRouter doesn't support embeddings)
        api_key = settings.OPENAI_API_KEY
        print("DOCS PROCCESSOR API KEY")
        print(api_key)
        # Always use the fixed OpenAI embedding regardless of the model provider
        # This ensures the knowledge base works with both OpenAI and OpenRouter
        if not api_key or api_key == "":
            logging.error("OpenAI API key is not configured. Please set OPENAI_API_KEY in your environment.")
            logging.error("Vector embeddings will not be available. Knowledge base cannot function without embeddings.")
            self.embeddings_available = False
            self.embeddings = None
        else:
            try:
                # We'll always try to use the OpenAI embeddings with the provided key
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=api_key,
                    model=settings.OPENAI_EMBEDDING_MODEL
                )
                # Test the embeddings to make sure they work
                self.embeddings.embed_query("Test query to verify embeddings")
                self.embeddings_available = True
                logging.info("OpenAI embeddings initialized successfully.")
            except Exception as e:
                # If there's an error with the embeddings, log it and disable embeddings
                logging.error(f"Error initializing OpenAI embeddings: {str(e)}")
                logging.error("Vector embeddings will not be available. Knowledge base cannot function without embeddings.")
                logging.error("Please check your OpenAI API key and network connection.")
                # Set a flag to indicate that embeddings are not available
                self.embeddings_available = False
                # Create a dummy embeddings object that won't be used
                self.embeddings = None
        
        # Initialize text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize vector store
        self._vector_store = None
    
    def get_vector_store(self):
        """Get or create the vector store."""
        # If embeddings are not available, return None
        if not hasattr(self, 'embeddings_available') or not self.embeddings_available:
            logging.warning("Cannot access vector store: OpenAI embeddings are not available.")
            return None
            
        if self._vector_store is None:
            # Check if vector store exists
            if os.path.exists(self.persist_directory) and len(os.listdir(self.persist_directory)) > 0:
                # Load existing vector store
                self._vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                # Create new vector store
                self._vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
        
        return self._vector_store
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """Process a single PDF file and return the extracted documents.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Extract metadata from filename
            filename = os.path.basename(file_path)
            for doc in documents:
                doc.metadata["source"] = filename
                doc.metadata["file_path"] = file_path
            
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            
            return split_docs
        except Exception as e:
            print(f"Error processing PDF {file_path}: {str(e)}")
            return []
    
    def process_all_pdfs(self) -> int:
        """Process all PDF files in the docs directory and add them to the vector store.
        
        Returns:
            Number of documents processed
        """
        all_docs = []
        
        # Get all PDF files in the docs directory
        pdf_files = [os.path.join(self.docs_dir, f) for f in os.listdir(self.docs_dir) 
                    if f.lower().endswith('.pdf')]
        
        # Process each PDF file
        for pdf_file in pdf_files:
            docs = self.process_pdf(pdf_file)
            all_docs.extend(docs)
        
        # Add documents to vector store in batches to avoid token limit issues
        if all_docs:
            vector_store = self.get_vector_store()
            
            # Process in batches of 100 documents to stay well under the 300k token limit
            batch_size = 100
            for i in range(0, len(all_docs), batch_size):
                batch = all_docs[i:i + batch_size]
                vector_store.add_documents(batch)
                print(f"Processed batch {i//batch_size + 1}/{(len(all_docs) + batch_size - 1)//batch_size} with {len(batch)} documents")
            
            vector_store.persist()
        
        return len(all_docs)
    
    def search_documents(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search for documents relevant to the query.
        
        Args:
            query: The search query
            k: Number of documents to return
            
        Returns:
            List of documents with their content and metadata
        """
        vector_store = self.get_vector_store()
        docs = vector_store.similarity_search(query, k=k)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return results


# Singleton instance
_document_processor = None


def get_document_processor() -> DocumentProcessor:
    """Get the document processor instance.
    
    Returns:
        A singleton instance of the DocumentProcessor
    """
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor