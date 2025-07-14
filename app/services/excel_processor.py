from typing import List, Dict, Any, Optional, Tuple
import os
import pandas as pd
from langchain.schema import Document
from langchain.vectorstores import Chroma

from app.core.config import settings
from app.services.document_processor import get_document_processor


class ExcelQAProcessor:
    """Service for processing Excel QA files and creating vector embeddings.
    
    This service handles loading Excel files with QA pairs, creating embeddings,
    and storing them in the vector database alongside PDF document chunks.
    """
    
    def __init__(self):
        """Initialize the Excel QA processor."""
        # Get the document processor to access embeddings and vector store
        self.document_processor = get_document_processor()
        
        # Set up the Excel directory path
        self.excel_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs")
        
    def validate_excel_structure(self, df: pd.DataFrame) -> bool:
        """Validate that the Excel file has the expected structure.
        
        Args:
            df: The pandas DataFrame containing the Excel data
            
        Returns:
            True if the structure is valid, False otherwise
        """
        # Check if required columns exist
        required_columns = ['Title', 'Question', 'Answer']
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: Required column '{col}' not found in Excel file")
                return False
        
        # Check if there's data in the required columns
        if df.empty or df['Question'].isna().all() or df['Answer'].isna().all():
            print("Error: Excel file contains no valid QA pairs")
            return False
            
        return True
    
    def process_excel_file(self, file_path: str) -> Tuple[int, List[Document]]:
        """Process a single Excel file and convert QA pairs to documents.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Tuple containing (number of QA pairs processed, list of Document objects)
        """
        try:
            # Read Excel file with proper encoding handling
            df = pd.read_excel(file_path, engine='openpyxl')
            
            # Validate structure
            if not self.validate_excel_structure(df):
                return 0, []
            
            # Extract QA pairs
            documents = []
            for _, row in df.iterrows():
                # Skip rows with missing Question or Answer
                if pd.isna(row['Question']) or pd.isna(row['Answer']):
                    continue
                
                # Create document for the QA pair
                title = row['Title'] if not pd.isna(row['Title']) else ""
                question = row['Question']
                answer = row['Answer']
                
                # Combine question and answer into a single document
                content = f"Question: {question}\nAnswer: {answer}"
                
                # Create document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": os.path.basename(file_path),
                        "file_path": file_path,
                        "source_type": "excel_qa",
                        "title": title,
                        "question": question,
                        "row_number": _ + 2,  # +2 because pandas is 0-indexed and Excel has a header row
                    }
                )
                
                documents.append(doc)
            
            return len(documents), documents
        except Exception as e:
            print(f"Error processing Excel file {file_path}: {str(e)}")
            return 0, []
    
    def process_all_excel_files(self) -> int:
        """Process all Excel files in the docs directory and add them to the vector store.
        
        Returns:
            Number of QA pairs processed
        """
        all_docs = []
        total_qa_pairs = 0
        
        # Get all Excel files in the docs directory
        excel_files = [os.path.join(self.excel_dir, f) for f in os.listdir(self.excel_dir) 
                      if f.lower().endswith(('.xlsx', '.xls'))]
        
        # Process each Excel file
        for excel_file in excel_files:
            qa_count, docs = self.process_excel_file(excel_file)
            total_qa_pairs += qa_count
            all_docs.extend(docs)
        
        # Add documents to vector store in batches to avoid token limit issues
        if all_docs:
            vector_store = self.document_processor.get_vector_store()
            
            # Process in batches of 100 documents to stay well under the 300k token limit
            batch_size = 100
            for i in range(0, len(all_docs), batch_size):
                batch = all_docs[i:i + batch_size]
                vector_store.add_documents(batch)
                print(f"Processed batch {i//batch_size + 1}/{(len(all_docs) + batch_size - 1)//batch_size} with {len(batch)} documents")
            
            vector_store.persist()
        
        return total_qa_pairs


# Singleton instance
_excel_qa_processor = None


def get_excel_qa_processor():
    """Get the Excel QA processor instance.
    
    Returns:
        A singleton instance of the ExcelQAProcessor
    """
    global _excel_qa_processor
    if _excel_qa_processor is None:
        _excel_qa_processor = ExcelQAProcessor()
    return _excel_qa_processor