from typing import List, Dict, Any, Optional
import uuid
import logging
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from app.core.config import settings
from app.services.document_processor import get_document_processor

# Set up logging for human referrals
referral_logger = logging.getLogger("human_referral")
file_handler = logging.FileHandler("human_referrals.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
referral_logger.addHandler(file_handler)
referral_logger.setLevel(logging.INFO)


class KnowledgeBaseService:
    """Service for retrieving information from the knowledge base using RAG.
    
    This service integrates the document processor with LangChain's retrieval
    capabilities to provide context-aware responses based on the document collection.
    """
    
    def __init__(self):
        """Initialize the knowledge base service."""
        self.document_processor = get_document_processor()
        
        # System prompt for agricultural knowledge in Persian
        self.system_prompt = settings.PERSIAN_AGRICULTURE_SYSTEM_PROMPT
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.OPENAI_MODEL_NAME,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS
        )
        
        # Initialize the retrieval QA chain
        self._qa_chain = None
    
    def _get_qa_chain(self):
        """Get or create the QA chain."""
        if self._qa_chain is None:
            # Get the vector store
            vector_store = self.document_processor.get_vector_store()
            
            # Create a retriever
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            # Create a prompt template that includes the system prompt
            template = """\n{system_prompt}\n\nمنابع:\n{context}\n\nسوال: {question}\n\nپاسخ:"""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"],
                partial_variables={"system_prompt": self.system_prompt}
            )
            
            # Create the QA chain
            self._qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
        
        return self._qa_chain
    
    def _calculate_confidence_score(self, query: str, source_documents: List[Any]) -> float:
        """Calculate a confidence score based on the relevance of retrieved documents.
        
        Args:
            query: The original query
            source_documents: The retrieved source documents
            
        Returns:
            A confidence score between 0 and 1
        """
        if not source_documents:
            return 0.0
            
        # For now, use a simple heuristic based on the number of documents and their scores
        # This can be enhanced with more sophisticated relevance scoring in the future
        scores = []
        for doc in source_documents:
            # Get similarity score if available, otherwise use a default value
            score = doc.metadata.get("score", 0.5)
            scores.append(score)
            
        # Average the scores
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return avg_score
    
    def _log_human_referral(self, query: str, sources: List[Dict[str, Any]], query_id: str) -> None:
        """Log a query that requires human attention.
        
        Args:
            query: The original query
            sources: The retrieved sources (if any)
            query_id: Unique identifier for the query
        """
        referral_logger.info(
            f"HUMAN REFERRAL NEEDED\n"
            f"Query ID: {query_id}\n"
            f"Query: {query}\n"
            f"Retrieved Sources: {len(sources)}\n"
            f"Timestamp: {datetime.now().isoformat()}\n"
            f"{'=' * 50}"
        )
    
    async def query_knowledge_base(self, query: str) -> Dict[str, Any]:
        """Query the knowledge base with a question.
        
        Args:
            query: The question to ask
            
        Returns:
            Dictionary containing the answer, source documents, and human referral flag
        """
        try:
            # Get the QA chain
            qa_chain = self._get_qa_chain()
            
            # Query the knowledge base
            result = qa_chain({"query": query})
            
            # Extract answer and sources
            answer = result["result"]
            source_documents = result["source_documents"]
            
            # Format sources for response
            sources = []
            for doc in source_documents:
                sources.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 0)
                })
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(query, source_documents)
            
            # Check if human referral is needed
            requires_human_support = confidence_score < settings.KNOWLEDGE_BASE_CONFIDENCE_THRESHOLD
            
            # Prepare response
            response = {
                "answer": answer if not requires_human_support else settings.HUMAN_REFERRAL_MESSAGE,
                "sources": sources,
                "requires_human_support": requires_human_support,
                "query_id": None
            }
            
            # If human referral is needed, generate a query ID and log the request
            if requires_human_support:
                query_id = str(uuid.uuid4())
                response["query_id"] = query_id
                self._log_human_referral(query, sources, query_id)
            
            return response
        except Exception as e:
            error_msg = f"Error querying knowledge base: {str(e)}"
            raise Exception(error_msg)


# Singleton instance
_knowledge_base_service = None


def get_knowledge_base_service() -> KnowledgeBaseService:
    """Get the knowledge base service instance.
    
    Returns:
        A singleton instance of the KnowledgeBaseService
    """
    global _knowledge_base_service
    if _knowledge_base_service is None:
        _knowledge_base_service = KnowledgeBaseService()
    return _knowledge_base_service