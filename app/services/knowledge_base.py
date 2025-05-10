from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from app.core.config import settings
from app.services.document_processor import get_document_processor


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
    
    async def query_knowledge_base(self, query: str) -> Dict[str, Any]:
        """Query the knowledge base with a question.
        
        Args:
            query: The question to ask
            
        Returns:
            Dictionary containing the answer and source documents
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
            
            return {
                "answer": answer,
                "sources": sources
            }
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