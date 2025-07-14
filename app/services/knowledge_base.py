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
from app.services.excel_processor import get_excel_qa_processor

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
        self.excel_processor = get_excel_qa_processor()
        
        # System prompt for knowledge base
        self.system_prompt = settings.SYSTEM_PROMPT
        
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
    
    def process_excel_files(self) -> int:
        """Process all Excel QA files in the configured directory.
        
        Returns:
            Number of QA pairs processed
        """
        return self.excel_processor.process_all_excel_files()

    async def add_knowledge_contribution(
        self,
        title: str,
        content: str,
        source: str,
        meta_tags: List[str],
        author_name: Optional[str] = None,
        additional_references: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Adds a new knowledge entry to the vector store.

        Args:
            title: Title of the entry.
            content: Main body/content in Persian.
            source: The origin or reference for the knowledge.
            meta_tags: Comma-separated keywords for categorization.
            author_name: Optional name of the contributor.
            additional_references: Optional URLs or citations.

        Returns:
            A dictionary with the ID and timestamp of the new entry.
        """
        try:
            doc_id = str(uuid.uuid4())
            submitted_at = datetime.now().isoformat()

            # Prepare metadata
            metadata = {
                "source": source,
                "title": title,
                "meta_tags": ",".join(meta_tags), # Store as comma-separated string as per existing patterns if any, or adjust if vector store handles lists
                "author_name": author_name if author_name else "Unknown",
                "additional_references": additional_references if additional_references else "None",
                "submission_timestamp": submitted_at,
                "entry_type": "user_contribution", # Differentiate from PDF/Excel
                "id": doc_id
            }

            # Create Langchain Document
            # The main content for embedding should be a combination of title and content for better retrieval
            document_content = f"Title: {title}\n\nContent: {content}"
            langchain_document = self.document_processor.text_splitter.create_documents(
                texts=[document_content],
                metadatas=[metadata] # Pass metadata for each document
            )
            
            # Ensure documents are created and metadata is correctly assigned
            if not langchain_document:
                 raise ValueError("Failed to create document for vector store.")

            # Add to vector store
            vector_store = self.document_processor.get_vector_store()
            vector_store.add_documents(langchain_document)
            vector_store.persist() # Ensure data is saved

            return {
                "id": doc_id,
                "title": title,
                "submitted_at": submitted_at,
                "meta_tags": meta_tags,
                "source": source,
                "author_name": author_name,
                "additional_references": additional_references
            }
        except Exception as e:
            logging.error(f"Error adding knowledge contribution: {str(e)}")
            # Re-raise the exception so the route can handle it and return a 500 error
            raise
    
    def _is_content_relevant(self, query: str, qa_content: str) -> bool:
        """Check if the QA content is relevant to the user's query using semantic similarity.
        
        Args:
            query: The user's question
            qa_content: The QA pair content to check
            
        Returns:
            True if content appears relevant, False otherwise
        """
        # Convert to lowercase for comparison
        query_lower = query.lower()
        content_lower = qa_content.lower()
        
        # Define domain-specific keywords to detect topic mismatch
        domain_keywords = {
            'agriculture': [
                'کود', 'کشاورزی', 'خاک', 'کاشت', 'برداشت', 'آفت', 'بیماری', 'گیاه', 'محصول',
                'آبیاری', 'بذر', 'نهال', 'درخت', 'میوه', 'سبزی', 'غلات', 'دام', 'طیور',
                'fertilizer', 'agriculture', 'soil', 'plant', 'crop', 'farming', 'irrigation'
            ],
            'health_beauty': [
                'پوست', 'صورت', 'ماسک', 'کرم', 'زیبایی', 'سلامت', 'درمان', 'دارو', 'بیمار',
                'معده', 'شکم', 'ماساژ', 'روغن', 'مالت', 'خرما', 'شیر', 'کودک', 'نوزاد',
                'skin', 'face', 'mask', 'cream', 'beauty', 'health', 'treatment', 'medicine'
            ],
            'technology': [
                'کامپیوتر', 'نرم افزار', 'اپلیکیشن', 'وب سایت', 'برنامه نویسی', 'شبکه',
                'computer', 'software', 'application', 'website', 'programming', 'network'
            ],
            'finance': [
                'پول', 'بانک', 'سرمایه گذاری', 'بورس', 'اقتصاد', 'مالی', 'حسابداری',
                'money', 'bank', 'investment', 'stock', 'economy', 'financial', 'accounting'
            ]
        }
        
        # Detect query domain
        query_domain = None
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                query_domain = domain
                break
        
        # Detect content domain
        content_domain = None
        for domain, keywords in domain_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                content_domain = domain
                break
        
        print(f"Query: {query}")
        print(f"QA Content: {qa_content}")
        print(f"Query Domain: {query_domain}")
        print(f"Content Domain: {content_domain}")

        # If query domain is known and content domain is known and they differ, it's irrelevant
        if query_domain and content_domain and query_domain != content_domain:
            print("Domains mismatch, returning False")
            return False
        
        # If the query is clearly about one domain, but the content has no clear domain,
        # we should be cautious. For now, we allow it, but this could be refined.
        
        # If the content contains highly specific patterns from a different domain, it's likely irrelevant
        if query_domain and query_domain != 'health_beauty':
            highly_specific_health_patterns = [
                'مالت خرما',
                'شکم کودک',
                'سفیر سلامت',
                'پوست صورت',
                'جوشهای سرسیاه'
            ]
            if any(pattern in content_lower for pattern in highly_specific_health_patterns):
                return False

        return True
    
    async def query_knowledge_base(self, query: str) -> Dict[str, Any]:
        """Query the knowledge base with a question.
        
        Args:
            query: The question to ask
            
        Returns:
            Dictionary containing the answer, source documents, and human referral flag
        """
        try:
            # Get the vector store
            vector_store = self.document_processor.get_vector_store()
            
            # First, try to find exact or semantically similar matches in the QA database
            qa_results = vector_store.similarity_search_with_score(
                query, 
                k=8,  # Increased to get more candidates for relevance filtering
                filter={"source_type": "excel_qa"}
            )
            
            # Check if we have a high-confidence QA match that is also relevant
            qa_match_found = False
            qa_answer = ""
            qa_sources = []
            best_confidence = 0.0
            
            if qa_results:
                best_match = None
                best_confidence = 0.0

                # Filter results for relevance and find the best relevant match
                for qa_match, score in qa_results:
                    confidence = 1.0 - min(score, 1.0)
                    
                    if confidence > best_confidence and self._is_content_relevant(query, qa_match.page_content):
                        best_confidence = confidence
                        best_match = qa_match

                if best_match and best_confidence >= settings.QA_MATCH_THRESHOLD:
                    qa_match_found = True
                    
                    # Extract answer from the QA pair
                    content_parts = best_match.page_content.split("\nAnswer: ")
                    if len(content_parts) > 1:
                        qa_answer = content_parts[1]
                    
                    # Format QA sources
                    qa_sources = [{
                        "content": best_match.page_content,
                        "source": best_match.metadata.get("source", "Unknown"),
                        "page": 0,
                        "source_type": "excel_qa",
                        "title": best_match.metadata.get("title", "")
                    }]
            
            # If we have a high-confidence relevant QA match, return it directly
            if qa_match_found and qa_answer:
                return {
                    "answer": qa_answer,
                    "sources": qa_sources,
                    "requires_human_support": False,
                    "query_id": None,
                    "source_type": "excel_qa",
                    "confidence_score": best_confidence
                }
            
            # Otherwise, fall back to the PDF-based knowledge retrieval
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
                    "page": doc.metadata.get("page", 0),
                    "source_type": doc.metadata.get("source_type", "pdf")
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
                "query_id": None,
                "source_type": "pdf",
                "confidence_score": confidence_score
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