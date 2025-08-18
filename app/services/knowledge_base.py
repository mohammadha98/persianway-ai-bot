from typing import List, Dict, Any, Optional
import uuid
import logging
import os
from datetime import datetime
from langchain.chains import RetrievalQA
from app.services.chat_service import get_llm
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from app.core.config import settings
from app.services.document_processor import get_document_processor
from app.services.excel_processor import get_excel_qa_processor
from app.services.config_service import ConfigService

# Set up logging for human referrals
referral_logger = logging.getLogger("human_referral")
file_handler = logging.FileHandler("human_referrals.log", encoding="utf-8")
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
        self.config_service = ConfigService()
        
        # Initialize the retrieval QA chain
        self._qa_chain = None
        self.llm = None
    
    async def _get_qa_chain(self):
        """Get or create the QA chain.
        
        Returns:
            A RetrievalQA chain or None if vector store is not available
        """
        if self._qa_chain is None:
            # Load dynamic configuration
            await self.config_service._load_config()
            rag_settings =await self.config_service.get_rag_settings()
            
            # Initialize LLM if not already done
            if self.llm is None:
                self.llm = await get_llm(temperature=rag_settings.temperature)
            
            # Get the vector store
            vector_store = self.document_processor.get_vector_store()
            
            # Check if vector store is available
            if vector_store is None:
                logging.warning("Vector store is not available. QA chain cannot be created.")
                return None
            
            # Create a retriever with dynamic settings
            retriever = vector_store.as_retriever(
                search_type=rag_settings.search_type,
                search_kwargs={"k": rag_settings.top_k_results}
            )
            
            # Create a prompt template without including the system prompt to avoid duplication
            template = rag_settings.prompt_template
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create the QA chain with dynamic settings
            self._qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
        
        return self._qa_chain
    
    def _calculate_confidence_score(self, query: str, document: Any) -> float:
        """Calculate a confidence score based on the relevance of a document.
        
        Args:
            query: The original query
            document: The retrieved document
            
        Returns:
            A confidence score between 0 and 1
        """
        if not document:
            return 0.0
        
        query_lower = query.lower()
        doc_content = document.page_content.lower()
        
        # Check if document is relevant to the query
        if not self._is_content_relevant(query, document.page_content):
            return 0.2  # Low confidence for irrelevant documents
        
        # Calculate relevance score based on multiple factors
        relevance_score = 0.0
        
        # Factor 1: Direct keyword matches (40% weight)
        query_words = [word for word in query_lower.split() if len(word) > 2]
        if query_words:
            matches = sum(1 for word in query_words if word in doc_content)
            keyword_score = matches / len(query_words)
            relevance_score += keyword_score * 0.4
        
        # Factor 2: Document length and completeness (20% weight)
        # Longer, more complete documents tend to be more reliable
        doc_length = len(document.page_content)
        length_score = min(doc_length / 500, 1.0)  # Normalize to 500 chars
        relevance_score += length_score * 0.2
        
        # Factor 3: Source type preference (20% weight)
        source_type = document.metadata.get("source_type", "unknown")
        if source_type == "excel_qa":
            source_score = 1.0  # Prefer structured QA pairs
        elif source_type == "pdf":
            source_score = 0.8  # PDF content is good but less structured
        else:
            source_score = 0.6  # Other sources
        relevance_score += source_score * 0.2
        
        # Factor 4: Semantic similarity (20% weight)
        # Check for semantic relationships
        semantic_score = 0.0
        semantic_keywords = {
            'پوست': ['skin', 'face', 'facial', 'dermal'],
            'مو': ['hair', 'scalp'],
            'ریزش': ['loss', 'fall', 'thinning'],
            'زیبایی': ['beauty', 'cosmetic'],
            'سلامت': ['health', 'wellness'],
            'درمان': ['treatment', 'therapy', 'cure'],
            'دارو': ['medicine', 'medication', 'drug']
        }
        
        for persian_term, english_terms in semantic_keywords.items():
            if persian_term in query_lower:
                if persian_term in doc_content or any(term in doc_content for term in english_terms):
                    semantic_score += 0.5
            for eng_term in english_terms:
                if eng_term in query_lower:
                    if persian_term in doc_content or eng_term in doc_content:
                        semantic_score += 0.5
        
        semantic_score = min(semantic_score, 1.0)  # Cap at 1.0
        relevance_score += semantic_score * 0.2
        
        return min(relevance_score, 1.0)
    
    def _log_human_referral(self, query: str, answer: str, confidence: float) -> None:
        """Log a query that requires human attention.
        
        Args:
            query: The original query
            answer: The generated answer
            confidence: The confidence score
        """
        query_id = str(uuid.uuid4())
        referral_logger.info(
            f"HUMAN REFERRAL NEEDED\n"
            f"Query ID: {query_id}\n"
            f"Query: {query}\n"
            f"Answer: {answer[:100]}...\n"
            f"Confidence: {confidence}\n"
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
        uploaded_file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Adds a new knowledge entry to the vector store.

        Args:
            title: Title of the entry.
            content: Main body/content in Persian.
            source: The origin or reference for the knowledge.
            meta_tags: Comma-separated keywords for categorization.
            author_name: Optional name of the contributor.
            additional_references: Optional URLs or citations.
            uploaded_file_path: Optional path to an uploaded PDF or Excel file.

        Returns:
            A dictionary with the ID and timestamp of the new entry, and file processing information if applicable.
        """
        try:
            doc_id = str(uuid.uuid4())
            submitted_at = datetime.now().isoformat()
            processed_file = False
            file_type = None
            file_docs = []

            # Process uploaded file if provided
            if uploaded_file_path:
                file_ext = uploaded_file_path.lower().split('.')[-1]
                
                # Process PDF file
                if file_ext == 'pdf':
                    file_type = 'pdf'
                    # Use the document processor to process the PDF
                    pdf_docs = self.document_processor.process_pdf(uploaded_file_path)
                    if pdf_docs:
                        # Add custom metadata to the PDF documents
                        for doc in pdf_docs:
                            doc.metadata["source"] = os.path.basename(uploaded_file_path)
                            doc.metadata["title"] = title
                            doc.metadata["meta_tags"] = ",".join(meta_tags)
                            doc.metadata["author_name"] = author_name if author_name else "Unknown"
                            doc.metadata["additional_references"] = additional_references if additional_references else "None"
                            doc.metadata["submission_timestamp"] = submitted_at
                            doc.metadata["entry_type"] = "user_contribution_pdf"
                            doc.metadata["id"] = doc_id
                        
                        file_docs.extend(pdf_docs)
                        processed_file = True
                
                # Process Excel file
                elif file_ext in ['xlsx', 'xls']:
                    file_type = 'excel'
                    # Use the excel processor to process the Excel file
                    qa_count, excel_docs = self.excel_processor.process_excel_file(uploaded_file_path)
                    if excel_docs:
                        # Add custom metadata to the Excel documents
                        for doc in excel_docs:
                            doc.metadata["meta_tags"] = ",".join(meta_tags)
                            doc.metadata["author_name"] = author_name if author_name else "Unknown"
                            doc.metadata["additional_references"] = additional_references if additional_references else "None"
                            doc.metadata["submission_timestamp"] = submitted_at
                            doc.metadata["entry_type"] = "user_contribution_excel"
                            doc.metadata["id"] = doc_id
                        
                        file_docs.extend(excel_docs)
                        processed_file = True

            # Prepare metadata for text contribution
            metadata = {
                "source": source,
                "title": title,
                "meta_tags": ",".join(meta_tags), # Store as comma-separated string as per existing patterns if any, or adjust if vector store handles lists
                "author_name": author_name if author_name else "Unknown",
                "additional_references": additional_references if additional_references else "None",
                "submission_timestamp": submitted_at,
                "entry_type": "user_contribution", # Differentiate from PDF/Excel
                "source_type": "qa_contribution", # Mark as QA contribution for retrieval priority
                "question": title, # Store the title as the question
                "answer": content, # Store the content as the answer
                "id": doc_id
            }

            # Create Langchain Document for text contribution
            document_content = f"Title: {title}\n\nContent: {content}"
            langchain_document = self.document_processor.text_splitter.create_documents(
                texts=[document_content],
                metadatas=[metadata] # Pass metadata for each document
            )
            
            # Ensure documents are created and metadata is correctly assigned
            if not langchain_document and not file_docs:
                 raise ValueError("Failed to create document for vector store.")

            # Add to vector store
            vector_store = self.document_processor.get_vector_store()
            
            # Add text contribution documents
            if langchain_document:
                vector_store.add_documents(langchain_document)
            
            # Add file documents if any
            if file_docs:
                # Process in batches to avoid token limit issues
                batch_size = 100
                for i in range(0, len(file_docs), batch_size):
                    batch = file_docs[i:i + batch_size]
                    vector_store.add_documents(batch)
                    logging.info(f"Processed batch {i//batch_size + 1}/{(len(file_docs) + batch_size - 1)//batch_size} with {len(batch)} documents from uploaded file")
            
            # Persist changes
            vector_store.persist() # Ensure data is saved

            # After adding new documents, the QA chain should be reset to reflect the changes.
            self._qa_chain = None
            logging.info("Knowledge base updated. QA chain has been reset.")

            # Prepare response
            response = {
                "id": doc_id,
                "title": title,
                "submitted_at": submitted_at,
                "meta_tags": meta_tags,
                "source": source,
                "author_name": author_name,
                "additional_references": additional_references
            }
            
            # Add file information if a file was processed
            if processed_file:
                response["file_processed"] = True
                response["file_type"] = file_type
                response["file_name"] = os.path.basename(uploaded_file_path) if uploaded_file_path else None
                if file_type == 'excel':
                    response["qa_count"] = qa_count if 'qa_count' in locals() else 0
            
            return response
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
        
        # Extract key terms from query (remove common words)
        common_words = {'که', 'این', 'آن', 'در', 'به', 'از', 'با', 'برای', 'تا', 'و', 'یا', 'اما', 'چه', 'چی', 'کی', 'کجا', 'چرا', 'چگونه', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words = [word for word in query_lower.split() if word not in common_words and len(word) > 2]
        
        # Check for direct word matches
        direct_matches = sum(1 for word in query_words if word in content_lower)
        match_ratio = direct_matches / max(len(query_words), 1)
        
        # If we have good word overlap, consider it relevant
        if match_ratio >= 0.3:  # At least 30% of query words should match
            return True
            
        # Define domain-specific keywords to detect topic mismatch
        domain_keywords = {
            'agriculture': [
                'کود', 'کشاورزی', 'خاک', 'کاشت', 'برداشت', 'آفت', 'بیماری', 'گیاه', 'محصول',
                'آبیاری', 'بذر', 'نهال', 'درخت', 'میوه', 'سبزی', 'غلات', 'دام', 'طیور',
                'fertilizer', 'agriculture', 'soil', 'plant', 'crop', 'farming', 'irrigation'
            ],
            'health_beauty': [
                'پوست', 'صورت', 'ماسک', 'کرم', 'زیبایی', 'سلامت', 'درمان', 'دارو', 'بیمار',
                'معده', 'شکم', 'ماساژ', 'روغن', 'مالت', 'خرما', 'شیر', 'کودک', 'نوزاد', 'مو', 'ریزش',
                'skin', 'face', 'mask', 'cream', 'beauty', 'health', 'treatment', 'medicine', 'hair'
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
        
        # If query domain is known and content domain is known and they differ, it's irrelevant
        if query_domain and content_domain and query_domain != content_domain:
            return False
        
        # Check for semantic relevance using expanded keyword matching
        semantic_keywords = {
            # Health and beauty semantic mapping
            'پوست': ['skin', 'face', 'facial', 'dermal'],
            'مو': ['hair', 'scalp'],
            'ریزش': ['loss', 'fall', 'thinning'],
            'زیبایی': ['beauty', 'cosmetic'],
            'سلامت': ['health', 'wellness'],
            'درمان': ['treatment', 'therapy', 'cure'],
            'دارو': ['medicine', 'medication', 'drug'],
            # Agriculture semantic mapping
            'کود': ['fertilizer', 'nutrient'],
            'کشاورزی': ['agriculture', 'farming'],
            'خاک': ['soil', 'earth'],
            'گیاه': ['plant', 'vegetation'],
            'محصول': ['crop', 'produce']
        }
        
        # Check for semantic matches
        for persian_term, english_terms in semantic_keywords.items():
            if persian_term in query_lower:
                if persian_term in content_lower or any(term in content_lower for term in english_terms):
                    return True
            for eng_term in english_terms:
                if eng_term in query_lower:
                    if persian_term in content_lower or eng_term in content_lower:
                        return True
        
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

        # Add explicit checks for political/unrelated content
        political_patterns = [
            'سیاست', 'انتخابات', 'دولت', 'مکتب', 'دیدگاه سیاسی', 'politics', 'political', 'government'
        ]
        if any(pattern in query_lower for pattern in political_patterns):
            return False
            
        # Be more conservative - only return True if we have strong evidence of relevance
        # Either through domain matching or semantic keyword matching
        if query_domain and content_domain and query_domain == content_domain:
            return True
            
        # Check if we found semantic matches earlier
        for persian_term, english_terms in semantic_keywords.items():
            if persian_term in query_lower:
                if persian_term in content_lower or any(term in content_lower for term in english_terms):
                    return True
            for eng_term in english_terms:
                if eng_term in query_lower:
                    if persian_term in content_lower or eng_term in content_lower:
                        return True
        
        # Default to False for better precision
        return False
    
    async def query_knowledge_base(self, query: str) -> Dict[str, Any]:
        """Query the knowledge base with a question.
        
        Args:
            query: The question to ask
            
        Returns:
            A dictionary with the answer, confidence score, and source information
        """
        try:
            # First, check if we have an exact or semantically similar match in our QA database
            vector_store = self.document_processor.get_vector_store()
            
            # If vector store is not available, raise an exception to be handled by chat service
            if vector_store is None:
                raise RuntimeError("Vector store not available. OpenAI embeddings may not be properly configured. Please check your OPENAI_API_KEY and ensure the vector database is initialized.")
                

            rag_settings = await self.config_service.get_rag_settings()
            # Search for similar documents with scores
            docs_with_scores = vector_store.similarity_search_with_score(query, k=rag_settings.top_k_results)
            
            # Filter and prioritize "excel_qa" and "qa_contribution" source types if available
            qa_docs = [doc for doc, score in docs_with_scores if doc.metadata.get("source_type") in ["excel_qa", "qa_contribution"]]
            
            # Check if we have a high-confidence, relevant QA match
            for doc in qa_docs:
                qa_question = doc.metadata.get("question", "") or doc.metadata.get("title", "")
                qa_answer = doc.metadata.get("answer", "") or doc.metadata.get("content", "")
                
                # Check if this QA pair is relevant to the query
                if self._is_content_relevant(query, qa_question) and qa_answer:
                    # Calculate confidence based on similarity and relevance
                    confidence = self._calculate_confidence_score(query, doc)
                    
                    # Load dynamic configuration for thresholds
                    await self.config_service._load_config()
                    rag_settings = await self.config_service.get_rag_settings()
                    
                    # If confidence is high enough, return the answer directly
                    if confidence >= rag_settings.qa_match_threshold:
                        return {
                            "answer": qa_answer,
                            "confidence_score": confidence,
                            "source_type": "excel_qa",
                            "requires_human_support": False,
                            "query_id": None,
                            "sources": [{
                                "content": qa_answer,
                                "source": doc.metadata.get("source", "Unknown"),
                                "page": doc.metadata.get("page", 1),
                                "source_type": doc.metadata.get("source_type", "qa_contribution"),
                                "title": qa_question
                            }]
                        }
            
            # If no high-confidence QA match, fall back to PDF-based knowledge retrieval
            qa_chain = await self._get_qa_chain()
            
            # If QA chain is not available, return a fallback message
            if qa_chain is None:
                logging.warning("QA chain not available. Cannot query knowledge base.")
                await self.config_service._load_config()
                rag_settings =await self.config_service.get_rag_settings()
                return {
                    "answer": rag_settings.human_referral_message,
                    "confidence_score": 0.0,
                    "source_type": "system",
                    "requires_human_support": True,
                    "query_id": str(uuid.uuid4()),
                    "sources": []
                }
                
            # Get answer from QA chain
            result = qa_chain({"query": query})
            answer = result["result"]
            source_docs = result.get("source_documents", [])
            
            # Calculate confidence score
            confidence = 0.0
            if source_docs:
                # Use the first (most relevant) document for confidence calculation
                confidence = self._calculate_confidence_score(query, source_docs[0])
            
            # Get dynamic configuration if not already loaded
            if 'rag_settings' not in locals():
                await self.config_service._load_config()
                rag_settings =await self.config_service.get_rag_settings()
            
            # Determine if human referral is needed based on confidence
            requires_human = confidence < rag_settings.knowledge_base_confidence_threshold
            
            # If confidence is too low, log for human review
            if requires_human:
                self._log_human_referral(query, answer, confidence)
            
            # Prepare sources list
            sources = []
            if source_docs:
                for doc in source_docs:
                    sources.append({
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", 1),
                        "source_type": doc.metadata.get("source_type", "unknown")
                    })
            
            # Prepare response
            source_type = "pdf"
            if source_docs:
                source_type = source_docs[0].metadata.get("source_type", "pdf")
            elif qa_docs:  # If we have QA docs but didn't use them (low confidence), still show their type
                source_type = qa_docs[0].metadata.get("source_type", "pdf")
            response = {
                "answer": answer,
                "confidence_score": confidence,
                "source_type": source_type,
                "requires_human_support": requires_human,
                "query_id": str(uuid.uuid4()) if requires_human else None,
                "sources": sources
            }
            print("RESPONSE IN KNOWLEDGEBASE:::::")
            print(response)
            return response
            
        except Exception as e:
            logging.error(f"Error querying knowledge base: {str(e)}")
            # Return a proper error response instead of None
            try:
                await self.config_service._load_config()
                rag_settings = await self.config_service.get_rag_settings()
                error_message = rag_settings.human_referral_message
            except:
                error_message = "متأسفانه، خطایی در سیستم رخ داده است. لطفاً دوباره تلاش کنید."
            
            return {
                "answer": error_message,
                "confidence_score": 0.0,
                "source_type": "system",
                "requires_human_support": True,
                "query_id": str(uuid.uuid4()),
                "sources": []
            }


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