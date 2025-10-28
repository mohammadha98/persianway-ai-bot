from typing import List, Dict, Any, Optional
import uuid
import logging
import os
from datetime import datetime
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from app.services.chat_service import get_llm
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
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
            
            # Create a comprehensive prompt template that includes system prompt for better model behavior control
            system_prompt = rag_settings.system_prompt
            base_template = rag_settings.prompt_template
            
            # Combine system prompt with the RAG template
            combined_template = f"""{system_prompt}

{base_template}"""

            # The new create_retrieval_chain expects 'input' instead of 'question',
            # so we replace the placeholder in the template.
            final_template = combined_template.replace("{question}", "{input}")
            
            # Create the QA chain with dynamic settings using new LangChain approach
            # Create a chat prompt template for the new chain
            chat_prompt = ChatPromptTemplate.from_template(final_template)
            
            # Create the document chain
            document_chain = create_stuff_documents_chain(self.llm, chat_prompt)
            
            # Create the retrieval chain
            self._qa_chain = create_retrieval_chain(retriever, document_chain)
        
        return self._qa_chain
    
    def _calculate_confidence_score(self, similarity_score: float) -> float:
        """
        Converts the similarity score (distance) from the vector store to a confidence score.
        Lower distance scores indicate higher similarity.

        Args:
            similarity_score: The distance score from the vector search (e.g., ChromaDB's L2 distance).

        Returns:
            A confidence score between 0 and 1, where 1 is most confident.
        """
        # ChromaDB returns L2 distance scores which can be any positive value.
        # Lower scores mean higher similarity. We use exponential decay to convert to confidence.
        # This approach handles scores > 1.0 gracefully and provides smooth scaling.
        
        # Use exponential decay: confidence = e^(-distance)
        # This naturally maps: distance=0 -> confidence=1, distance=∞ -> confidence=0
        import math
        confidence = math.exp(-similarity_score)
        
        # Ensure the result is within [0, 1] range
        return max(0.0, min(confidence, 1.0))
    
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
        meta_tags: List[str],
        source: Optional[str] = None,
        author_name: Optional[str] = None,
        additional_references: Optional[str] = None,
        uploaded_file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Adds a new knowledge entry to the vector store and relational database.

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
            # Generate a unique hash_id that will be used in both vector store and database
            hash_id = str(uuid.uuid4())
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
                            doc.metadata["hash_id"] = hash_id  # Use hash_id as common identifier
                            doc.metadata["id"] = hash_id       # Keep id for backward compatibility
                        
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
                            doc.metadata["hash_id"] = hash_id  # Use hash_id as common identifier
                            doc.metadata["id"] = hash_id       # Keep id for backward compatibility
                        
                        file_docs.extend(excel_docs)
                        processed_file = True

            # Prepare metadata for text contribution
            metadata = {
                "source": source if source else "Unknown",
                "title": title,
                "meta_tags": ",".join(meta_tags), # Store as comma-separated string as per existing patterns if any, or adjust if vector store handles lists
                "author_name": author_name if author_name else "Unknown",
                "additional_references": additional_references if additional_references else "None",
                "submission_timestamp": submitted_at,
                "entry_type": "user_contribution", # Differentiate from PDF/Excel
                "source_type": "qa_contribution", # Mark as QA contribution for retrieval priority
                "question": title, # Store the title as the question
                "answer": content, # Store the content as the answer
                "hash_id": hash_id,  # Use hash_id as common identifier
                "id": hash_id        # Keep id for backward compatibility
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
            
            # Persist changes to vector store
            vector_store.persist() # Ensure data is saved

            # After adding new documents, the QA chain should be reset to reflect the changes.
            self._qa_chain = None
            logging.info("Vector store updated. QA chain has been reset.")
            
            # Prepare document for database insertion
            db_document = {
                "hash_id": hash_id,
                "title": title,
                "content": content,
                "meta_tags": meta_tags,
                "author_name": author_name if author_name else "Unknown",
                "additional_references": additional_references.split(",") if additional_references else [],
                "submission_timestamp": submitted_at,
                "synced": True,
                "entry_type": "user_contribution"
            }
            
            # Add file information if a file was processed
            if processed_file:
                db_document["file_processed"] = True
                db_document["file_type"] = file_type
                db_document["file_name"] = os.path.basename(uploaded_file_path) if uploaded_file_path else None
                if file_type == 'excel' and 'qa_count' in locals():
                    db_document["qa_count"] = qa_count
            
            # Insert document into database
            from app.services.database import get_database_service
            db_service = await get_database_service()
            db_id = await db_service.insert_knowledge_document(db_document)
            logging.info(f"Document inserted into database with ID: {db_id}")

            # Prepare response
            response = {
                "id": hash_id,
                "title": title,
                "submitted_at": submitted_at,
                "meta_tags": meta_tags,
                "source": source,
                "author_name": author_name,
                "additional_references": additional_references,
                "db_id": db_id
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
        common_words = {'که', 'این', 'آن', 'در', 'به', 'از', 'با', 'برای', 'تا', 'و', 'یا', 'اما', 'چه', 'چی', 'کی', 'کجا', 'چرا', 'چگونه', 'بین', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words = [word for word in query_lower.split() if word not in common_words and len(word) > 2]
        
        # Check for direct word matches
        direct_matches = sum(1 for word in query_words if word in content_lower)
        match_ratio = direct_matches / max(len(query_words), 1)
        
        # If we have good word overlap, consider it relevant
        if match_ratio >= 0.5:  # At least 50% of query words should match
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
            'سیاست', 'انتخابات', 'دولت', 'مکتب', 'دیدگاه سیاسی', 'جنگ', 'politics', 'political', 'government', 'war'
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
    

    async def expand_query(self, query: str) -> Dict[str, Any]:
        """Expand a query using GPT-4o-mini to improve search results.
        
        Args:
            query: The original query string
            
        Returns:
            A dictionary containing the original query and expanded queries
        """
        try:
            from openai import AsyncOpenAI
            
            # Initialize OpenAI client
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Create a prompt for query expansion
            prompt = f"""Given the following query, generate 3 alternative versions that capture the same intent but with different wording or additional context. 
            Focus on expanding concepts, adding synonyms, and considering both Persian and English terminology.
            
            Original query: {query}
            
            Return ONLY a JSON object with the following format:
            {{
                "expanded_queries": [
                    "first alternative query",
                    "second alternative query",
                    "third alternative query"
                ]
            }}
            """
            
            # Call GPT-4o-mini
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that expands search queries to improve retrieval results."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=700
            )
            
            # Parse the response
            import json
            expanded_queries = json.loads(response.choices[0].message.content)
            
            # Return the original query and expanded queries
            return {
                "original_query": query,
                "expanded_queries": expanded_queries.get("expanded_queries", [])
            }
        except Exception as e:
            logging.error(f"Error in query expansion: {str(e)}")
            # Return just the original query if there's an error
            return {
                "original_query": query,
                "expanded_queries": []
            }

    def _extract_conversation_history(self, conversation_history) -> List[Dict[str, str]]:
        """
        Extract conversation messages from ConversationResponse format.
        
        Args:
            conversation_history: ConversationResponse object or list of messages
            
        Returns:
            List[Dict[str, str]]: List of messages with 'role' and 'content' keys
        """
        if not conversation_history:
            return []
            
        # Handle ConversationResponse object
        if hasattr(conversation_history, 'messages'):
            messages = []
            for msg in conversation_history.messages:
                messages.append({
                    'role': msg.role,
                    'content': msg.content
                })
            return messages
        
        # Handle list of ConversationResponse objects
        elif isinstance(conversation_history, list) and conversation_history:
            # If it's a list of ConversationResponse objects, take the latest one
            if hasattr(conversation_history[0], 'messages'):
                latest_conversation = conversation_history[-1]  # Get the most recent conversation
                messages = []
                for msg in latest_conversation.messages:
                    messages.append({
                        'role': msg.role,
                        'content': msg.content
                    })
                return messages
            # If it's already a list of dict messages, return as is
            elif isinstance(conversation_history[0], dict) and 'role' in conversation_history[0]:
                return conversation_history
        
        return []

    async def rewrite_query_with_context(self, history: List[Dict[str, str]], 
                                       user_message: str, 
                                       max_history: int = 4) -> str:
        """
        Rewrites the user's query based on recent conversation context
        (without adding knowledge outside the chat).
        Designed for contextual query building in Persian RAG pipelines.
        
        Args:
            history: List of conversation messages with 'role' and 'content' keys
            user_message: The current user message to rewrite
            max_history: Maximum number of previous messages to consider
            
        Returns:
            str: The rewritten query that incorporates conversation context
        """
        # If no history, return original message
        if not history:
            return user_message
            
        # Build multi-turn context window
        recent_context = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-max_history:]]
        )
        
        prompt = f"""
بازنویسی کن پرسش زیر به گونه‌ای که بدون نیاز به متن‌های قبلی قابل
جست‌وجو در پایگاه دانش باشد.
فقط از اطلاعات خود گفتگو استفاده کن و هیچ دانشی از بیرون اضافه نکن.
---
گفتگو:
{recent_context}
پرسش جدید کاربر:
{user_message}
---
پرسش بازنویسی‌شده:
"""
        
        try:
            # Use the existing LLM infrastructure from chat_service
            llm = await get_llm(temperature=0, max_tokens=100)
            
            # Create messages for the LLM
            from langchain.schema import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content="تو فقط بازنویسی contextual انجام می‌دهی."),
                HumanMessage(content=prompt)
            ]
            
            # Get response from LLM
            response = await llm.agenerate([messages])
            rewritten_text = response.generations[0][0].text.strip()
            
            # Return rewritten query or fallback to original
            return rewritten_text if rewritten_text else user_message
            
        except Exception as e:
            logging.error(f"Error in rewrite_query_with_context: {e}")
            # Fallback to original message if rewriting fails
            return user_message

    async def query_knowledge_base(self, query: str, conversation_history: List = None) -> Dict[str, Any]:
        """Query the knowledge base with a question.
        
        Args:
            query: The question to ask
            conversation_history: Previous conversation messages for context
            
        Returns:
            A dictionary with the answer, confidence score, and source information
        """
        try:
            # Extract conversation history and rewrite query with context if available
            extracted_history = self._extract_conversation_history(conversation_history)
            if extracted_history:
                # Rewrite the query to include conversation context
                query = await self.rewrite_query_with_context(extracted_history, query)
                logging.info(f"Query rewritten with context: {query}")
            
            # First, check if we have an exact or semantically similar match in our QA database
            vector_store = self.document_processor.get_vector_store()
            
            # If vector store is not available, raise an exception to be handled by chat service
            if vector_store is None:
                raise RuntimeError("Vector store not available. OpenAI embeddings may not be properly configured. Please check your OPENAI_API_KEY and ensure the vector database is initialized.")
                

            rag_settings = await self.config_service.get_rag_settings()
            
            # Expand the query for better search results
            expanded_query_result = await self.expand_query(query)
            all_queries = [expanded_query_result["original_query"]] + expanded_query_result["expanded_queries"]
            
            # Search for similar documents with scores using all queries
            all_docs_with_scores = []
            for search_query in all_queries:
                if search_query.strip():  # Only search non-empty queries
                    docs_with_scores = vector_store.similarity_search_with_score(search_query, k=20)
                    all_docs_with_scores.extend(docs_with_scores)
            
            # Remove duplicates and sort by score (lower is better for similarity)
            seen_docs = set()
            unique_docs_with_scores = []
            for doc, score in sorted(all_docs_with_scores, key=lambda x: x[1]):
                doc_key = (doc.page_content, doc.metadata.get("source", ""), doc.metadata.get("page", 0))
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    unique_docs_with_scores.append((doc, score))
            
            # Take the top results after deduplication
            docs_with_scores = unique_docs_with_scores[:rag_settings.top_k_results]
            
            # Check all documents equally for high-confidence matches, regardless of source type
            for doc, score in docs_with_scores:
                # For QA-type documents, check if we have a direct answer
                if doc.metadata.get("source_type") in ["excel_qa", "qa_contribution"]:
                    qa_question = doc.metadata.get("question", "") or doc.metadata.get("title", "")
                    qa_answer = doc.metadata.get("answer", "") or doc.metadata.get("content", "")
                    
                    # Check if this QA pair is relevant to the query
                    if self._is_content_relevant(query, qa_question) and qa_answer:
                        # Calculate confidence based on the vector search similarity score
                        confidence = self._calculate_confidence_score(score)
                        
                        # Load dynamic configuration for thresholds
                        await self.config_service._load_config()
                        rag_settings = await self.config_service.get_rag_settings()
                        
                        # # If confidence is high enough, return the answer directly
                        # if confidence >= rag_settings.qa_match_threshold:
                        #     return {
                        #         "answer": qa_answer,
                        #         "confidence_score": confidence,
                        #         "source_type": doc.metadata.get("source_type", "qa_contribution"),
                        #         "requires_human_support": False,
                        #         "query_id": None,
                        #         "sources": [{
                        #             "content": qa_answer,
                        #             "source": doc.metadata.get("source", "Unknown"),
                        #             "page": doc.metadata.get("page", 1),
                        #             "source_type": doc.metadata.get("source_type", "qa_contribution"),
                        #             "title": qa_question
                        #         }]
                        #     }
            
            # If no high-confidence direct match found, use the QA chain with all documents
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
                
            # Get answer from QA chain using new input format
            docs = [doc for doc, score in docs_with_scores]
            result = qa_chain.invoke({"input": query, "context": docs})
            answer = result["answer"]
            source_docs = result.get("context", [])
            
            # Calculate confidence score
            confidence = 0.0
            if source_docs:
                # Use the score of the most relevant document for confidence calculation
                if docs_with_scores:
                    most_relevant_score = docs_with_scores[0][1]  # score is the second item in the tuple
                    confidence = self._calculate_confidence_score(most_relevant_score)
                else:
                    confidence = 0.0
            
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
            source_type = "unknown"
            if source_docs:
                source_type = source_docs[0].metadata.get("source_type", "unknown")
            response = {
                "answer": answer,
                "confidence_score": confidence,
                "source_type": source_type,
                "requires_human_support": requires_human,
                "query_id": str(uuid.uuid4()) if requires_human else None,
                "sources": sources
            }
         
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

    async def remove_knowledge_contribution(self, hash_id: str) -> Dict[str, Any]:
        """Removes a knowledge entry from both the vector store and relational database.
    
        Args:
            hash_id: The unique hash_id of the entry to remove.
    
        Returns:
            A dictionary with the removal status and details.
        """
        try:
            removed_count = 0
            vector_removal_success = False
            db_removal_success = False
            
            # Get vector store
            vector_store = self.document_processor.get_vector_store()
            
            if vector_store is None:
                logging.warning("Vector store is not available. Cannot remove documents from vector database.")
            else:
                # ChromaDB delete by metadata using where clause <mcreference link="https://github.com/langchain-ai/langchain/discussions/1690" index="5">5</mcreference>
                try:
                    # Access the underlying ChromaDB collection to delete by metadata
                    collection = vector_store._collection
                    
                    # Get documents with the specified hash_id to count them before deletion
                    existing_docs = collection.get(where={"hash_id": hash_id})
                    
                    # Safely get the count of documents
                    if isinstance(existing_docs, dict) and 'ids' in existing_docs:
                        removed_count = len(existing_docs['ids']) if existing_docs['ids'] else 0
                    else:
                        # Fallback: if the structure is unexpected, assume no documents found
                        removed_count = 0
                        logging.warning(f"Unexpected structure from collection.get(): {type(existing_docs)}")
                    
                    if removed_count > 0:
                        # Delete documents by metadata <mcreference link="https://github.com/langchain-ai/langchain/discussions/1690" index="5">5</mcreference>
                        collection.delete(where={"hash_id": hash_id})
                        
                        # Persist changes to vector store
                        vector_store.persist()
                        
                        # Reset QA chain to reflect changes
                        self._qa_chain = None
                        
                        vector_removal_success = True
                        logging.info(f"Successfully removed {removed_count} documents from vector store with hash_id: {hash_id}")
                    else:
                        logging.info(f"No documents found in vector store with hash_id: {hash_id}")
                        vector_removal_success = True  # Consider it successful if nothing to remove
                        
                except Exception as e:
                    logging.error(f"Error removing documents from vector store: {str(e)}")
                    # Try alternative method using LangChain's delete method if available <mcreference link="https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html" index="3">3</mcreference>
                    try:
                        # Get all documents and find IDs with matching hash_id
                        all_docs = collection.get(include=['metadatas'])
                        ids_to_delete = []
                        
                        # Safely handle the response structure
                        if isinstance(all_docs, dict) and 'metadatas' in all_docs and 'ids' in all_docs:
                            if all_docs['metadatas']:
                                for i, metadata in enumerate(all_docs['metadatas']):
                                    if metadata and metadata.get('hash_id') == hash_id:
                                        ids_to_delete.append(all_docs['ids'][i])
                        else:
                            logging.warning(f"Unexpected structure from collection.get(include=['metadatas']): {type(all_docs)}")
                        
                        if ids_to_delete:
                            # Use LangChain's delete method <mcreference link="https://github.com/langchain-ai/langchain/discussions/17797" index="1">1</mcreference>
                            vector_store.delete(ids=ids_to_delete)
                            removed_count = len(ids_to_delete)
                            vector_removal_success = True
                            
                            # Persist changes and reset QA chain
                            vector_store.persist()
                            self._qa_chain = None
                            
                            logging.info(f"Successfully removed {removed_count} documents using alternative method with hash_id: {hash_id}")
                        else:
                            logging.info(f"No documents found with hash_id: {hash_id}")
                            vector_removal_success = True
                            
                    except Exception as e2:
                        logging.error(f"Alternative removal method also failed: {str(e2)}")
                        vector_removal_success = False
            
            # Update database document (mark as unsynced instead of deleting)
            try:
                from app.services.database import get_database_service
                db_service = await get_database_service()
                
                # If vector removal was successful, mark the document as unsynced
                if vector_removal_success and removed_count > 0:
                    try:
                        db_update_success = await db_service.update_knowledge_document_sync_status(hash_id, synced=False)
                        if db_update_success:
                            db_removal_success = True
                            logging.info(f"Successfully marked document as unsynced with hash_id: {hash_id}")
                        else:
                            db_removal_success = False
                            logging.warning(f"No document found in database with hash_id: {hash_id}")
                    except Exception as sync_error:
                        logging.error(f"Failed to update sync status for hash_id {hash_id}: {str(sync_error)}")
                        db_removal_success = False
                else:
                    # Even if vector removal failed, try to mark as unsynced
                    try:
                        db_update_success = await db_service.update_knowledge_document_sync_status(hash_id, synced=False)
                        if db_update_success:
                            db_removal_success = True
                            logging.info(f"Marked document as unsynced (vector removal failed) with hash_id: {hash_id}")
                        else:
                            db_removal_success = False
                            logging.warning(f"No document found in database with hash_id: {hash_id}")
                    except Exception as sync_error:
                        logging.error(f"Failed to update sync status for hash_id {hash_id}: {str(sync_error)}")
                        db_removal_success = False
                    
            except Exception as e:
                logging.error(f"Error updating document in database: {str(e)}")
                db_removal_success = False
            
            # Prepare response
            overall_success = vector_removal_success and db_removal_success
            
            response = {
                "success": overall_success,
                "hash_id": hash_id,
                "removed_from_vector_store": vector_removal_success,
                "removed_from_database": db_removal_success,
                "documents_removed_count": removed_count,
                "timestamp": datetime.now().isoformat()
            }
            
            if overall_success:
                logging.info(f"Successfully removed knowledge contribution with hash_id: {hash_id}")
            else:
                logging.warning(f"Partial or failed removal of knowledge contribution with hash_id: {hash_id}")
            
            return response
            
        except Exception as e:
            logging.error(f"Error removing knowledge contribution: {str(e)}")
            return {
                "success": False,
                "hash_id": hash_id,
                "error": str(e),
                "removed_from_vector_store": False,
                "removed_from_database": False,
                "documents_removed_count": 0,
                "timestamp": datetime.now().isoformat()
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