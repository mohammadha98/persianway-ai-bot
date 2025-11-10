from typing import List, Dict, Any, Optional
import uuid
import logging
import os
import hashlib
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
    
    def _normalize_documents_for_context(self, docs: List[Any], max_content_length: int = 1500) -> List[Any]:
        """
        Normalize and clean documents before passing to the QA chain.
        Removes unnecessary metadata and limits content length for better LLM performance.
        
        This function:
        - Removes technical metadata (file paths, creation dates, producers, etc.)
        - Keeps only essential metadata (source, title, page, author, etc.)
        - Truncates long content to avoid token limit issues
        - Limits metadata string lengths
        
        Args:
            docs: List of LangChain Document objects
            max_content_length: Maximum length for document content (default: 1500 chars)
            
        Returns:
            List of cleaned Document objects with minimal, relevant metadata
        """
        from langchain.schema import Document
        
        normalized_docs = []
        total_original_size = 0
        total_normalized_size = 0
        
        for idx, doc in enumerate(docs):
            # Track original size
            original_size = len(str(doc.page_content)) + len(str(doc.metadata))
            total_original_size += original_size
            
            # Extract only relevant metadata fields
            clean_metadata = {}
            
            # Keep only essential metadata fields
            essential_fields = {
                'source': 100,        # File name or source identifier
                'title': 200,         # Document or section title
                'page': None,         # Page number (keep as is)
                'source_type': 50,    # Type of source (qa, pdf, excel, etc.)
                'question': 300,      # For QA pairs
                'answer': 500,        # For QA pairs
                'meta_tags': 200,     # Tags for categorization
                'author_name': 100    # Author information
            }
            
            for field, max_length in essential_fields.items():
                if field in doc.metadata and doc.metadata[field]:
                    value = doc.metadata[field]
                    
                    # Handle different value types
                    if isinstance(value, str):
                        # Trim whitespace
                        value = value.strip()
                        # Limit string length if max_length is specified
                        if max_length and len(value) > max_length:
                            value = value[:max_length] + "..."
                        clean_metadata[field] = value
                    elif isinstance(value, (int, float, bool)):
                        # Keep numeric and boolean values as is
                        clean_metadata[field] = value
                    elif value is not None:
                        # Convert other types to string and limit length
                        str_value = str(value)
                        if max_length and len(str_value) > max_length:
                            str_value = str_value[:max_length] + "..."
                        clean_metadata[field] = str_value
            
            # Limit page content to reasonable length (avoid token limit issues)
            clean_content = doc.page_content.strip()
            if len(clean_content) > max_content_length:
                # Try to cut at a sentence boundary if possible
                truncated = clean_content[:max_content_length]
                last_period = truncated.rfind('.')
                last_newline = truncated.rfind('\n')
                cut_point = max(last_period, last_newline)
                
                if cut_point > max_content_length * 0.8:  # Only use sentence boundary if it's not too far back
                    clean_content = truncated[:cut_point + 1] + "..."
                else:
                    clean_content = truncated + "..."
            
            # Create a new document with cleaned data
            normalized_doc = Document(
                page_content=clean_content,
                metadata=clean_metadata
            )
            normalized_docs.append(normalized_doc)
            
            # Track normalized size
            normalized_size = len(clean_content) + len(str(clean_metadata))
            total_normalized_size += normalized_size
            
            # Log individual document normalization (debug level)
            logging.debug(
                f"Normalized doc {idx+1}: "
                f"original_size={original_size}, "
                f"normalized_size={normalized_size}, "
                f"reduction={((original_size - normalized_size) / original_size * 100):.1f}%"
            )
        
        # Log overall normalization stats
        if total_original_size > 0:
            reduction_percent = ((total_original_size - total_normalized_size) / total_original_size * 100)
            logging.info(
                f"Document normalization complete: "
                f"{len(docs)} docs, "
                f"original_size={total_original_size} chars, "
                f"normalized_size={total_normalized_size} chars, "
                f"reduction={reduction_percent:.1f}%"
            )
        
        return normalized_docs
    
    def _calculate_confidence_score(self, docs_with_scores: List[tuple], top_n: int = 3) -> float:
        """
        Calculates multi-factor confidence score based on:
        1. Best document score (primary factor)
        2. Score consistency across top results (secondary factor)
        3. Number of relevant documents found (coverage factor)
        
        Lower distance scores indicate higher similarity in L2 distance.

        Args:
            docs_with_scores: List of (document, score) tuples from vector search
            top_n: Number of top documents to consider for confidence calculation

        Returns:
            A confidence score between 0 and 1, where 1 is most confident.
        """
        import math
        import numpy as np
        
        if not docs_with_scores:
            return 0.0
        
        # Extract scores from top N documents
        top_scores = [score for _, score in docs_with_scores[:min(top_n, len(docs_with_scores))]]
        
        # --- Factor 1: Best Score (60% weight) ---
        # Convert best similarity score to confidence using logistic decay
        best_score = top_scores[0]
        midpoint = 1.5  # Distance at which confidence is 50%
        scale = 5.0     # Steepness of decay
        best_confidence = 1.0 / (1.0 + math.exp(scale * (best_score - midpoint)))
        
        # --- Factor 2: Score Consistency (30% weight) ---
        # Lower standard deviation = more consistent = higher confidence
        if len(top_scores) > 1:
            score_std = float(np.std(top_scores))
            # Normalize std to 0-1 range (assuming std usually < 0.5 for good results)
            consistency_score = 1.0 / (1.0 + score_std * 2.0)
        else:
            consistency_score = 1.0  # Single result = perfect consistency
        
        # --- Factor 3: Coverage (10% weight) ---
        # Having multiple relevant docs increases confidence
        coverage_score = min(len(docs_with_scores) / top_n, 1.0)
        
        # --- Combined Confidence ---
        final_confidence = (
            best_confidence * 0.6 + 
            consistency_score * 0.3 + 
            coverage_score * 0.1
        )
        
        return max(0.0, min(final_confidence, 1.0))
    
    def _calculate_single_score_confidence(self, similarity_score: float) -> float:
        """
        Legacy method for single score confidence calculation.
        Used for backward compatibility.
        
        Args:
            similarity_score: The distance score from the vector search

        Returns:
            A confidence score between 0 and 1
        """
        import math
        
        midpoint = 1.5
        scale = 5.0
        confidence = 1.0 / (1.0 + math.exp(scale * (similarity_score - midpoint)))
        
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
        is_public: bool = False,
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
            is_public: Flag indicating if the contribution is public-facing metadata.

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
                            doc.metadata["is_public"] = is_public
                            doc.metadata["hash_id"] = hash_id  # Use hash_id as common identifier
                            doc.metadata["id"] = hash_id       # Keep id for backward compatibility
                        
                        file_docs.extend(pdf_docs)
                        processed_file = True
                
                # Process DOCX file
                elif file_ext == 'docx':
                    file_type = 'docx'
                    # Use the document processor to process the DOCX file
                    docx_docs = self.document_processor.process_docx(uploaded_file_path)
                    if docx_docs:
                        # Add custom metadata to the DOCX documents
                        for doc in docx_docs:
                            doc.metadata["source"] = os.path.basename(uploaded_file_path)
                            doc.metadata["title"] = title
                            doc.metadata["meta_tags"] = ",".join(meta_tags)
                            doc.metadata["author_name"] = author_name if author_name else "Unknown"
                            doc.metadata["additional_references"] = additional_references if additional_references else "None"
                            doc.metadata["submission_timestamp"] = submitted_at
                            doc.metadata["entry_type"] = "user_contribution_docx"
                            doc.metadata["is_public"] = is_public
                            doc.metadata["hash_id"] = hash_id  # Use hash_id as common identifier
                            doc.metadata["id"] = hash_id       # Keep id for backward compatibility
                        
                        file_docs.extend(docx_docs)
                        processed_file = True
                        logging.info(f"Processed DOCX file with {len(docx_docs)} document chunks")
                
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
                            doc.metadata["is_public"] = is_public
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
                "id": hash_id,        # Keep id for backward compatibility
                "is_public": is_public
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
            db_document["is_public"] = is_public
            
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
                "db_id": db_id,
                "is_public": is_public
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
    

    async def expand_query_with_context(
        self, 
        query: str, 
        conversation_history: List[Dict[str, str]] = None,
        max_history: int = 4
    ) -> Dict[str, Any]:
        """Rewrite query based on conversation context and expand it for better search results.
        
        This method combines contextual query rewriting and query expansion:
        1. If conversation history exists, rewrites the query to be self-contained
        2. Expands the query (original or rewritten) into multiple variations
        3. Returns all queries for comprehensive search
        
        Args:
            query: The original query string
            conversation_history: List of conversation messages with 'role' and 'content' keys
            max_history: Maximum number of previous messages to consider (default: 4 = 2 exchanges)
            
        Returns:
            A dictionary containing:
                - original_query: The original user query
                - rewritten_query: Query rewritten with context (same as original if no history)
                - expanded_queries: List of expanded query variations
                - all_queries: Combined list of all queries for search
        """
        try:
            from openai import AsyncOpenAI
            import json
            
            # Initialize OpenAI client
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            
            rewritten_query = query
            
            # Step 1: Rewrite query with conversation context if history exists
            if conversation_history:
                # Take only the most recent messages
                recent_history = conversation_history[-max_history:] if len(conversation_history) > max_history else conversation_history
                
                # Truncate very long messages to avoid token limit
                truncated_history = []
                for msg in recent_history:
                    content = msg.get('content', '')
                    if len(content) > 300:
                        content = content[:300] + "..."
                    truncated_history.append({
                        'role': msg['role'],
                        'content': content
                    })
                
                # Build context window
                recent_context = "\n".join(
                    [f"{msg['role'].capitalize()}: {msg['content']}" for msg in truncated_history]
                )
                
                # Create prompt for contextual rewriting and expansion
                prompt = f"""با توجه به گفتگوی زیر، دو کار انجام بده:
1. پرسش جدید را بازنویسی کن تا بدون نیاز به متن‌های قبلی قابل جست‌وجو باشد
2. سه نسخه جایگزین از پرسش بازنویسی‌شده ایجاد کن با کلمات مختلف و مترادف‌ها

فقط از اطلاعات خود گفتگو استفاده کن. اگر پرسش به شرکت یا خدماتش اشاره دارد بدون ذکر نام، "پرشین وی" و "Persian Way" را به پرسش‌ها اضافه کن.
---
گفتگو:
{recent_context}

پرسش جدید کاربر:
{query}
---
فقط یک JSON برگردان با این فرمت:
{{
    "rewritten_query": "پرسش بازنویسی‌شده",
    "expanded_queries": [
        "نسخه جایگزین اول",
        "نسخه جایگزین دوم",
        "نسخه جایگزین سوم"
    ]
}}
"""
            else:
                # No conversation history - just expand the original query
                prompt = f"""برای پرسش زیر، سه نسخه جایگزین ایجاد کن که همان منظور را با کلمات و عبارات متفاوت بیان کنند.
روی گسترش مفاهیم، اضافه کردن مترادف‌ها و در نظر گرفتن اصطلاحات فارسی و انگلیسی تمرکز کن.
اگر پرسش به شرکت یا خدماتش اشاره دارد بدون ذکر نام، "پرشین وی" و "Persian Way" را به هر پرسش اضافه کن.

پرسش: {query}

فقط یک JSON برگردان با این فرمت:
{{
    "rewritten_query": "{query}",
    "expanded_queries": [
        "نسخه جایگزین اول",
        "نسخه جایگزین دوم",
        "نسخه جایگزین سوم"
    ]
}}
"""
            
            # Call gpt-4o-mini for both rewriting and expansion
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "تو دستیار هوشمندی هستی که پرسش‌ها را بازنویسی و گسترش می‌دهی برای بهبود نتایج جست‌وجو."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=800
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            rewritten_query = result.get("rewritten_query", query)
            expanded_queries = result.get("expanded_queries", [])
            
            # Combine all queries for search (rewritten + expanded)
            # Remove duplicates while preserving order
            all_queries = [rewritten_query]
            for eq in expanded_queries:
                if eq and eq not in all_queries:
                    all_queries.append(eq)
            
            logging.info(f"[Query Expansion] Original: '{query[:50]}...'")
            if rewritten_query != query:
                logging.info(f"[Query Expansion] Rewritten: '{rewritten_query[:50]}...'")
            logging.info(f"[Query Expansion] Generated {len(expanded_queries)} expanded variations")
            
            return {
                "original_query": query,
                "rewritten_query": rewritten_query,
                "expanded_queries": expanded_queries,
                "all_queries": all_queries
            }
            
        except Exception as e:
            logging.error(f"Error in expand_query_with_context: {str(e)}")
            # Return just the original query if there's an error
            return {
                "original_query": query,
                "rewritten_query": query,
                "expanded_queries": [],
                "all_queries": [query]
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

    async def query_knowledge_base(self, query: str, conversation_history: List = None, is_public: bool = False) -> Dict[str, Any]:
        """Query the knowledge base with a question using improved retrieval strategy.
        
        Improvements:
        - Weighted multi-query search (original query gets higher weight)
        - Similarity threshold filtering
        - MMR for diversity
        - Multi-factor confidence calculation
        
        Args:
            query: The question to ask
            conversation_history: Previous conversation messages for context
            is_public: When True, restricts retrieval to documents tagged with public metadata
            
        Returns:
            A dictionary with the answer, confidence score, and source information
        """
        try:
            # Log the incoming query for debugging
            logging.info(f"[KB Query] Original query: '{query[:100]}...'")
            
            # Extract conversation history
            extracted_history = self._extract_conversation_history(conversation_history)
            logging.debug(f"[KB Query] Extracted {len(extracted_history)} messages from conversation history")
            
            # Filter out the current query from history to prevent circular context
            filtered_history = []
            if extracted_history:
                for msg in extracted_history:
                    # Skip if this message matches the current query
                    if msg.get('role') == 'user' and msg.get('content', '').strip() == query.strip():
                        logging.info(f"[KB Query] Skipping current query from history: '{query[:50]}...'")
                        continue
                    filtered_history.append(msg)
                
                logging.debug(f"[KB Query] After filtering: {len(filtered_history)} messages remain for context")
            
            # Use the combined method to rewrite query with context and expand it
            # This handles both contextual rewriting and query expansion in one step
            query_expansion_result = await self.expand_query_with_context(
                query=query,
                conversation_history=filtered_history if filtered_history else None
            )
            
            # Get all queries for search
            all_queries = query_expansion_result["all_queries"]
            rewritten_query = query_expansion_result["rewritten_query"]
            
            # Log the query transformation
            if rewritten_query != query:
                logging.info(f"[KB Query] Query rewritten with context: '{query[:50]}...' -> '{rewritten_query[:50]}...'")
            
            # First, check if we have an exact or semantically similar match in our QA database
            vector_store = self.document_processor.get_vector_store()
            
            # If vector store is not available, raise an exception to be handled by chat service
            if vector_store is None:
                raise RuntimeError("Vector store not available. OpenAI embeddings may not be properly configured. Please check your OPENAI_API_KEY and ensure the vector database is initialized.")
                

            rag_settings = await self.config_service.get_rag_settings()
            
            # ===== IMPROVEMENT 1: Weighted Multi-Query Search with MMR =====
            logging.info(f"[KB Query] Performing weighted multi-query search with {len(all_queries)} queries")
            
            # Prepare base search kwargs
            base_search_kwargs = {}
            if is_public:
                base_search_kwargs["filter"] = {"is_public": True}
            
            # Calculate fetch_k for MMR (fetch more candidates, then apply diversity)
            fetch_k = rag_settings.top_k_results * rag_settings.fetch_k_multiplier
            
            # Weighted search across all queries
            weighted_docs_with_scores = []
            for idx, search_query in enumerate(all_queries):
                if not search_query.strip():
                    continue
                
                # Assign weight: original/rewritten query gets higher weight
                # First query is the rewritten one (most important)
                if idx == 0:
                    weight = rag_settings.original_query_weight
                    query_type = "rewritten"
                else:
                    weight = rag_settings.expanded_query_weight
                    query_type = f"expanded_{idx}"
                
                logging.debug(f"[KB Query] Searching with {query_type} query (weight={weight:.2f}): '{search_query[:50]}...'")
                
                try:
                    # Use MMR search for diversity (better than plain similarity)
                    # Note: ChromaDB doesn't have max_marginal_relevance_search_with_score,
                    # so we use MMR to get diverse docs, then calculate scores separately
                    
                    # First, get diverse documents using MMR
                    mmr_kwargs = {
                        **base_search_kwargs,
                        "k": rag_settings.top_k_results,
                        "fetch_k": fetch_k,
                        "lambda_mult": rag_settings.mmr_diversity_score
                    }
                    
                    # MMR search returns documents without scores
                    mmr_docs = vector_store.max_marginal_relevance_search(
                        search_query,
                        **mmr_kwargs
                    )
                    
                    if not mmr_docs:
                        logging.debug(f"[KB Query] No documents found via MMR for {query_type} query")
                        continue
                    
                    # Now calculate scores for MMR documents using similarity search
                    # We need to get scores for the documents returned by MMR
                    # Strategy: Get more results with similarity search, then match with MMR results
                    similarity_docs_with_scores = vector_store.similarity_search_with_score(
                        search_query,
                        k=fetch_k,  # Get enough candidates to match MMR results
                        **base_search_kwargs
                    )
                    
                    # Create a mapping of document to score for quick lookup
                    # Use full content hash + metadata for precise matching
                    doc_to_score = {}
                    for sim_doc, sim_score in similarity_docs_with_scores:
                        # Use a unique key based on content hash and metadata for precise matching
                        content_hash = hashlib.md5(sim_doc.page_content.encode('utf-8')).hexdigest()[:8]
                        doc_key = (
                            content_hash,
                            sim_doc.metadata.get("source", ""),
                            sim_doc.metadata.get("page", 0)
                        )
                        # Keep the best (lowest) score if duplicate
                        if doc_key not in doc_to_score or sim_score < doc_to_score[doc_key]:
                            doc_to_score[doc_key] = sim_score
                    
                    # Match MMR documents with their scores
                    docs_with_scores = []
                    for mmr_doc in mmr_docs:
                        mmr_content_hash = hashlib.md5(mmr_doc.page_content.encode('utf-8')).hexdigest()[:8]
                        mmr_key = (
                            mmr_content_hash,
                            mmr_doc.metadata.get("source", ""),
                            mmr_doc.metadata.get("page", 0)
                        )
                        
                        if mmr_key in doc_to_score:
                            score = doc_to_score[mmr_key]
                            docs_with_scores.append((mmr_doc, score))
                        else:
                            # If not found in similarity results, try to find by content similarity
                            # This can happen if MMR returns a doc that's not in top fetch_k similarity results
                            # In this case, we'll use a conservative score (threshold)
                            logging.debug(f"[KB Query] MMR doc not in similarity top {fetch_k}, using threshold score")
                            docs_with_scores.append((mmr_doc, rag_settings.similarity_threshold))
                    
                    # Apply weight to scores (divide by weight to boost - lower score is better)
                    for doc, score in docs_with_scores:
                        weighted_score = score / weight
                        weighted_docs_with_scores.append((doc, weighted_score, search_query, query_type))
                        
                    logging.debug(f"[KB Query] Found {len(docs_with_scores)} docs via MMR for {query_type} query")
                    
                except Exception as e:
                    logging.warning(f"[KB Query] MMR search failed for query '{search_query[:50]}...': {str(e)}")
                    # Fallback to regular similarity search
                    try:
                        docs_with_scores = vector_store.similarity_search_with_score(
                            search_query,
                            k=rag_settings.top_k_results,
                            **base_search_kwargs
                        )
                        for doc, score in docs_with_scores:
                            weighted_score = score / weight
                            weighted_docs_with_scores.append((doc, weighted_score, search_query, query_type))
                        logging.debug(f"[KB Query] Fallback to similarity search: found {len(docs_with_scores)} docs")
                    except Exception as e2:
                        logging.error(f"[KB Query] Fallback search also failed: {str(e2)}")
                        continue
            
            # ===== IMPROVEMENT 2: Similarity Threshold Filtering =====
            # Filter out documents with scores above threshold (higher score = less similar in L2 distance)
            filtered_docs = [
                (doc, score, query, qtype) 
                for doc, score, query, qtype in weighted_docs_with_scores
                if score <= rag_settings.similarity_threshold
            ]
            
            if filtered_docs:
                removed_count = len(weighted_docs_with_scores) - len(filtered_docs)
                if removed_count > 0:
                    logging.info(f"[KB Query] Filtered out {removed_count} documents below similarity threshold ({rag_settings.similarity_threshold})")
            else:
                # If all filtered out, keep best ones anyway (graceful degradation)
                logging.warning(f"[KB Query] All documents below threshold, keeping top {rag_settings.top_k_results} anyway")
                filtered_docs = weighted_docs_with_scores
            
            # ===== IMPROVEMENT 3: Deduplication with Source Tracking =====
            # Remove duplicates while preserving the best score and tracking source query
            seen_docs = {}
            for doc, score, source_query, query_type in sorted(filtered_docs, key=lambda x: x[1]):
                doc_key = (doc.page_content, doc.metadata.get("source", ""), doc.metadata.get("page", 0))
                
                # Keep only the best score for each unique document
                if doc_key not in seen_docs or score < seen_docs[doc_key][1]:
                    seen_docs[doc_key] = (doc, score, source_query, query_type)
            
            # Convert back to list and sort by score
            unique_docs_with_scores = sorted(
                [(doc, score) for doc, score, _, _ in seen_docs.values()],
                key=lambda x: x[1]
            )
            
            # Take top K results
            docs_with_scores = unique_docs_with_scores[:rag_settings.top_k_results]
            
            logging.info(f"[KB Query] Final retrieval: {len(docs_with_scores)} unique documents (from {len(weighted_docs_with_scores)} initial candidates)")
            if docs_with_scores:
                logging.debug(f"[KB Query] Score range: {docs_with_scores[0][1]:.4f} (best) to {docs_with_scores[-1][1]:.4f} (worst)")
            
            # Use the QA chain with retrieved documents
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
            
            # Normalize documents before passing to QA chain
            normalized_docs = self._normalize_documents_for_context(docs)
            
            # Log normalized document info for debugging
            logging.info(f"Passing {len(normalized_docs)} normalized documents to QA chain")
            for i, doc in enumerate(normalized_docs):
                logging.debug(f"Doc {i+1}: content_length={len(doc.page_content)}, metadata_keys={list(doc.metadata.keys())}")
            
            # Extract page content to pass as context
            context_snippets = [doc.page_content for doc in normalized_docs]
            logging.debug(f"Context snippet lengths: {[len(snippet) for snippet in context_snippets]}")
            
    
                # knowledge base information
            full_context = "\n\n".join(f"سند {i+1}:\n{snippet}" for i, snippet in enumerate(context_snippets))
            
            # Log context details for debugging
            logging.info(f"Full context length: {len(full_context)} chars, ~{len(full_context)//4} tokens")
            logging.debug(f"First 500 chars of context: {full_context[:500]}")
            
            result = qa_chain.invoke({"input": rewritten_query, "context": full_context})
            answer = result.get("answer") or result.get("result")
            if answer is None:
                raise KeyError("answer")
            source_docs = result.get("context") or result.get("source_documents") or []
            
            # ===== IMPROVEMENT 4: Multi-factor Confidence Calculation =====
            # Calculate confidence based on multiple factors:
            # - Best document score (60% weight)
            # - Score consistency across top documents (30% weight)
            # - Number of relevant documents found (10% weight)
            if docs_with_scores:
                confidence = self._calculate_confidence_score(docs_with_scores, top_n=3)
                logging.info(f"[KB Query] Multi-factor confidence score: {confidence:.4f}")
                
                # Log individual document scores for debugging
                for i, (doc, score) in enumerate(docs_with_scores[:3]):
                    logging.debug(f"[KB Query] Top doc {i+1} score: {score:.4f}")
            else:
                confidence = 0.0
                logging.warning("[KB Query] No documents found, confidence = 0.0")
            
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