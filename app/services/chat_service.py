from typing import Dict, List, Optional, Any
import re
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from app.core.config import settings
from app.schemas.chat import ChatMessage
from app.services.knowledge_base import get_knowledge_base_service


class ChatService:
    """Service for managing chat interactions with OpenAI models.
    
    This service is responsible for handling chat sessions, maintaining conversation
    history, and interacting with the OpenAI API via LangChain.
    """
    
    def __init__(self):
        """Initialize the chat service."""
        self._sessions: Dict[str, ConversationChain] = {}
        self._memories: Dict[str, ConversationBufferMemory] = {}
        
        # Validate OpenAI API key
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in environment variables or .env file")
    
    def _get_or_create_session(self, user_id: str) -> ConversationChain:
        """Get an existing chat session or create a new one.
        
        Args:
            user_id: Unique identifier for the user session
            
        Returns:
            A LangChain ConversationChain for the user
        """
        if user_id not in self._sessions:
            # Create a new memory for this user
            memory = ConversationBufferMemory()
            self._memories[user_id] = memory
            
            # Create a new chat model with the configured settings
            llm = ChatOpenAI(
                openai_api_key=settings.OPENAI_API_KEY,
                model_name=settings.OPENAI_MODEL_NAME,
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS
            )
            
            # Create a conversation chain with the memory
            self._sessions[user_id] = ConversationChain(
                llm=llm,
                memory=memory,
                verbose=False
            )
        
        return self._sessions[user_id]
    
    def _is_topic_related_to_domain(self, query: str) -> bool:
        """Check if the query is related to the knowledge base domain.
        
        Args:
            query: The user's question
            
        Returns:
            True if the query is related to the domain, False otherwise
        """
        query_lower = query.lower()
        
        # Define domain-related keywords (expand based on your knowledge base content)
        domain_keywords = [
            # Health and beauty related
            'پوست', 'صورت', 'ماسک', 'کرم', 'زیبایی', 'سلامت', 'درمان', 'دارو', 'بیمار',
            'معده', 'شکم', 'ماساژ', 'روغن', 'مالت', 'خرما', 'شیر', 'کودک', 'نوزاد', 'مو', 'ریزش',
            'skin', 'face', 'mask', 'cream', 'beauty', 'health', 'treatment', 'medicine', 'hair',
            # Agriculture related (if still relevant)
            'کود', 'کشاورزی', 'خاک', 'کاشت', 'برداشت', 'آفت', 'بیماری', 'گیاه', 'محصول',
            'آبیاری', 'بذر', 'نهال', 'درخت', 'میوه', 'سبزی', 'غلات', 'دام', 'طیور',
            'fertilizer', 'agriculture', 'soil', 'plant', 'crop', 'farming', 'irrigation'
        ]
        
        # Unrelated topics that should be referred to humans
        unrelated_keywords = [
            # Technology
            'کامپیوتر', 'نرم افزار', 'اپلیکیشن', 'وب سایت', 'برنامه نویسی', 'شبکه', 'اینترنت',
            'computer', 'software', 'application', 'website', 'programming', 'network', 'internet',
            # Finance
            'پول', 'بانک', 'سرمایه گذاری', 'بورس', 'اقتصاد', 'مالی', 'حسابداری',
            'money', 'bank', 'investment', 'stock', 'economy', 'financial', 'accounting',
            # Politics/History
            'سیاست', 'انتخابات', 'دولت', 'تاریخ', 'جنگ', 'مکتب', 'دیدگاه', 'ایدئولوژی', 'فلسفه سیاسی',
            'politics', 'election', 'government', 'history', 'war', 'ideology', 'political view', 'philosophy'
        ]
        
        # Check for unrelated topics first
        if any(keyword in query_lower for keyword in unrelated_keywords):
            return False
            
        # Check for domain-related topics
        if any(keyword in query_lower for keyword in domain_keywords):
            return True
            
        # For ambiguous queries, be more conservative - require explicit domain match
        return False

    async def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """Process a user message using a hybrid approach.

        This service implements a three-tier approach:
        1. Check knowledge base for high-confidence answers
        2. Use general knowledge for domain-related topics with low KB confidence
        3. Refer unrelated topics to humans

        Args:
            user_id: Unique identifier for the user session
            message: The message from the user

        Returns:
            A dictionary representing the ChatResponse schema.
        """
        KB_CONFIDENCE_THRESHOLD = 0.6  # Lowered threshold for better coverage
        HUMAN_REFERRAL_MESSAGE = settings.HUMAN_REFERRAL_MESSAGE

        query_analysis = {
            "confidence_score": 0.0,
            "knowledge_source": "none",
            "requires_human_referral": False,
            "reasoning": ""
        }
        response_parameters = {
            "model": settings.OPENAI_MODEL_NAME,
            "temperature": settings.OPENAI_TEMPERATURE,
            "max_tokens": settings.OPENAI_MAX_TOKENS
        }
        answer = ""

        try:
            # First, check if the topic is related to our domain
            is_domain_related = self._is_topic_related_to_domain(message)
            
            if not is_domain_related:
                # Unrelated topic - refer to human
                answer = HUMAN_REFERRAL_MESSAGE
                query_analysis["confidence_score"] = 0.0
                query_analysis["knowledge_source"] = "none"
                query_analysis["requires_human_referral"] = True
                query_analysis["reasoning"] = "Query is outside our domain expertise and requires human specialist attention."
            else:
                # Domain-related topic - try knowledge base first
                kb_service = get_knowledge_base_service()
                kb_result = await kb_service.query_knowledge_base(message)
                
                kb_confidence = kb_result.get("confidence_score", 0) if kb_result else 0

                if kb_confidence >= KB_CONFIDENCE_THRESHOLD:
                    # High confidence answer from knowledge base
                    answer = kb_result["answer"]
                    query_analysis["confidence_score"] = kb_confidence
                    query_analysis["knowledge_source"] = kb_result.get("source_type", "knowledge_base")
                    query_analysis["requires_human_referral"] = False
                    query_analysis["reasoning"] = "High confidence answer found in knowledge base."
                    response_parameters["temperature"] = 0.1  # Low temperature for factual answers
                else:
                    # Low KB confidence but domain-related - use general knowledge with context
                    conversation = self._get_or_create_session(user_id)
                    
                    # Create a context-aware prompt
                    context_prompt = f"""
{settings.SYSTEM_PROMPT}

اگر سوال در حوزه تخصص شما نیست، لطفاً به کاربر بگویید که این سوال نیاز به بررسی توسط کارشناس دارد.
اگر سوال مرتبط است، پاسخ مفیدی ارائه دهید.

سوال کاربر: {message}
                    """
                    
                    # Get response using general knowledge
                    response = conversation.predict(input=context_prompt)
                    
                    # Check if the model indicated it needs human referral
                    referral_indicators = [
                        "نیاز به بررسی توسط کارشناس",
                        "به کارشناس مراجعه کنید",
                        "خارج از حوزه تخصص",
                        "نمی‌توانم پاسخ دهم"
                    ]
                    
                    if any(indicator in response for indicator in referral_indicators):
                        answer = HUMAN_REFERRAL_MESSAGE
                        query_analysis["requires_human_referral"] = True
                        query_analysis["reasoning"] = "Model determined the query requires specialist attention."
                    else:
                        answer = response
                        query_analysis["confidence_score"] = 0.7  # Medium confidence for general knowledge
                        query_analysis["knowledge_source"] = "general_knowledge"
                        query_analysis["requires_human_referral"] = False
                        query_analysis["reasoning"] = "Answer provided using general knowledge with domain context."
                        response_parameters["temperature"] = 0.3  # Moderate temperature for general knowledge

            # Add the interaction to the conversation history
            conversation = self._get_or_create_session(user_id)
            conversation.memory.chat_memory.add_user_message(message)
            conversation.memory.chat_memory.add_ai_message(answer)

            # Construct the final response dictionary
            return {
                "query_analysis": query_analysis,
                "response_parameters": response_parameters,
                "answer": answer
            }

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            # Fallback to human referral on any processing error
            query_analysis["requires_human_referral"] = True
            query_analysis["reasoning"] = f"An internal error occurred: {error_msg}"
            query_analysis["confidence_score"] = 0.0
            query_analysis["knowledge_source"] = "none"
            return {
                "query_analysis": query_analysis,
                "response_parameters": response_parameters,
                "answer": HUMAN_REFERRAL_MESSAGE
            }
    
    def get_conversation_history(self, user_id: str) -> Optional[List[ChatMessage]]:
        """Get the conversation history for a user.
        
        Args:
            user_id: Unique identifier for the user session
            
        Returns:
            A list of ChatMessage objects or None if no history exists
        """
        if user_id not in self._memories:
            return None
        
        memory = self._memories[user_id]
        history = []
        
        # Convert LangChain memory to our ChatMessage schema
        for message in memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history.append(ChatMessage(role="user", content=message.content))
            elif isinstance(message, AIMessage):
                history.append(ChatMessage(role="assistant", content=message.content))
            elif isinstance(message, SystemMessage):
                history.append(ChatMessage(role="system", content=message.content))
        
        return history


# Singleton instance
_chat_service = None


def get_chat_service() -> ChatService:
    """Get the chat service instance.
    
    Returns:
        A singleton instance of the ChatService
    """
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service