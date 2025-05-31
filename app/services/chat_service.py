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
    
    def _is_agriculture_query(self, message: str) -> bool:
        """Detect if the message is an agriculture-related query.

        Args:
            message: The message from the user

        Returns:
            True if the message is likely about agriculture, False otherwise
        """
        # Enhanced list of agriculture-related keywords (English and Persian)
        # This is a simplified approach. A more robust solution might involve
        # a text classification model or more sophisticated NLP techniques.
        agriculture_keywords = [
            # English
            'agriculture', 'soil', 'fertilizer', 'irrigation', 'crop', 'farming', 'horticulture',
            'seed', 'planting', 'harvest', 'pest', 'pesticide', 'plant', 'tree',
            'fruit', 'vegetable', 'weed', 'livestock', 'poultry', 'farm', 'garden',
            'greenhouse', 'breeding', 'seedling', 'grafting', 'nutrition', 'deficiency',
            'root', 'leaf', 'stem', 'flower', 'grain', 'disease', 'fungus', 'bacteria',
            'virus', 'insect', 'mite', 'nematode', 'herbicide', 'fungicide', 'insecticide',
            'acaricide', 'chemical fertilizer', 'organic fertilizer', 'manure', 'biofertilizer',
            'ph', 'ec', 'npk', 'nitrogen', 'phosphorus', 'potassium', 'calcium', 'magnesium',
            'iron', 'zinc', 'copper', 'manganese', 'boron', 'molybdenum', 'sulfur', 'chlorine',
            'cultivation', 'tillage', 'soil science', 'plant nutrition', 'crop management',
            'pest control', 'agricultural sustainability', 'water management', 'agronomy',
            # Persian (similar to previous list, can be expanded)
            'کشاورزی', 'خاک', 'کود', 'آبیاری', 'محصول', 'زراعت', 'باغبانی',
            'بذر', 'کاشت', 'داشت', 'برداشت', 'آفت', 'سم', 'گیاه', 'درخت',
            'میوه', 'سبزی', 'علف', 'هرز', 'دام', 'طیور', 'مزرعه', 'باغ',
            'گلخانه', 'اصلاح', 'نهال', 'پیوند', 'تغذیه', 'کمبود', 'عناصر',
            'ریشه', 'برگ', 'ساقه', 'گل', 'دانه', 'بیماری', 'قارچ', 'باکتری',
            'ویروس', 'حشره', 'کنه', 'نماتد', 'علف‌کش', 'قارچ‌کش', 'حشره‌کش',
            'کنه‌کش', 'کود شیمیایی', 'کود آلی', 'کود حیوانی', 'کود زیستی',
            'پی اچ', 'ای سی', 'ان پی کا', 'ازت', 'فسفر', 'پتاسیم', 'کلسیم', 'منیزیم',
            'آهن', 'روی', 'مس', 'منگنز', 'بور', 'مولیبدن', 'گوگرد', 'کلر'
        ]

        message_lower = message.lower()
        for keyword in agriculture_keywords:
            if keyword in message_lower:
                return True
        return False
    
    def _calculate_general_confidence(self, response: str) -> float:
        """Calculate a general confidence score for non-knowledge base responses.
        
        Args:
            response: The model's response
            
        Returns:
            A confidence score between 0 and 1
        """
        # For general chat responses, we'll use a default high confidence
        # This could be improved with more sophisticated methods in the future
        # such as analyzing response certainty markers or model logprobs
        return 0.9
    
    async def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """Process a user message and get a response according to the advanced response system.

        Args:
            user_id: Unique identifier for the user session
            message: The message from the user

        Returns:
            A dictionary representing the ChatResponse schema.
        """
        CONFIDENCE_THRESHOLD = 0.75
        HUMAN_REFERRAL_MESSAGE = "I need to refer this question to a human specialist. This question appears to be outside my knowledge domain or requires specialized expertise beyond my current capabilities."

        query_analysis = {
            "is_agriculture_related": False,
            "confidence_score": 0.0,
            "knowledge_source": "none",
            "requires_human_referral": False,
            "reasoning": ""
        }
        response_parameters = {
            "model": settings.OPENAI_MODEL_NAME,
            "temperature": settings.OPENAI_TEMPERATURE, # Default temperature
            "max_tokens": settings.OPENAI_MAX_TOKENS
        }
        answer = ""

        try:
            is_agri_query = self._is_agriculture_query(message)
            query_analysis["is_agriculture_related"] = is_agri_query

            if not is_agri_query:
                query_analysis["requires_human_referral"] = True
                query_analysis["reasoning"] = "Query is not agriculture-related."
                query_analysis["confidence_score"] = 0.0 # No confidence as it's out of domain
                answer = HUMAN_REFERRAL_MESSAGE
                response_parameters["temperature"] = 0.3 # Default for non-agri, though referred
            else:
                # Query is agriculture-related, proceed with knowledge base or general model
                kb_service = get_knowledge_base_service()
                kb_result = await kb_service.query_knowledge_base(message)

                if kb_result and kb_result.get("answer") and kb_result.get("confidence_score", 0) >= CONFIDENCE_THRESHOLD:
                    # High confidence answer from knowledge base
                    answer = kb_result["answer"]
                    query_analysis["confidence_score"] = kb_result["confidence_score"]
                    query_analysis["knowledge_source"] = kb_result.get("source_type", "knowledge_base")
                    query_analysis["requires_human_referral"] = False
                    query_analysis["reasoning"] = "High confidence answer found in knowledge base."
                    response_parameters["temperature"] = 0.1 # Factual agricultural knowledge
                    
                    # Add to conversation history
                    conversation = self._get_or_create_session(user_id)
                    conversation.memory.chat_memory.add_user_message(message)
                    conversation.memory.chat_memory.add_ai_message(answer)

                else:
                    # Low confidence from KB or no KB result, or general agri question
                    # Use general model, but assess confidence
                    conversation = self._get_or_create_session(user_id)
                    # Adjust temperature for explanatory content if not strictly factual from KB
                    current_llm_temp = 0.3 
                    conversation.llm.temperature = current_llm_temp # Update LLM temp for this call
                    response_parameters["temperature"] = current_llm_temp

                    general_response = conversation.predict(input=message)
                    general_confidence = self._calculate_general_confidence(general_response) # This might need refinement

                    if general_confidence >= CONFIDENCE_THRESHOLD:
                        answer = general_response
                        query_analysis["confidence_score"] = general_confidence
                        query_analysis["knowledge_source"] = "general_knowledge"
                        query_analysis["requires_human_referral"] = False
                        query_analysis["reasoning"] = "Answer generated from general knowledge model with sufficient confidence."
                    else:
                        answer = HUMAN_REFERRAL_MESSAGE
                        query_analysis["confidence_score"] = general_confidence # Report the lower confidence
                        query_analysis["knowledge_source"] = "general_knowledge" # Still attempted general knowledge
                        query_analysis["requires_human_referral"] = True
                        query_analysis["reasoning"] = "Query is agriculture-related, but confidence in general knowledge response is low."

            # Construct the final response dictionary
            return {
                "query_analysis": query_analysis,
                "response_parameters": response_parameters,
                "answer": answer
            }

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            # Fallback to human referral on error
            query_analysis["requires_human_referral"] = True
            query_analysis["reasoning"] = f"An internal error occurred: {error_msg}"
            query_analysis["confidence_score"] = 0.0
            query_analysis["knowledge_source"] = "none"
            return {
                "query_analysis": query_analysis,
                "response_parameters": response_parameters, # Use default params
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