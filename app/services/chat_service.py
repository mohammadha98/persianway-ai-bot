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
    
    async def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """Process a user message and get a response according to the advanced response system.

        This service will first check the knowledge base for an answer. If the confidence
        is low, it will refer the query to a human instead of using a general model.

        Args:
            user_id: Unique identifier for the user session
            message: The message from the user

        Returns:
            A dictionary representing the ChatResponse schema.
        """
        CONFIDENCE_THRESHOLD = 0.75
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
            kb_service = get_knowledge_base_service()
            kb_result = await kb_service.query_knowledge_base(message)
            
            kb_confidence = kb_result.get("confidence_score", 0) if kb_result else 0

            if kb_confidence >= CONFIDENCE_THRESHOLD:
                # High confidence answer found in the knowledge base
                answer = kb_result["answer"]
                query_analysis["confidence_score"] = kb_confidence
                query_analysis["knowledge_source"] = kb_result.get("source_type", "knowledge_base")
                query_analysis["requires_human_referral"] = False
                query_analysis["reasoning"] = "High confidence answer found in knowledge base."
                response_parameters["temperature"] = 0.1  # Use low temperature for factual KB answers
            else:
                # Low confidence from KB, indicating the query is outside the domain.
                # Refer to a human specialist as per system design.
                answer = HUMAN_REFERRAL_MESSAGE
                query_analysis["confidence_score"] = kb_confidence
                query_analysis["knowledge_source"] = "none"
                query_analysis["requires_human_referral"] = True
                query_analysis["reasoning"] = "Query is outside the scope of the knowledge base and requires human attention."

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