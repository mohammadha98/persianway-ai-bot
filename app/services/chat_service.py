from typing import Dict, List, Optional
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from app.core.config import settings
from app.schemas.chat import ChatMessage


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
    
    async def process_message(self, user_id: str, message: str) -> str:
        """Process a user message and get a response from the AI.
        
        Args:
            user_id: Unique identifier for the user session
            message: The message from the user
            
        Returns:
            The AI model's response
        """
        try:
            # Get or create a conversation chain for this user
            conversation = self._get_or_create_session(user_id)
            
            # Get a response from the model
            response = conversation.predict(input=message)
            
            return response
        except Exception as e:
            # Handle errors from the OpenAI API
            error_msg = f"Error processing message: {str(e)}"
            raise Exception(error_msg)
    
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