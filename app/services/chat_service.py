from logging import Logger
from typing import Dict, List, Optional, Any
import re
from langchain_community.chat_models import ChatOpenAI
import logging
from loguru import logger
from app.services.spell_corrector import get_spell_corrector

async def get_llm(model_name: str = None, temperature: float = None, max_tokens: int = None, top_p: float = None):
    """Initializes and returns the appropriate language model client.
    
    This function selects the API provider based on the PREFERRED_API_PROVIDER setting.
    It handles model name prefixes to ensure compatibility with both providers.
    
    When using OpenRouter, you can specify models from different providers:
    - OpenAI models: "openai/gpt-4", "openai/gpt-3.5-turbo", or just "gpt-4" (prefix added automatically)
    - Google models: "google/gemini-pro"
    - Anthropic models: "anthropic/claude-2"
    - Meta models: "meta-llama/llama-2-70b-chat"
    
    When using OpenAI directly, provider prefixes are automatically removed.
    """
    from app.services.config_service import ConfigService
    
    # Get dynamic configuration
    config_service = ConfigService()
    await config_service._load_config()
    llm_settings = await config_service.get_llm_settings()
    
    # Determine which API provider to use based on configuration
    preferred_provider = llm_settings.preferred_api_provider.lower()
    
    # Check if we have the necessary API keys
    has_openai_key = bool(llm_settings.openai_api_key)
    has_openrouter_key = bool(llm_settings.openrouter_api_key)
    
    # Determine which provider to use based on preference and available keys
    use_openrouter = False
    
    if preferred_provider == "auto":
        # In auto mode, use OpenRouter if available, otherwise fall back to OpenAI
        use_openrouter = has_openrouter_key
    elif preferred_provider == "openrouter":
        # Explicitly use OpenRouter
        if has_openrouter_key:
            use_openrouter = True
        else:
            print("OpenRouter is preferred but API key is not set. Falling back to OpenAI if available.")
            use_openrouter = False
    elif preferred_provider == "openai":
        # Explicitly use OpenAI
        use_openrouter = False
    else:
        logging.warning(f"Unknown provider preference '{preferred_provider}'. Using auto selection.")
        use_openrouter = has_openrouter_key
    
    # Use the selected provider
    if use_openrouter and has_openrouter_key:
        # Using OpenRouter
        print(f"Using OpenRouter API with model: {model_name or llm_settings.default_model}")
        
        # Make sure the model name is properly formatted for OpenRouter
        # OpenRouter model names should include the provider prefix (e.g., google/gemini-pro, anthropic/claude-2)
        selected_model = model_name or llm_settings.default_model
        
        # Ensure the model has a provider prefix
        if '/' not in selected_model:
            # If no provider prefix, assume it's an OpenAI model
            selected_model = f"openai/{selected_model}"
            logging.info(f"Added default 'openai/' prefix to model name: {selected_model}")
        
        # Create the ChatOpenAI instance with OpenRouter configuration
        return ChatOpenAI(
            model_name=selected_model,
            temperature= llm_settings.temperature,
            max_tokens= llm_settings.max_tokens,
            openai_api_key=llm_settings.openrouter_api_key,
            openai_api_base=llm_settings.openrouter_api_base,
        )
    elif has_openai_key:
        # Using OpenAI directly
        logging.info(f"Using OpenAI API with model: {model_name or llm_settings.default_model}")
        
        # For OpenAI, we need to remove any provider prefix if present
        selected_model = model_name or llm_settings.default_model
        if selected_model.startswith("openai/"):
            selected_model = selected_model.replace("openai/", "")
        
        return ChatOpenAI(
            model_name=selected_model,
            temperature=llm_settings.temperature,
            max_tokens=llm_settings.max_tokens,
            openai_api_key=llm_settings.openai_api_key,
        )
    else:
        raise ValueError("Either OPENAI_API_KEY or OPENROUTER_API_KEY must be set")
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from app.core.config import settings
from app.schemas.chat import ChatMessage
from app.services.knowledge_base import get_knowledge_base_service
from app.services.config_service import ConfigService


class ChatService:
    """Service for managing chat interactions with various language models.
    
    This service is responsible for handling chat sessions, maintaining conversation
    history, and interacting with language models via LangChain. It supports:
    
    - OpenAI models directly through the OpenAI API
    - Multiple model providers (OpenAI, Google, Anthropic, Meta, etc.) through OpenRouter
    
    The provider selection is controlled by the PREFERRED_API_PROVIDER setting.
    """
    
    def __init__(self):
        """Initialize the chat service."""
        self._sessions: Dict[str, ConversationChain] = {}
        self._memories: Dict[str, ConversationBufferMemory] = {}
        self.config_service = ConfigService()
        
        # Note: API key validation is now done dynamically in get_llm function
        

    
    async def _get_or_create_session(self, user_id: str, model: str = None, parameters: dict = None) -> ConversationChain:
        """Get an existing chat session or create a new one.
        
        Args:
            user_id: Unique identifier for the user session
            
        Returns:
            A LangChain ConversationChain for the user
        """
        if user_id not in self._sessions:
            # Create a new memory for this user
            memory = ConversationBufferMemory(return_messages=True)
            self._memories[user_id] = memory
            
            # Create a new chat model with the configured settings
            params = parameters or {}
            llm = await get_llm()
            
            # Create a conversation chain with the memory
            self._sessions[user_id] = ConversationChain(
                llm=llm,
                memory=memory,
                verbose=False
            )
            # Clear any existing messages to prevent the system prompt from being added
            self._sessions[user_id].memory.clear()
        
        return self._sessions[user_id]
    
    def _is_topic_related_to_domain(self, query: str) -> bool:
        """Check if the query is related to the knowledge base domain.
        
        Args:
            query: The user's question
            
        Returns:
            True if the query is related to the domain, False otherwise
        """
        query_lower = query.lower()
        
        # Strongly unrelated topics that should be referred to humans
        # These are topics completely outside PersianWay's domain
        strongly_unrelated_keywords = [
            # سیاست و حکومت (Politics & Government)
            'سیاست', 'انتخابات', 'دولت', 'مجلس', 'رئیس جمهور', 'وزیر', 'حزب',
            'سیاستمدار', 'رای', 'کاندیدا', 'کابینه', 'پارلمان', 'قانون', 'قضاوت',
            'دادگاه', 'وکیل', 'قاضی', 'جرم', 'مجازات', 'زندان', 'پلیس',
            'politics', 'election', 'government', 'parliament', 'president', 'minister',
            'party', 'politician', 'vote', 'candidate', 'cabinet', 'law', 'court',
            'lawyer', 'judge', 'crime', 'punishment', 'prison', 'police', 'america',
            'usa', 'iran', 'china', 'russia', 'europe', 'country', 'nation', 'diplomacy',
            
            # ورزش (Sports)
            'فوتبال', 'والیبال', 'بسکتبال', 'تنیس', 'شنا', 'دوچرخه سواری',
            'کوهنوردی', 'اسکی', 'کشتی', 'جودو', 'کاراته', 'تکواندو', 'بوکس',
            'ورزش', 'تیم', 'بازیکن', 'مربی', 'استادیوم', 'مسابقه', 'قهرمانی',
            'المپیک', 'جام جهانی', 'لیگ', 'فینال', 'گل', 'امتیاز',
            'football', 'volleyball', 'basketball', 'tennis', 'swimming', 'cycling',
            'mountaineering', 'skiing', 'wrestling', 'judo', 'karate', 'taekwondo',
            'boxing', 'sport', 'team', 'player', 'coach', 'stadium', 'competition',
            'championship', 'olympics', 'world cup', 'league', 'final', 'goal', 'score',
            
            # سرگرمی و هنر (Entertainment & Arts)
            'فیلم', 'سینما', 'بازیگر', 'کارگردان', 'تلویزیون', 'سریال', 'برنامه',
            'موسیقی', 'خواننده', 'آهنگ', 'کنسرت', 'آلبوم', 'پیانو', 'گیتار',
            'نقاشی', 'مجسمه سازی', 'عکاسی', 'تئاتر', 'رقص', 'باله', 'اپرا',
            'کتاب', 'رمان', 'شعر', 'نویسنده', 'شاعر', 'ادبیات', 'داستان',
            'movie', 'cinema', 'actor', 'director', 'television', 'series', 'program',
            'music', 'singer', 'song', 'concert', 'album', 'instrument', 'piano',
            'guitar', 'painting', 'sculpture', 'photography', 'theater', 'dance',
            'ballet', 'opera', 'book', 'novel', 'poetry', 'writer', 'poet',
            'literature', 'story',
            
            # فناوری و الکترونیک (Technology & Electronics)
            'کامپیوتر', 'لپ تاپ', 'موبایل', 'تبلت', 'نرم افزار', 'برنامه نویسی',
            'اپلیکیشن', 'وب سایت', 'اینترنت', 'شبکه', 'سرور', 'دیتابیس',
            'هوش مصنوعی', 'ربات', 'بلاک چین', 'ارز دیجیتال', 'بیت کوین',
            'گیم', 'بازی', 'کنسول', 'پلی استیشن', 'ایکس باکس', 'نینتندو',
            'computer', 'laptop', 'mobile', 'tablet', 'software', 'programming',
            'application', 'website', 'internet', 'network', 'server', 'database',
            'artificial intelligence', 'robot', 'blockchain', 'cryptocurrency',
            'bitcoin', 'game', 'gaming', 'console', 'playstation', 'xbox', 'nintendo',
            
            # حمل و نقل (Transportation)
            'اتومبیل', 'ماشین', 'موتور', 'دوچرخه', 'قطار', 'هواپیما', 'کشتی',
            'اتوبوس', 'تاکسی', 'مترو', 'ترام', 'کامیون', 'تریلر', 'جرثقیل',
            'بنزین', 'گازوئیل', 'گاز', 'باتری', 'موتور', 'چرخ', 'ترمز',
            'car', 'automobile', 'motorcycle', 'bicycle', 'train', 'airplane',
            'ship', 'bus', 'taxi', 'metro', 'tram', 'truck', 'trailer', 'crane',
            'gasoline', 'diesel', 'gas', 'battery', 'engine', 'wheel', 'brake',
            
            # املاک و مسکن (Real Estate & Housing)
            'خانه', 'آپارتمان', 'ویلا', 'زمین', 'ساختمان', 'اجاره', 'خرید', 'فروش',
            'رهن', 'ودیعه', 'مشاور املاک', 'قیمت مسکن', 'متراژ', 'اتاق',
            'آشپزخانه', 'حمام', 'پارکینگ', 'انباری', 'بالکن', 'حیاط',
            'house', 'apartment', 'villa', 'land', 'building', 'rent', 'buy',
            'sell', 'mortgage', 'deposit', 'real estate agent', 'housing price',
            'area', 'room', 'kitchen', 'bathroom', 'parking', 'storage', 'balcony',
            
            # مالی و بانکی (Finance & Banking)
            'بانک', 'وام', 'سپرده', 'سود', 'بهره', 'چک', 'کارت اعتباری',
            'حساب', 'پول', 'ارز', 'دلار', 'یورو', 'بورس', 'سهام', 'سرمایه گذاری',
            'بیمه', 'مالیات', 'حسابداری', 'اقتصاد', 'تورم', 'رکود',
            'money', 'bank', 'investment', 'stock', 'economy', 'financial', 'accounting',
            'loan', 'deposit', 'profit', 'interest', 'check', 'credit card',
            'account', 'currency', 'dollar', 'euro', 'stock market',
            'shares', 'insurance', 'tax', 'inflation', 'recession',
            
            # آموزش و تحصیل (Education)
            'دانشگاه', 'مدرسه',
            'نمره', 'دیپلم', 'لیسانس', 'فوق لیسانس', 'دکترا',
            'ریاضی', 'فیزیک', 'شیمی', 'زیست شناسی', 'تاریخ', 'جغرافیا',
            'university', 'school', 'class', 'teacher', 'professor', 'student',
            'exam', 'grade', 'certificate', 'diploma', 'bachelor', 'master',
            'phd', 'mathematics', 'physics', 'chemistry', 'biology', 'history',
            'geography'
        ]
        
        # Only check for strongly unrelated topics
        # Return False only if the query contains strongly unrelated keywords
        for keyword in strongly_unrelated_keywords:
            # Use word boundary check to avoid matching substrings within words
            # For example, to avoid matching 'رای' in 'برای'
            # Create a pattern with word boundaries
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, query_lower):
                # Log the keyword that caused the rejection
                logger.info(f"Query rejected due to unrelated keyword: '{keyword}' found in query: '{query}'")
                # You can access this log in your application logs
                return False, keyword
            
        # For all other queries, assume they are related to the domain
        return True, None



    async def generate_conversation_title(self, message: str) -> str:
        """Generate a conversation title based on the user's message.
        
        Args:
            message: The user's message to generate a title from
            
        Returns:
            A concise title for the conversation
        """
        try:
            # Get LLM instance
            llm = await get_llm()
            
            # Create a prompt to generate a concise title
            title_prompt = f"""Based on the following user message, generate a concise and descriptive title (maximum 5-7 words) for this conversation. The title should be in the same language as the user's message.

User message: {message}

Title:"""
            
            # Generate title using the LLM
            response = await llm.ainvoke([HumanMessage(content=title_prompt)])
            
            # Extract and clean the title
            title = response.content.strip()
            
            # Remove quotes if present
            if title.startswith('"') and title.endswith('"'):
                title = title[1:-1]
            if title.startswith("'") and title.endswith("'"):
                title = title[1:-1]
                
            # Limit title length as a safety measure
            if len(title) > 100:
                title = title[:97] + "..."
                
            return title
            
        except Exception as e:
            logger.error(f"Error generating conversation title: {str(e)}")
            # Return a default title if generation fails
            return "New Conversation"

    async def process_message(self, user_id: str, message: str, model: str = None, parameters: dict = None) -> Dict[str, Any]:
        """Process a user message using a hybrid approach.

        This service implements a three-tier approach:
        1. Check knowledge base for high-confidence answers
        2. Use general knowledge for domain-related topics with low KB confidence
        3. Refer unrelated topics to humans

        Args:
            user_id: Unique identifier for the user session
            message: The message from the user
            model: The model to use for processing
            parameters: Additional parameters for the model

        Returns:
            A dictionary representing the ChatResponse schema.
        """
        # We'll try to use the knowledge base regardless of which API key is configured
        # The document_processor will handle the availability of embeddings
        # This allows the system to work with either OpenAI or OpenRouter as the model provider
        # while still using OpenAI embeddings for the knowledge base if available

        
        # Get dynamic configuration
        await self.config_service._load_config()
        llm_settings = await self.config_service.get_llm_settings()
        rag_settings = await self.config_service.get_rag_settings()
        HUMAN_REFERRAL_MESSAGE = rag_settings.human_referral_message
        KB_CONFIDENCE_THRESHOLD = rag_settings.knowledge_base_confidence_threshold
        query_analysis = {
            "confidence_score": 0.0,
            "knowledge_source": "none",
            "requires_human_referral": False,
            "reasoning": ""
        }
        params = parameters or {}
        response_parameters = {
            "model": model or llm_settings.default_model,
            "temperature": params.get("temperature", llm_settings.temperature),
            "max_tokens": params.get("max_tokens", llm_settings.max_tokens),
            "top_p": params.get("top_p", llm_settings.top_p)
        }
        answer = ""
    

        try:
            # First, check if the topic is related to our domain
             # Check if the topic is related to the domain
            is_domain_related, unrelated_keyword = self._is_topic_related_to_domain(message)
            # is_domain_related = True
            # is_domain_related=True
            if not is_domain_related:
                # Unrelated topic - refer to human
                answer = HUMAN_REFERRAL_MESSAGE
                query_analysis["confidence_score"] = 0.0
                query_analysis["knowledge_source"] = "none"
                query_analysis["requires_human_referral"] = True
                query_analysis["reasoning"] = f"Query is outside our domain expertise because it contains the keyword '{unrelated_keyword}', and requires human specialist attention."
            else:
                # Domain-related topic - try knowledge base first
                try:
                    kb_service = get_knowledge_base_service()
                    kb_result = await kb_service.query_knowledge_base(message)
                    kb_confidence = kb_result.get("confidence_score", 0) if kb_result else 0
                except RuntimeError as kb_error:
                    # Vector store configuration error - log and inform user
                    logging.error(f"Knowledge base configuration error: {str(kb_error)}")
                    answer = f"خطا در دسترسی به پایگاه دانش: {str(kb_error)}"
                    query_analysis["confidence_score"] = 0.0
                    query_analysis["knowledge_source"] = "error"
                    query_analysis["requires_human_referral"] = True
                    query_analysis["reasoning"] = "Knowledge base configuration error - vector store not available."
                    return {
                        "query_analysis": query_analysis,
                        "response_parameters": response_parameters,
                        "answer": answer
                    }
                except Exception as kb_error:
                    # Other knowledge base errors - fall back to general knowledge
                    logging.warning(f"Knowledge base query failed: {str(kb_error)}")
                    kb_confidence = 0  # Set to 0 to trigger general knowledge fallback

                # Define referral indicators once for reuse
                referral_indicators = [
                    "نیاز به بررسی توسط کارشناس",
                    "به کارشناس مراجعه کنید",
                    "خارج از حوزه تخصص",
                    "نمی‌توانم پاسخ دهم",
                    "نیاز به کارشناس",
                    "مطمئن نیستم",
                    "متاسفانه",
                    "اطلاعات کافی",
                    "متأسفانه",
                    "توصیه می‌کنم با کارشناس تماس بگیرید",
                    "بهتر است با کارشناس مشورت کنید",
                    "نیاز به مشاوره تخصصی",
                    "این موضوع نیاز به بررسی بیشتر دارد",
                    "اطلاعات دقیق‌تری نیاز است",
                    "پاسخ دقیق به این سوال نیازمند بررسی بیشتر است",
                    "برای اطلاعات دقیق‌تر با کارشناسان تماس بگیرید",
                    "این مورد خاص نیاز به بررسی دارد"
                ]
                
                if kb_confidence >= KB_CONFIDENCE_THRESHOLD:
                    # High confidence answer from knowledge base - priority source
                    answer = kb_result["answer"]
                    
                    # Check for referral indicators even with high confidence
                    if any(indicator in answer for indicator in referral_indicators):
                        answer = HUMAN_REFERRAL_MESSAGE
                        query_analysis["requires_human_referral"] = True
                        query_analysis["reasoning"] = "Knowledge base answer contains referral indicators despite high confidence."
                    else:
                        query_analysis["confidence_score"] = kb_confidence
                        query_analysis["knowledge_source"] = kb_result.get("source_type", "knowledge_base")
                        query_analysis["requires_human_referral"] = False
                        query_analysis["reasoning"] = "High confidence answer found in knowledge base (priority source)."
                    
                    response_parameters["temperature"] = 0.1  # Low temperature for factual answers
                        
                else:
                    # Low KB confidence but domain-related - try general knowledge as fallback
                    conversation = await self._get_or_create_session(user_id, model, parameters)
                    
                    # Get response using general knowledge. The conversation object already has the system prompt.
                    response = conversation.predict(input=message)
                    
                    # Check if the model indicated it needs human referral
                    if any(indicator in response for indicator in referral_indicators):
                        answer = HUMAN_REFERRAL_MESSAGE
                        query_analysis["requires_human_referral"] = True
                        query_analysis["reasoning"] = "Model determined the query requires specialist attention."
                    else:
                        answer = response
                        query_analysis["confidence_score"] = 0.6  # Assign a default confidence for general knowledge
                        query_analysis["knowledge_source"] = "general_knowledge"
                        query_analysis["requires_human_referral"] = False
                        query_analysis["reasoning"] = "Answer provided from general knowledge."
                        response_parameters["temperature"] = 0.3  # Moderate temperature for general knowledge

            # Add the final interaction to the conversation history.
            # The `conversation.predict` call above already adds the user message and the AI response to the memory
            # for the general_knowledge case. We need to manually add it for other cases.
            if query_analysis["knowledge_source"] != "general_knowledge":
                conversation = await self._get_or_create_session(user_id, model, parameters)
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
        # Filter out SystemMessage to avoid including system prompts in conversation history
        for message in memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history.append(ChatMessage(role="user", content=message.content))
            elif isinstance(message, AIMessage):
                history.append(ChatMessage(role="assistant", content=message.content))
            # SystemMessage is intentionally excluded from conversation history
        
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