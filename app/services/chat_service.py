
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import json
import re
import time
from langchain_openai import ChatOpenAI
import logging
from loguru import logger

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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AIMessageChunk, HumanMessageChunk
from langchain_core.runnables import Runnable
from app.services.utility import search_persianway

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
        self._sessions: Dict[str, Runnable] = {}
        self._message_history: Dict[str, List[Any]] = {}
        self.config_service = ConfigService()
        self.generalAnswer = False
        self._config_updated_at: Optional[str] = None
        # Note: API key validation is now done dynamically in get_llm function


    
    async def _get_or_create_session(self, user_id: str, model: str = None, parameters: dict = None) -> Runnable:
        logger.debug(f"[DEBUG] _get_or_create_session called with model: {model}")
        """Get an existing chat session or create a new one.
        
        Args:
            user_id: Unique identifier for the user session
            
        Returns:
            A LangChain Chain for the user
        """
        await self._ensure_latest_config()
        if user_id not in self._sessions:
            # Get dynamic configuration for system prompt
            await self.config_service._load_config()
            rag_settings = await self.config_service.get_rag_settings()
            
            # Initialize empty message history for this user (replaces ConversationBufferMemory)
            self._message_history[user_id] = []
            
            # Create a new chat model with the configured settings
            params = parameters or {}
            llm = await get_llm(model_name=model)
            
            # Add system prompt
            system_prompt = rag_settings.system_prompt
            
            # Create a simple LLM chain with system prompt
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])
            
            # Build runnable pipeline (modern LangChain style)
            self._sessions[user_id] = prompt | llm
        
        return self._sessions[user_id]

    async def _ensure_latest_config(self) -> None:
        await self.config_service._load_config()
        cfg = await self.config_service.get_config()
        ts = cfg.updated_at or cfg.created_at
        if ts != self._config_updated_at:
            self._sessions.clear()
            self._message_history.clear()
            self._config_updated_at = ts

    async def refresh(self) -> None:
        await self.config_service._load_config()
        cfg = await self.config_service.get_config()
        self._sessions.clear()
        self._message_history.clear()
        self._config_updated_at = cfg.updated_at or cfg.created_at

    def _append_history(self, user_id: str, user_message: str, ai_message: str) -> None:
        if user_id not in self._message_history:
            self._message_history[user_id] = []
        self._message_history[user_id].append(HumanMessage(content=user_message))
        self._message_history[user_id].append(AIMessage(content=ai_message))
    
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
            'دادگاه', 'وکیل', 'قاضی', 'جرم', 'مجازات', 'زندان', 'پلیس','جنگ',
            'سفیر', 'دیپلمات', 'سفارت', 'کنسولگری','سازمان ملل',
            'ناتو', 'اتحادیه اروپا', 'سنا', 'کنگره', 'مذاکره', 'تحریم', 'معاهده',
            'استیضاح', 'فساد', 'رشوه', 'اختلاس', 'براندازی', 'کودتا', 'انقلاب',
            'تظاهرات', 'اعتصاب', 'حقوق بشر', 'سانسور',
            'politics', 'election', 'government', 'parliament', 'president', 'minister',
            'party', 'politician', 'vote', 'candidate', 'cabinet', 'law', 'court',
            'lawyer', 'judge', 'crime', 'punishment', 'prison', 'police', 'america',
            'usa', 'iran', 'china', 'russia', 'europe', 'country', 'nation', 'diplomacy',
            'ambassador', 'diplomat', 'embassy', 'consulate', 'representative', 'un',
            'nato', 'eu', 'senate', 'congress', 'negotiation', 'sanction', 'treaty',
            'impeachment', 'corruption', 'bribery', 'embezzlement', 'coup', 'revolution',
            'protest', 'demonstration', 'strike', 'human rights', 'freedom of speech', 'censorship',
            'democracy', 'dictatorship', 'monarchy', 'republic', 'constitution', 'referendum',
            'propaganda', 'military', 'army', 'navy', 'air force', 'defense', 'nuclear',
            'terrorism', 'extremism', 'intelligence', 'spy', 'security council',
            
            # ورزش (Sports)
            'فوتبال', 'والیبال', 'بسکتبال', 'تنیس', 'شنا', 'دوچرخه سواری',
            'کوهنوردی', 'اسکی', 'کشتی', 'جودو', 'کاراته', 'تکواندو', 'بوکس',
          'بازیکن', 'استادیوم', 'مسابقه', 'قهرمانی',
            'المپیک', 'جام جهانی', 'لیگ', 'فینال',
            'football', 'volleyball', 'basketball', 'tennis', 'swimming', 'cycling',
            'mountaineering', 'skiing', 'wrestling', 'judo', 'karate', 'taekwondo',
            'boxing', 'sport', 'team', 'player', 'coach', 'stadium', 'competition',
            'championship', 'olympics', 'world cup', 'league', 'final', 'goal', 'score',
            
            # سرگرمی و هنر (Entertainment & Arts)
             'سینما', 'بازیگر', 'کارگردان', 'تلویزیون',
            'موسیقی', 'خواننده', 'آهنگ', 'کنسرت', 'آلبوم', 'پیانو', 'گیتار',
            'نقاشی', 'مجسمه سازی', 'عکاسی', 'تئاتر', 'رقص', 'باله', 'اپرا',
            'رمان', 'شعر', 'نویسنده', 'شاعر', 'ادبیات',
            'movie', 'cinema', 'actor', 'director', 'television', 'series', 'program',
            'music', 'singer', 'song', 'concert', 'album', 'instrument', 'piano',
            'guitar', 'painting', 'sculpture', 'photography', 'theater', 'dance',
            'ballet', 'opera', 'book', 'novel', 'poetry', 'writer', 'poet',
            'literature', 'story',
            
            # فناوری و الکترونیک (Technology & Electronics)
            'کامپیوتر', 'لپ تاپ', 'موبایل', 'تبلت', 'نرم افزار', 'برنامه نویسی',
            'اپلیکیشن', 'وب سایت', 'اینترنت', 'دیتابیس',
             'بلاک چین', 'ارز دیجیتال', 'بیت کوین',
             'کنسول', 'پلی استیشن', 'ایکس باکس', 'نینتندو',
            'computer', 'laptop', 'mobile', 'tablet', 'software', 'programming',
            'application', 'website', 'internet', 'network', 'server', 'database',
            'artificial intelligence', 'robot', 'blockchain', 'cryptocurrency',
            'bitcoin', 'game', 'gaming', 'console', 'playstation', 'xbox', 'nintendo',
            
           
            # املاک و مسکن (Real Estate & Housing)
            'آپارتمان', 'ویلا',
            'رهن', 'ودیعه', 'مشاور املاک', 'قیمت مسکن', 'متراژ'
            
            'house', 'apartment', 'villa', 'land', 'building', 'rent', 'buy',
            'sell', 'mortgage', 'deposit', 'real estate agent', 'housing price',
            'area', 'room', 'kitchen', 'bathroom', 'parking', 'storage', 'balcony',
            
            # مالی و بانکی (Finance & Banking)
             'وام', 'سپرده', 'سود', 'بهره', 'چک', 'کارت اعتباری'
            , 'دلار', 'یورو', 'بورس', 'سهام', 'سرمایه گذاری',
            'بیمه', 'مالیات', 'حسابداری', 'اقتصاد',
            'money', 'bank', 'investment', 'stock', 'economy', 'financial', 'accounting',
            'loan', 'deposit', 'profit', 'interest', 'check', 'credit card',
            'account', 'currency', 'dollar', 'euro', 'stock market',
            'shares', 'insurance', 'tax', 'inflation', 'recession',
            
            # آموزش و تحصیل (Education)
            'دانشگاه'
            , 'دیپلم', 'لیسانس', 'فوق لیسانس', 'دکترا',
            'ریاضی', 'فیزیک', 'شیمی', 'زیست شناسی', 'جغرافیا',
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


    async def _get_search_decision(self, message: str, model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
        llm = await get_llm(model_name=model_name)
        prompt = (
            "You decide whether a web search on persianway.ir is needed for a user message.\n"
            "Return ONLY a JSON object with keys search_needed and search_query.\n"
            "search_needed must be true or false.\n"
            "If search_needed is false, search_query must be an empty string.\n"
            "If search_needed is true, search_query must be a concise Persian query about PersianWay products or services.\n"
            f"User message: {message}"
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    parsed = None
        if not isinstance(parsed, dict):
            logger.warning(f"Search decision parsing failed. Raw response: {raw}")
            return {"search_needed": False, "search_query": ""}
        search_needed = parsed.get("search_needed", False)
        if isinstance(search_needed, str):
            search_needed = search_needed.strip().lower() in ["true", "1", "yes"]
        search_query = parsed.get("search_query", "")
        if search_query is None:
            search_query = ""
        search_query = str(search_query).strip()
        if not search_needed or not search_query:
            return {"search_needed": False, "search_query": ""}
        return {"search_needed": True, "search_query": search_query}


    async def generate_conversation_title(self, message: str) -> str:
        """Generate a conversation title based on the user's message.
        
        Args:
            message: The user's message to generate a title from
            
        Returns:
            A concise title for the conversation
        """
        try:
            # Get LLM instance
            llm = await get_llm(model_name="gpt-4o-mini",temperature=0.1, max_tokens=100)
            
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

    async def detect_public_data_intent(
        self,
        message: str,
        conversation_history: Optional[Any] = None,
        *,
        llm: Optional[Any] = None
    ) -> bool:
        """Legacy method for backward compatibility.
        
        Detects whether the user's message is about PersianWay public data.
        This is now a wrapper around detect_query_intent.
        
        Args:
            message: The latest user message.
            conversation_history: Prior conversation exchanges.
            llm: Optional pre-configured LLM instance.
        
        Returns:
            True if the intent relates to public PersianWay company information, otherwise False.
        """
        result = await self.detect_query_intent(message, conversation_history, llm=llm)
        return result.get("is_public", False)
    
    async def detect_query_intent(
        self,
        message: str,
        conversation_history: Optional[Any] = None,
        *,
        llm: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Detect the intent of the user's query.
        
        Args:
            message: The latest user message.
            conversation_history: Prior conversation exchanges (can be ConversationResponse, list of messages, or None).
            llm: Optional pre-configured LLM instance (used mainly for testing).
        
        Returns:
            A dictionary with:
                - intent: One of "PUBLIC", "PRIVATE", or "OFF_TOPIC"
                - is_public: Boolean indicating if it's a public query (for backward compatibility)
                - explanation: Reason for the classification
                - off_topic_message: Optional message to redirect user (for OFF_TOPIC)
        """
        if not message or not message.strip():
            return {
                "intent": "NEEDS_CLARIFICATION",
                "is_public": False,
                "explanation": "Empty message",
                "clarification_prompt": "لطفاً سوال یا درخواست خود را مشخص کنید."
            }

        formatted_history: List[str] = []
        
        if conversation_history:
            messages_to_process = []
            
            # Handle ConversationResponse object
            if hasattr(conversation_history, 'messages'):
                messages_to_process = conversation_history.messages[-6:]  # Last 6 messages
            # Handle list of ConversationResponse objects
            elif isinstance(conversation_history, list) and conversation_history:
                if hasattr(conversation_history[0], 'messages'):
                    # It's a list of ConversationResponse, take the last one
                    messages_to_process = conversation_history[-1].messages[-6:]
                else:
                    # It's already a list of messages
                    messages_to_process = conversation_history[-6:]
            
            # Process messages
            for entry in messages_to_process:
                role = None
                content = None

                # Handle MessageResponse objects (from ConversationResponse)
                if hasattr(entry, 'role') and hasattr(entry, 'content'):
                    role = entry.role
                    content = entry.content
                # Handle ChatMessage objects
                elif isinstance(entry, ChatMessage):
                    role = entry.role
                    content = entry.content
                # Handle dict
                elif isinstance(entry, dict):
                    role = entry.get("role")
                    content = entry.get("content")

                if role and content:
                    formatted_history.append(f"{role}: {content}")
        
        # Log the extracted history for debugging
        if formatted_history:
            logger.debug(f"Intent detection extracted {len(formatted_history)} messages from conversation history")
        
        history_block = "\n".join(formatted_history) if formatted_history else "No prior conversation."

        classifier_prompt = (
    "You are an intent classifier for PersianWay (پرشین وی) customer support.\n\n"
    
    "PersianWay is a Network Marketing company operating under Iranian MLM regulations.\n"
    "The company has THREE main product areas: Agriculture, Health, and Beauty.\n\n"

    "🛍️ PRODUCT CATALOG CONTEXT (List of Key Items Sold):\n"
    "════════════════════════════════════════════════════\n"
    "The user may refer to these specific products. Treat them as COMPANY PRODUCTS, not general concepts:\n"
    "• Drinks & Beverages: Kombucha (کامبوچا), Aloe Vera (آلوئه‌ورا), Energy Drinks, Herbal Teas\n"
    "• Supplements: Gabri Golden (گابری گلدن), Ganoderma (گانودرما), Ginseng (جینسینگ)\n"
    "• Health & Personal Care: Hand Creams, Shampoos, Body Splash, Masks\n"
    "• Brands: Hapix, Celux, Frei Öl, Magical, Dream World, PersianWay\n"
    "• Agriculture: Fertilizers (کود), Pesticides (سم), Growth promoters\n\n"
    
    "📋 CLASSIFICATION CATEGORIES:\n"
    "═══════════════════════════════\n\n"
    
    "1️⃣ PUBLIC (اطلاعات عمومی شرکت و محصولات)\n"
    "   ALL questions about the company, business operations, AND product identification:\n"
    "   \n"
    "   🏢 Company & Product Identity:\n"
    "   • 'What is [Product Name]?' (Product definitions)\n"
    "   • Identifying specific products (Gabri, Kombucha, Aloe Vera, etc.)\n"
    "   • Company history, licenses, office locations\n"
    "   • All brands info (Hapix, Celux, etc.)\n"
    "   \n"
    "   💼 Network Marketing Business Operations:\n"
    "   • Membership, Registration, Status\n"
    "   • Commission, Compensation Plan, Income\n"
    "   • Violations, Penalties, Yellow Symbol (نماد زرد)\n"
    "   • Returns, Refunds, Orders, Shipping\n"
    "   • Rules & Regulations\n"
    "   \n"
    "   Examples:\n"
    "   ✓ 'شرکت پرشین وی چیست؟'\n"
    "   ✓ 'کامبوچا چیست؟' (Product Identity → PUBLIC)\n"
    "   ✓ 'گابری گلدن چیه؟' (Product Identity → PUBLIC)\n"
    "   ✓ 'نوشیدنی آلوئه ورا دارید؟' (Product Availability → PUBLIC)\n"
    "   ✓ 'محصولات شما چیه؟'\n"
    "   ✓ 'چطور عضو بشم؟'\n"
    "   ✓ 'پورسانت چطور محاسبه میشه؟'\n"
    "   ✓ 'شرایط مرجوع کالا چیست؟'\n\n"
    
    "2️⃣ PRIVATE (سوالات تخصصی و مشاوره‌ای)\n"
    "   ONLY specialized technical questions requiring EXPERT advice:\n"
    "   \n"
    "   🌾 Agriculture (کشاورزی):\n"
    "   • Technical farming instructions\n"
    "   • Dosage of fertilizers for specific crops\n"
    "   • Treating plant diseases\n"
    "   \n"
    "   💊 Health & Wellness (سلامت):\n"
    "   • Medical advice, curing diseases\n"
    "   • Specific dosage for medical conditions\n"
    "   • Interaction with other drugs\n"
    "   • 'How to use X for diabetes?'\n"
    "   \n"
    "   💄 Beauty & Skincare (زیبایی):\n"
    "   • Routine for specific skin types (oily, dry)\n"
    "   • Treating acne, hair loss, skin diseases\n"
    "   \n"
    "   Examples:\n"
    "   ✓ 'بهترین کود برای گندم چیه؟' (agriculture)\n"
    "   ✓ 'کامبوچا برای دیابت خوبه؟' (health advice → PRIVATE)\n"
    "   ✓ 'گابری گلدن رو چند بار در روز بخورم؟' (dosage/usage → PRIVATE)\n"
    "   ✓ 'برای پوست خشک چی پیشنهاد میدی؟' (beauty advice)\n"
    "   ✓ 'روغن آرگان برای ریزش مو خوبه؟' (beauty)\n"
    "   ✗ 'کامبوچا چیه؟' → PUBLIC (Definition)\n"
    "   ✗ 'قیمت گابری چنده؟' → PUBLIC (Pricing)\n\n"
    
    "3️⃣ OFF_TOPIC (خارج از حوزه)\n"
    "   Questions COMPLETELY unrelated to PersianWay products or business.\n"
    "   ⚠️ IMPORTANT: If a user asks about 'Kombucha', 'Aloe Vera', or 'Mushrooms', check if it relates to PersianWay products. If yes, it is NOT Off-Topic.\n"
    "   \n"
    "   Examples:\n"
    "   ✓ 'بهترین تیم فوتبال؟'\n"
    "   ✓ 'قیمت دلار امروز؟'\n"
    "   ✓ 'طرز تهیه قورمه سبزی؟'\n"
    "   ✓ 'آب‌وهوای تهران؟'\n\n"
    
    "═══════════════════════════════\n\n"

    "4️⃣ GREETING (سلام و شروع مکالمه)\n"
    "   Examples: 'سلام', 'درود', 'خسته نباشید', 'وقت بخیر', 'Hi'\n\n"
    "5️⃣ FAREWELL (خداحافظی)\n"
    "   Examples: 'خداحافظ', 'ممنون', 'فعلا'\n\n"
    "6️⃣ SMALL_TALK (گفت‌وگوی کوتاه)\n"
    "   Examples: 'چطوری؟', 'چه خبر؟', 'شما رباتی؟'\n\n"
  
    "🎯 DECISION FLOWCHART:\n"
    "═══════════════════════\n\n"
    
    "Step 1: Is the input about a SPECIFIC PRODUCT NAME found in PersianWay's catalog (e.g., Kombucha, Gabri, Aloe Vera)?\n"
    "        → If asking 'What is it?' or 'Price/Order': PUBLIC\n"
    "        → If asking 'How to use for [Disease]?' or 'Medical benefits': PRIVATE\n\n"
    
    "Step 2: Is it about Company Info, MLM Business, or General Operations?\n"
    "        → YES: PUBLIC\n\n"
    
    "Step 3: Is it a SPECIALIZED TECHNICAL question (Agriculture/Health/Beauty usage)?\n"
    "        → YES: PRIVATE\n\n"
    
    "Step 4: Is it unrelated to everything above?\n"
    "        → YES: OFF_TOPIC\n\n"
    
    "═══════════════════════════════\n\n"
    
    "⚠️ CRITICAL RULES:\n"
    "• 'What is Kombucha?' = PUBLIC (Product Info)\n"
    "• 'Does Kombucha cure cancer?' = PRIVATE (Health Advice)\n"
    "• 'What is Gabri Golden?' = PUBLIC (Product Info)\n"
    "• 'How to farm wheat?' = PRIVATE (Agriculture)\n"
    "• 'Price of dollar?' = OFF_TOPIC\n\n"
    
    f"Conversation History:\n{history_block}\n\n"
    
    "Respond with valid JSON only:\n"
    "{\n"
    "  \"intent\": \"PUBLIC\" | \"PRIVATE\" | \"OFF_TOPIC\" | \"GREETING\" | \"FAREWELL\" | \"SMALL_TALK\",\n"
    "  \"category\": \"company_info\" | \"mlm_business\" | \"product_info\" | \"agriculture\" | \"health\" | \"beauty\" | \"unrelated\",\n"
    "  \"confidence\": 0.0-1.0,\n"
    "  \"explanation\": \"brief reason in English\",\n"
    "  \"off_topic_message\": \"optional: redirect message in Persian if OFF_TOPIC\"\n"
    "  \"greeting_message\": \"optional: greeting in Persian if GREETING\"\n"
    "  \"farewell_message\": \"optional: farewell in Persian if FAREWELL\"\n"
    "  \"small_talk_message\": \"optional: small talk reply in Persian if SMALL_TALK\"\n"
    "}")


        try:
            classifier_llm = llm or await get_llm(
                model_name="openai/gpt-4o-mini",
                temperature=0.0,
            )
        except Exception as e:
            logger.error(f"Failed to initialize intent detection LLM: {e}")
            return {
                "intent": "PRIVATE",
                "is_public": False,
                "explanation": f"Failed to initialize LLM: {str(e)}",
                "clarification_prompt": None
            }

        try:
            response = await classifier_llm.ainvoke([
                SystemMessage(content=classifier_prompt),
                HumanMessage(
                    content=(
                        f"Conversation history:\n{history_block}\n\n"
                        f"Latest user message:\n{message}\n\n"
                        "Classify the intent now."
                    )
                )
            ])
        except Exception as e:
            logger.error(f"Error during intent detection: {e}")
            return {
                "intent": "PRIVATE",
                "is_public": False,
                "explanation": f"Error during classification: {str(e)}",
                "clarification_prompt": None
            }

        content = (getattr(response, "content", "") or "").strip()
        if not content:
            logger.warning("Intent detection returned empty content")
            return {
                "intent": "PRIVATE",
                "is_public": False,
                "explanation": "Empty response from classifier",
                "clarification_prompt": None
            }

        # Parse JSON response
        payload = None
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from intent detection: {e}")
                payload = None

        # Process the response
        if isinstance(payload, dict) and "intent" in payload:
            intent = payload.get("intent", "PRIVATE").upper()
            explanation = payload.get("explanation", "No explanation provided")
            clarification_prompt = payload.get("clarification_prompt")
            off_topic_message = payload.get("off_topic_message")
            greeting_message = payload.get("greeting_message")
            farewell_message = payload.get("farewell_message")
            small_talk_message = payload.get("small_talk_message")
  
            
            # Validate intent
            if intent not in ["PUBLIC", "PRIVATE", "OFF_TOPIC", "GREETING", "FAREWELL", "SMALL_TALK", "HELP_CAPABILITIES"]:
                logger.warning(f"Invalid intent '{intent}', defaulting to PRIVATE")
                intent = "PRIVATE"
            
            # Determine is_public for backward compatibility
            is_public = (intent == "PUBLIC")
            
            # Log the classification result
            logger.info(
                f"Intent classification: message='{message[:50]}...', "
                f"intent={intent}, is_public={is_public}, explanation='{explanation}'"
            )
            
            return {
                "intent": intent,
                "is_public": is_public,
                "explanation": explanation,
                "clarification_prompt": clarification_prompt,
                "off_topic_message": off_topic_message,
                "greeting_message": greeting_message,
                "farewell_message": farewell_message,
                "small_talk_message": small_talk_message,
            
            }

        # Fallback: try old format for backward compatibility
        if isinstance(payload, dict) and "public_data" in payload:
            value = payload.get("public_data")
            explanation = payload.get("explanation", "No explanation provided")
            
            is_public = False
            if isinstance(value, bool):
                is_public = value
            elif isinstance(value, str):
                normalized_value = value.strip().lower()
                is_public = normalized_value in {"true", "yes", "public", "public_data"}
            elif isinstance(value, (int, float)):
                is_public = bool(value)
            
            logger.info(f"Intent detection (legacy format): message='{message[:50]}...', is_public={is_public}")
            
            return {
                "intent": "PUBLIC" if is_public else "PRIVATE",
                "is_public": is_public,
                "explanation": explanation,
                "clarification_prompt": None
            }

        # Ultimate fallback
        logger.warning(f"Intent detection failed to classify message: '{message[:50]}...', defaulting to PRIVATE")
        return {
            "intent": "PRIVATE",
            "is_public": False,
            "explanation": "Failed to parse classifier response",
            "clarification_prompt": None
        }

    async def process_message(self, user_id: str, message: str, conversation_history: List = None, model: str = None, parameters: dict = None) -> Dict[str, Any]:
        logger.debug(f"[DEBUG] process_message called with model: {model}")
        
        # === PERF: Timing Instrumentation ===
        t_pipeline_start = time.perf_counter()
        timings = {}
        
        """Process a user message using a hybrid approach.

        This service implements a three-tier approach:
        1. Check knowledge base for high-confidence answers
        2. Use general knowledge for domain-related topics with low KB confidence
        3. Refer unrelated topics to humans

        Args:
            user_id: Unique identifier for the user session
            message: The message from the user
            conversation_history: Previous conversation messages for context
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
        HISTORY=conversation_history
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
            # === PERF: Topic Check ===
            t0 = time.perf_counter()
            # First, check if the topic is related to our domain
             # Check if the topic is related to the domain
            # is_domain_related, unrelated_keyword = self._is_topic_related_to_domain(message)
            is_domain_related = True
            # is_domain_related=True
            if not is_domain_related:
                # Unrelated topic - refer to human
                answer = HUMAN_REFERRAL_MESSAGE
                query_analysis["confidence_score"] = 0.0
                query_analysis["knowledge_source"] = "none"
                query_analysis["requires_human_referral"] = True
                query_analysis["reasoning"] = f"Query is outside our domain expertise because it contains the keyword '{unrelated_keyword}', and requires human specialist attention."
            else:
                await self._ensure_latest_config()
                # Domain-related topic - first check intent
                
                # === PERF: Intent Detection ===
                t0 = time.perf_counter()
                intent_result = await self.detect_query_intent(message, conversation_history)
                timings['intent_detection'] = time.perf_counter() - t0
                logger.info(f"[PERF] step=intent_detection elapsed={timings['intent_detection']:.3f}s")
                
                # Handle off-topic questions
                if intent_result["intent"] == "OFF_TOPIC":
                    off_topic_msg = intent_result.get("off_topic_message") or (
                        "درود! 🌹\n\n"
                        "متأسفانه این سوال خارج از حوزه تخصص ماست. پرشین وی در حوزه‌های زیر آماده کمک به شماست:\n\n"
                        "🌱 **کشاورزی**: کاشت، داشت، کود، آبیاری، مبارزه با آفات\n"
                        "💊 **سلامت**: تغذیه، ویتامین‌ها، محصولات سلامتی\n"
                        "💄 **زیبایی**: مراقبت از پوست، محصولات آرایشی و بهداشتی\n"
                        "🏢 **اطلاعات شرکت**: درباره پرشین وی، خدمات و محصولات\n\n"
                        "چطور می‌تونم در این زمینه‌ها بهتون کمک کنم؟"
                    )
                    
                    answer = off_topic_msg
                    query_analysis["confidence_score"] = 0.3
                    query_analysis["knowledge_source"] = "off_topic_redirect"
                    query_analysis["requires_human_referral"] = True
                    query_analysis["reasoning"] = f"Query is off-topic: {intent_result['explanation']}"
                    
                    # Add to conversation memory
                    await self._get_or_create_session(user_id, model, parameters)
                    self._append_history(user_id, message, answer)
                    
                    return {
                        "query_analysis": query_analysis,
                        "response_parameters": response_parameters,
                        "answer": answer
                    }
                if intent_result["intent"] == "GREETING":
                    answer =  intent_result.get("greeting_message") or (
                        "درود! خوش آمدید به پرشین وی 🌷\n\n"
                        "می‌تونید در این زمینه‌ها سوال بپرسید:\n\n"
                        "🌱 کشاورزی: کاشت، داشت، کوددهی، آبیاری، کنترل آفات\n"
                        "💊 سلامت: مکمل‌ها، تداخل‌ها، دوز مصرف، تغذیه\n"
                        "💄 زیبایی: مراقبت از پوست، ترکیبات، روتین‌ها\n"
                        "🏢 اطلاعات شرکت: ثبت‌نام، پورسانت، قوانین، سفارش و ارسال\n\n"
                        "هر سوالی دارید بفرمایید؛ با کمال میل راهنمایی می‌کنم."
                    )
                    query_analysis["confidence_score"] = 0.5
                    query_analysis["knowledge_source"] = "greeting"
                    query_analysis["requires_human_referral"] = False
                    query_analysis["reasoning"] = "User initiated conversation with a greeting."
                    response_parameters["temperature"] = 0.2
                    await self._get_or_create_session(user_id, model, parameters)
                    self._append_history(user_id, message, answer)
                    return {
                        "query_analysis": query_analysis,
                        "response_parameters": response_parameters,
                        "answer": answer
                    }
                if intent_result["intent"] == "FAREWELL":
                    answer = intent_result.get("farewell_message") or (
                        "سپاس از همراهی شما 🌟\nاگر سوال دیگری دارید در هر زمان خوشحال می‌شوم کمک کنم. روزتون بخیر!"
                    )
                    query_analysis["confidence_score"] = 0.5
                    query_analysis["knowledge_source"] = "farewell"
                    query_analysis["requires_human_referral"] = False
                    query_analysis["reasoning"] = "User ended the conversation."
                    response_parameters["temperature"] = 0.2
                    await self._get_or_create_session(user_id, model, parameters)
                    self._append_history(user_id, message, answer)
                    return {
                        "query_analysis": query_analysis,
                        "response_parameters": response_parameters,
                        "answer": answer
                    }
                if intent_result["intent"] == "SMALL_TALK":
                    answer = intent_result.get("small_talk_message") or (
                        "روز شما هم بخیر 😊\nدر چه زمینه‌ای می‌تونم کمک کنم؟ کشاورزی، سلامت، زیبایی یا اطلاعات شرکت؟"
                    )
                    query_analysis["confidence_score"] = 0.5
                    query_analysis["knowledge_source"] = "small_talk"
                    query_analysis["requires_human_referral"] = False
                    query_analysis["reasoning"] = "User engaged in small talk."
                    response_parameters["temperature"] = 0.2
                    await self._get_or_create_session(user_id, model, parameters)
                    self._append_history(user_id, message, answer)
                    return {
                        "query_analysis": query_analysis,
                        "response_parameters": response_parameters,
                        "answer": answer
                    }
                if intent_result["intent"] == "HELP_CAPABILITIES":
                    answer =  (
                        "من دستیار هوشمند پرشین وی هستم 🤖\nمی‌تونم در این حوزه‌ها کمک کنم:\n\n"
                        "🌱 کشاورزی: کوددهی، آبیاری، آفات، روش‌های کشت\n"
                        "💊 سلامت: دوز مکمل‌ها، تداخل‌ها، تغذیه علمی\n"
                        "💄 زیبایی: روتین‌ها، ترکیبات، درمان‌های پوستی\n"
                        "🏢 اطلاعات شرکت: ثبت‌نام، پورسانت، قوانین، سفارش\n\n"
                        "کافیه سوالتون رو همین‌جا بپرسید تا راهنمایی کنم."
                    )
                    query_analysis["confidence_score"] = 0.6
                    query_analysis["knowledge_source"] = "help_capabilities"
                    query_analysis["requires_human_referral"] = False
                    query_analysis["reasoning"] = "User asked for assistant capabilities."
                    response_parameters["temperature"] = 0.2
                    await self._get_or_create_session(user_id, model, parameters)
                    self._append_history(user_id, message, answer)
                    return {
                        "query_analysis": query_analysis,
                        "response_parameters": response_parameters,
                        "answer": answer
                    }
                
           
                # Proceed with knowledge base query
                # Web search decision and execution is now handled inside query_knowledge_base
                is_public = intent_result["is_public"]
                
                # === PERF: Knowledge Base Query ===
                t0 = time.perf_counter()
                try:
                    kb_service = get_knowledge_base_service()
                    kb_result = await kb_service.query_knowledge_base(message, conversation_history, is_public)
                    kb_confidence = kb_result.get("confidence_score", 0) if kb_result else 0
                    logger.debug(f"[DEBUG] KB raw confidence: {kb_confidence:.3f}")
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
                    "متاسفانه اطلاعات",
                    "متأسفانه اطلاعات کافی",
                    "متأسفانه",
                    "متاسفانه",
                    "اطلاعات کافی در دسترس نیست",
                    "اطلاعات کافی درباره",
                    "اطلاعات کافی در مورد",
                    "اطلاعات کافی برای پاسخ",
                ]
                
                if kb_confidence >= KB_CONFIDENCE_THRESHOLD:
                    # High confidence answer from knowledge base - priority source
                    answer = kb_result["answer"]
                    
                    # Check for referral indicators even with high confidence
                    if any(indicator in answer for indicator in referral_indicators):
                        answer = answer
                        query_analysis["requires_human_referral"] = True
                        query_analysis["reasoning"] = "Knowledge base answer contains referral indicators despite high confidence."
                    else:
                        query_analysis["confidence_score"] = kb_confidence
                        query_analysis["knowledge_source"] = kb_result.get("source_type", "knowledge_base")
                        query_analysis["requires_human_referral"] = False
                        query_analysis["reasoning"] = "High confidence answer found in knowledge base (priority source)."
                    
                    response_parameters["temperature"] = 0.1  # Low temperature for factual answers
                        
                else:
                    # Low KB confidence - check if general answers are allowed
                    if self.generalAnswer:
                        # Domain-related but low confidence - try general knowledge as fallback
                        logger.info("Low KB confidence. Attempting Web Search Integration...")
                        
                        try:
                            # Use general conversation chain for low confidence cases
                            conversation = await self._get_or_create_session(user_id, model, parameters)
                            history = self._message_history.get(user_id, [])
                            response = await conversation.ainvoke({"input": message, "history": history})
                            response_content = getattr(response, "content", str(response))

                            if any(indicator in response_content for indicator in referral_indicators):
                                answer = response_content
                                query_analysis["requires_human_referral"] = True
                                query_analysis["reasoning"] = "Model determined the query requires specialist attention."
                            else:
                                answer = response_content
                                query_analysis["confidence_score"] = 0.6
                                query_analysis["knowledge_source"] = "general_knowledge"
                                query_analysis["requires_human_referral"] = False
                                query_analysis["reasoning"] = "Answer provided from general knowledge (fallback)."
                                response_parameters["temperature"] = 0.3
                                
                        except Exception as general_error:
                            logger.error(f"General knowledge processing failed: {general_error}")
                            # Fallback to human referral
                            answer = HUMAN_REFERRAL_MESSAGE
                            query_analysis["requires_human_referral"] = True
                            query_analysis["reasoning"] = f"General knowledge processing failed: {str(general_error)}"
                    else:
                        # General answers are disabled - refer to human
                        answer = HUMAN_REFERRAL_MESSAGE
                        query_analysis["requires_human_referral"] = True
                        query_analysis["reasoning"] = "Low knowledge base confidence and general answers are disabled."

            # Add the final interaction to the conversation history.
            # The `conversation.predict` call above already adds the user message and the AI response to the memory
            # for the general_knowledge case. We need to manually add it for other cases.
            if query_analysis["knowledge_source"] != "general_knowledge":
                await self._get_or_create_session(user_id, model, parameters)
                self._append_history(user_id, message, answer)
            else:
                self._append_history(user_id, message, answer)

            # Construct the final response dictionary
            logger.debug(f"[DEBUG] Final confidence: {query_analysis['confidence_score']:.3f}")
            
            # === PERF: KB Query Timing ===
            timings['retrieval'] = time.perf_counter() - t0
            
            # === PERF: Total Pipeline Timing ===
            timings['total_pipeline'] = time.perf_counter() - t_pipeline_start
            logger.info(f"[PERF] Pipeline timings: {timings}")
            
            return {
                "query_analysis": query_analysis,
                "response_parameters": response_parameters,
                "answer": answer
            }

        except Exception as e:
            logger.error(f"Error in process_message: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
        if user_id not in self._message_history:
            return None

        memory = self._message_history[user_id]
        history = []
        
        # Convert LangChain memory to our ChatMessage schema
        # Filter out SystemMessage to avoid including system prompts in conversation history
        for message in memory:
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
