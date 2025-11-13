from logging import Logger
from typing import Dict, List, Optional, Any
import json
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
        self.generalAnswer = False
        self._config_updated_at: Optional[str] = None
        # Note: API key validation is now done dynamically in get_llm function
        

    
    async def _get_or_create_session(self, user_id: str, model: str = None, parameters: dict = None) -> ConversationChain:
        """Get an existing chat session or create a new one.
        
        Args:
            user_id: Unique identifier for the user session
            
        Returns:
            A LangChain ConversationChain for the user
        """
        await self._ensure_latest_config()
        if user_id not in self._sessions:
            # Get dynamic configuration for system prompt
            await self.config_service._load_config()
            rag_settings = await self.config_service.get_rag_settings()
            
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
            
            # Add system prompt to establish model behavior for general knowledge responses
            system_prompt = rag_settings.system_prompt
            self._sessions[user_id].memory.chat_memory.add_message(SystemMessage(content=system_prompt))
        
        return self._sessions[user_id]

    async def _ensure_latest_config(self) -> None:
        await self.config_service._load_config()
        cfg = await self.config_service.get_config()
        ts = cfg.updated_at or cfg.created_at
        if ts != self._config_updated_at:
            self._sessions.clear()
            self._memories.clear()
            self._config_updated_at = ts

    async def refresh(self) -> None:
        await self.config_service._load_config()
        cfg = await self.config_service.get_config()
        self._sessions.clear()
        self._memories.clear()
        self._config_updated_at = cfg.updated_at or cfg.created_at
    
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
            'سفیر', 'دیپلمات', 'سفارت', 'کنسولگری', 'نماینده', 'سازمان ملل',
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
        "The company focuses on FOUR main areas:\n\n"
        
        "📋 CORE BUSINESS AREAS:\n"
        "═══════════════════════════\n\n"
        
        "1️⃣ COMPANY INFORMATION (اطلاعات شرکت)\n"
        "   - Company history, establishment date, licenses\n"
        "   - Mission, vision, values, brand information\n"
        "   - Office locations, contact details, addresses\n"
        "   - Organizational structure, management team\n"
        "   - Other Brands (برندهای دیگر)\n"
        "   - Questions like: 'شرکت پرشین وی چیست؟', 'دفتر شما کجاست؟'\n\n"
        
        "2️⃣ NETWORK MARKETING BUSINESS (کسب‌وکار و قوانین بازاریابی)\n"
        "   ⭐ THIS IS A MAJOR CATEGORY - NEVER classify as OFF_TOPIC!\n"
        "   Topics include:\n"
        "   • Membership & Registration (ثبت‌نام، عضویت، جایگاه)\n"
        "   • Commission & Compensation (پورسانت، درآمد، پاداش)\n"
        "   • Violations & Penalties (تخلفات، مجازات، اخطار)\n"
        "   • License & Permits (پروانه کسب، مجوز فعالیت)\n"
        "   • Network Status (فعال/غیرفعال، تعلیق، نماد)\n"
        "   • MLM Regulations (آیین‌نامه، قوانین، مقررات)\n"
        "   • Distributor Rights (حقوق نمایندگان، قراردادها)\n"
        "   • Returns & Refunds (مرجوعی، استرداد وجه)\n"
        "   • Invoicing & Documentation (فاکتور، اسناد مالی)\n"
        "   • Training & Events (آموزش، رویدادها، کلاس‌ها)\n"
        "   • Downline Management (زیرمجموعه، گروه، تیم)\n"
        "   Examples:\n"
        "   - 'شرایط غیرفعال شدن جایگاه چیست؟'\n"
        "   - 'پورسانت چطور محاسبه میشه؟'\n"
        "   - 'تخلفات و مجازات‌ها چیه؟'\n"
        "   - 'شرایط مرجوع کالا چیست؟'\n"
        "   - 'وضعیت نماد زرد یعنی چی؟'\n\n"
        
        "3️⃣ AGRICULTURE (کشاورزی)\n"
        "   - Farming, crops, planting, harvesting\n"
        "   - Fertilizers, pesticides, irrigation\n"
        "   - Soil management, pest control\n"
        "   - Agricultural products and equipment\n\n"
        
        "4️⃣ HEALTH & BEAUTY (سلامت و زیبایی)\n"
        "   - Health, wellness, nutrition\n"
        "   - Beauty products, skincare, cosmetics\n"
        "   - Medical questions, supplements\n"
        "   - Product recommendations (Hapix brand, etc.)\n\n"
        
        "═══════════════════════════\n\n"
        
        "📊 CLASSIFICATION RULES:\n"
        "══════════════════════\n\n"
        
        "Classify into ONE of these categories:\n\n"
        
        "1. PUBLIC → Questions SPECIFICALLY about the company itself\n"
        "   (NOT about business operations or regulations)\n"
        "   Examples:\n"
        "   ✓ 'تاریخ تأسیس شرکت؟'\n"
        "   ✓ 'دفتر مرکزی کجاست؟'\n"
        "   ✓ 'شعار شرکت چیست؟'\n"
        "   ✗ 'چطور عضو بشم؟' → PRIVATE (business operations)\n"
        "   ✗ 'شرایط ثبت‌نام چیه؟' → PRIVATE (business operations)\n\n"
        
        "2. PRIVATE → Questions about business operations, regulations, products, or services\n"
        "   (This is the DEFAULT for all relevant business questions)\n"
        "   \n"
        "   ⚠️ CRITICAL: Include ALL questions about:\n"
        "   • Network marketing operations (even if technical/legal)\n"
        "   • Regulations, violations, penalties\n"
        "   • Membership, commissions, status changes\n"
        "   • Agricultural topics\n"
        "   • Health & beauty topics\n"
        "   • Products and services\n"
        "   \n"
        "   Examples:\n"
        "   ✓ 'تفاوت غیرفعال موقت و دائم چیست؟' → PRIVATE\n"
        "   ✓ 'مجازات تخلفات چیه؟' → PRIVATE\n"
        "   ✓ 'شرایط تعلیق نماد؟' → PRIVATE\n"
        "   ✓ 'پورسانت کی واریز میشه؟' → PRIVATE\n"
        "   ✓ 'محصولات شما چیه؟' → PRIVATE\n"
        "   ✓ 'بهترین کود گندم؟' → PRIVATE\n\n"
        
        "3. OFF_TOPIC → ONLY questions CLEARLY unrelated to PersianWay's business\n"
        "   ⚠️ Be EXTREMELY LENIENT - only use if OBVIOUSLY unrelated\n"
        "   \n"
        "   Examples of TRUE OFF_TOPIC:\n"
        "   ✓ 'بهترین تیم فوتبال کدومه؟' (sports)\n"
        "   ✓ 'چطور برنامه‌نویسی یاد بگیرم؟' (programming)\n"
        "   ✓ 'قیمت دلار امروز؟' (finance/forex)\n"
        "   ✓ 'فیلم خوب پیشنهاد بده' (entertainment)\n"
        "   \n"
        "   ✗ 'تفاوت غیرفعال موقت و دائم؟' → PRIVATE (business rules)\n"
        "   ✗ 'وضعیت نماد چیست؟' → PRIVATE (business status)\n"
        "   ✗ 'مجازات تخلفات؟' → PRIVATE (regulations)\n\n"
        
        "═══════════════════════════\n\n"
        
        "🎯 DECISION FLOWCHART:\n"
        "═══════════════════════\n"
        "Step 1: Is it about PersianWay company facts/history/locations?\n"
        "        → YES: PUBLIC\n"
        "        → NO: Go to Step 2\n\n"
        
        "Step 2: Is it related to business operations, MLM rules, products, agriculture, or health/beauty?\n"
        "        → YES: PRIVATE\n"
        "        → NO: Go to Step 3\n\n"
        
        "Step 3: Is it OBVIOUSLY and COMPLETELY unrelated (sports, politics, entertainment, etc.)?\n"
        "        → YES: OFF_TOPIC\n"
        "        → UNSURE: PRIVATE (default to PRIVATE when uncertain)\n\n"
        
        "═══════════════════════════\n\n"
        
        "⚠️ SPECIAL ATTENTION - Common Misclassifications:\n"
        "════════════════════════════════════════════════\n"
        "These are PRIVATE, NOT OFF_TOPIC:\n"
        "• Questions about member status (active/inactive/suspended)\n"
        "• Questions about violations and penalties\n"
        "• Questions about commissions and payments\n"
        "• Questions about MLM regulations and rules\n"
        "• Questions about product returns and refunds\n"
        "• Questions about invoicing and documentation\n"
        "• Questions comparing different MLM statuses\n"
        "• Questions about downline management\n"
        "• Questions about training and events\n\n"
        
        "Conversation History:\n"
        f"{history_block}\n\n"
        
        "Respond with valid JSON only:\n"
        "{\n"
        "  \"intent\": \"PUBLIC\" | \"PRIVATE\" | \"OFF_TOPIC\",\n"
        "  \"category\": \"company_info\" | \"mlm_business\" | \"agriculture\" | \"health_beauty\" | \"general\" | \"unrelated\",\n"
        "  \"confidence\": 0.0-1.0,\n"
        "  \"explanation\": \"brief reason in English\",\n"
        "  \"off_topic_message\": \"optional: redirect message in Persian if OFF_TOPIC\"\n"
        "}"
    )

        try:
            classifier_llm = llm or await get_llm(
                model_name="gpt-4o-mini",
                temperature=0.1,
                top_p=0.1
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
            
            # Validate intent
            if intent not in ["PUBLIC", "PRIVATE", "OFF_TOPIC"]:
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
                "off_topic_message": off_topic_message
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
                await self._ensure_latest_config()
                # Domain-related topic - first check intent
                intent_result = await self.detect_query_intent(message, conversation_history)
                
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
                    conversation = await self._get_or_create_session(user_id, model, parameters)
                    conversation.memory.chat_memory.add_user_message(message)
                    conversation.memory.chat_memory.add_ai_message(answer)
                    
                    return {
                        "query_analysis": query_analysis,
                        "response_parameters": response_parameters,
                        "answer": answer
                    }
                
           
                # Proceed with knowledge base query
                is_public = intent_result["is_public"]
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
                    else:
                        # General answers are disabled - refer to human
                        answer = HUMAN_REFERRAL_MESSAGE
                        query_analysis["requires_human_referral"] = True
                        query_analysis["reasoning"] = "Low knowledge base confidence and general answers are disabled."

            # Add the final interaction to the conversation history.
            # The `conversation.predict` call above already adds the user message and the AI response to the memory
            # for the general_knowledge case. We need to manually add it for other cases.
            if query_analysis["knowledge_source"] != "general_knowledge":
                conversation = await self._get_or_create_session(user_id, model, parameters)
                conversation.memory.chat_memory.add_user_message(message)
                conversation.memory.chat_memory.add_ai_message(answer)

            # Construct the final response dictionary
            logger.debug(f"[DEBUG] Final confidence: {query_analysis['confidence_score']:.3f}")
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
