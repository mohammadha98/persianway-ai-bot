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
        
        # Initialize semantic evaluator LLM
        self._evaluator_llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name="gpt-4",  # Use GPT-4 for better evaluation
            temperature=0.1,  # Low temperature for consistent evaluation
            max_tokens=200
        )
    
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
        
        # Define domain-related keywords based on PersianWay product categories
        domain_keywords = [
            # تغذیه و سلامت (Nutrition & Health)
            'تغذیه', 'سلامت', 'ویتامین', 'مکمل', 'پروتئین', 'کربوهیدرات', 'چربی', 'فیبر',
            'آنتی اکسیدان', 'امگا', 'کلسیم', 'آهن', 'زینک', 'منیزیم', 'رژیم', 'کاهش وزن',
            'افزایش وزن', 'عضله سازی', 'انرژی', 'خستگی', 'استرس', 'خواب', 'هضم', 'معده',
            'روده', 'کبد', 'کلیه', 'قلب', 'فشار خون', 'قند خون', 'کلسترول', 'ایمنی',
            'سرماخوردگی', 'آنفولانزا', 'التهاب', 'درد', 'مفاصل', 'استخوان', 'پوکی استخوان',
            'nutrition', 'health', 'vitamin', 'supplement', 'protein', 'carbohydrate', 'fat',
            'fiber', 'antioxidant', 'omega', 'calcium', 'iron', 'zinc', 'magnesium', 'diet',
            'weight loss', 'weight gain', 'muscle building', 'energy', 'fatigue', 'stress',
            'sleep', 'digestion', 'stomach', 'intestine', 'liver', 'kidney', 'heart',
            'blood pressure', 'blood sugar', 'cholesterol', 'immunity', 'cold', 'flu',
            'inflammation', 'pain', 'joints', 'bone', 'osteoporosis',
            
            # مراقبت پوست (Skin Care)
            'پوست', 'صورت', 'ماسک', 'کرم', 'لوسیون', 'سرم', 'تونر', 'پاک کننده',
            'مرطوب کننده', 'ضد آفتاب', 'ضد چروک', 'ضد پیری', 'روشن کننده', 'لک',
            'جوش', 'آکنه', 'منافذ', 'چربی', 'خشکی', 'حساسیت', 'اگزما', 'پسوریازیس',
            'ملانین', 'کلاژن', 'الاستین', 'هیالورونیک اسید', 'رتینول', 'ویتامین سی',
            'نیاسینامید', 'سالیسیلیک اسید', 'گلیکولیک اسید', 'آلفا هیدروکسی',
            'skin', 'face', 'mask', 'cream', 'lotion', 'serum', 'toner', 'cleanser',
            'moisturizer', 'sunscreen', 'anti-wrinkle', 'anti-aging', 'brightening',
            'dark spots', 'acne', 'pimples', 'pores', 'oily', 'dry', 'sensitive',
            'eczema', 'psoriasis', 'melanin', 'collagen', 'elastin', 'hyaluronic acid',
            'retinol', 'vitamin c', 'niacinamide', 'salicylic acid', 'glycolic acid',
            
            # مراقبت مو (Hair Care)
            'مو', 'موی سر', 'ریزش مو', 'طاسی', 'شامپو', 'نرم کننده', 'ماسک مو',
            'روغن مو', 'سرم مو', 'رشد مو', 'تقویت مو', 'ضخامت مو', 'درخشندگی مو',
            'خشکی مو', 'چربی مو', 'شوره', 'التهاب پوست سر', 'رنگ مو', 'دکلره',
            'فر کردن', 'صاف کردن', 'کراتین', 'پروتئین مو', 'کلاژن مو', 'بیوتین',
            'فولیکول مو', 'ریشه مو', 'نوک مو', 'موخوره', 'قارچ پوست سر',
            'hair', 'scalp', 'hair loss', 'baldness', 'shampoo', 'conditioner',
            'hair mask', 'hair oil', 'hair serum', 'hair growth', 'hair strengthening',
            'hair thickness', 'hair shine', 'dry hair', 'oily hair', 'dandruff',
            'scalp inflammation', 'hair color', 'bleach', 'curling', 'straightening',
            'keratin', 'hair protein', 'hair collagen', 'biotin', 'hair follicle',
            'hair root', 'split ends', 'hair fungus',
            
            # مراقبت بدن (Body Care)
            'بدن', 'پوست بدن', 'لوسیون بدن', 'کرم بدن', 'روغن بدن', 'اسکراب',
            'پیلینگ', 'ضد تعریق', 'دئودورانت', 'عطر', 'ادکلن', 'ماساژ', 'سلولیت',
            'ترک پوست', 'خشکی پوست', 'نرمی پوست', 'صافی پوست', 'لایه برداری',
            'مرطوب سازی', 'تغذیه پوست', 'ضد باکتری', 'ضد قارچ', 'بهداشت',
            'حمام', 'دوش', 'صابون', 'ژل شستشو', 'پودر', 'تالک', 'وازلین',
            'body', 'body skin', 'body lotion', 'body cream', 'body oil', 'scrub',
            'exfoliation', 'antiperspirant', 'deodorant', 'perfume', 'cologne',
            'massage', 'cellulite', 'stretch marks', 'dry skin', 'soft skin',
            'smooth skin', 'moisturizing', 'skin nourishment', 'antibacterial',
            'antifungal', 'hygiene', 'bath', 'shower', 'soap', 'body wash',
            'powder', 'talc', 'vaseline',
            'vitamin', 'supplement', 'protein', 'carbohydrate', 'fat', 'fiber',
            'antioxidant', 'omega', 'calcium', 'iron', 'zinc', 'magnesium', 'diet',
            'weight loss', 'weight gain', 'muscle building', 'energy', 'fatigue',
            'stress', 'sleep', 'digestion', 'stomach', 'intestine', 'liver', 'kidney',
            'heart', 'blood pressure', 'blood sugar', 'cholesterol', 'immunity',
            'cold', 'flu', 'inflammation', 'pain', 'joints', 'bone', 'osteoporosis',
            
            # مراقبت زمین (Earth Care / Agriculture)
            'کود', 'کشاورزی', 'خاک', 'کاشت', 'برداشت', 'آفت', 'بیماری گیاه',
            'گیاه', 'محصول', 'آبیاری', 'بذر', 'نهال', 'درخت', 'میوه', 'سبزی',
            'غلات', 'دام', 'طیور', 'ارگانیک', 'طبیعی', 'زیست محیطی', 'پایدار',
            'کمپوست', 'ورمی کمپوست', 'کود شیمیایی', 'کود آلی', 'نیتروژن',
            'فسفر', 'پتاسیم', 'ریز مغذی', 'pH خاک', 'رطوبت خاک', 'تهویه خاک',
            'زهکشی', 'آفت کش', 'قارچ کش', 'علف کش', 'حشره کش', 'مبارزه بیولوژیک',
            'fertilizer', 'agriculture', 'soil', 'planting', 'harvest', 'pest',
            'plant disease', 'plant', 'crop', 'irrigation', 'seed', 'seedling',
            'tree', 'fruit', 'vegetable', 'grain', 'livestock', 'poultry',
            'organic', 'natural', 'environmental', 'sustainable', 'compost',
            'vermicompost', 'chemical fertilizer', 'organic fertilizer', 'nitrogen',
            'phosphorus', 'potassium', 'micronutrients', 'soil ph', 'soil moisture',
            'soil aeration', 'drainage', 'pesticide', 'fungicide', 'herbicide',
            'insecticide', 'biological control'
        ]
        
        # Unrelated topics that should be referred to humans (topics outside PersianWay's domain)
        unrelated_keywords = [
            # سیاست و حکومت (Politics & Government)
            'سیاست', 'انتخابات', 'دولت', 'مجلس', 'رئیس جمهور', 'وزیر', 'حزب',
            'سیاستمدار', 'رای', 'کاندیدا', 'کابینه', 'پارلمان', 'قانون', 'قضاوت',
            'دادگاه', 'وکیل', 'قاضی', 'جرم', 'مجازات', 'زندان', 'پلیس',
            'politics', 'election', 'government', 'parliament', 'president', 'minister',
            'party', 'politician', 'vote', 'candidate', 'cabinet', 'law', 'court',
            'lawyer', 'judge', 'crime', 'punishment', 'prison', 'police',
            
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
            'موسیقی', 'خواننده', 'آهنگ', 'کنسرت', 'آلبوم', 'ساز', 'پیانو', 'گیتار',
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
            'دانشگاه', 'مدرسه', 'کلاس', 'معلم', 'استاد', 'دانشجو', 'امتحان',
            'نمره', 'مدرک', 'دیپلم', 'لیسانس', 'فوق لیسانس', 'دکترا',
            'ریاضی', 'فیزیک', 'شیمی', 'زیست شناسی', 'تاریخ', 'جغرافیا',
            'university', 'school', 'class', 'teacher', 'professor', 'student',
            'exam', 'grade', 'certificate', 'diploma', 'bachelor', 'master',
            'phd', 'mathematics', 'physics', 'chemistry', 'biology', 'history',
            'geography'
        ]
        
        # Check for unrelated topics first
        if any(keyword in query_lower for keyword in unrelated_keywords):
            return False
            
        # Check for domain-related topics
        if any(keyword in query_lower for keyword in domain_keywords):
            return True
            
        # For ambiguous queries, be more conservative - require explicit domain match
        return False

    def _parse_evaluation_fallback(self, response: str) -> Dict[str, Any]:
        """Fallback method to parse evaluation when JSON parsing fails.
        
        Args:
            response: The raw response from the evaluator LLM
            
        Returns:
            Dict containing parsed evaluation
        """
        response_lower = response.lower()
        
        # Check for positive indicators
        positive_indicators = ["مرتبط", "relevant", "true", "مفید", "helpful", "قابل قبول"]
        negative_indicators = ["نامرتبط", "irrelevant", "false", "خارج از حوزه", "نیاز به کارشناس"]
        
        is_relevant = any(indicator in response for indicator in positive_indicators)
        requires_referral = any(indicator in response for indicator in negative_indicators)
        
        # Be more lenient for domain-related content
        if not requires_referral and not any(indicator in response for indicator in negative_indicators):
            is_relevant = True  # Default to relevant if no clear negative signals
        
        return {
            "is_relevant": is_relevant,
            "confidence": 0.6 if is_relevant else 0.3,
            "reasoning": "Fallback parsing - could not parse JSON response",
            "requires_human_referral": requires_referral
        }

    async def _evaluate_response_relevance(self, user_query: str, bot_response: str) -> Dict[str, Any]:
        """Evaluate if the bot's response is relevant to the user's query using LLM.
        
        Args:
            user_query: The original user question
            bot_response: The bot's generated response
            
        Returns:
            Dict containing relevance score and reasoning
        """
        evaluation_prompt = f"""شما یک ارزیاب هوشمند هستید که باید کیفیت و مرتبط بودن پاسخ‌های ربات را بررسی کنید.

حوزه‌های تخصصی ربات:
- سلامت و زیبایی (پوست، مو، درمان، دارو، تغذیه، بهداشت)
- کشاورزی (کود، کاشت، آفات، گیاهان، دامداری)

سوال کاربر: {user_query}

پاسخ ربات: {bot_response}

ارزیابی کنید:
1. آیا پاسخ به سوال کاربر مرتبط است؟
2. آیا پاسخ در حوزه‌های تخصصی ربات قرار دارد؟
3. آیا پاسخ مفید و قابل استفاده است؟

توجه: اگر پاسخ در حوزه تخصص است و به سوال مرتبط است، حتی اگر کاملاً دقیق نباشد، آن را قبول کنید.
فقط پاسخ‌های کاملاً نامرتبط یا خارج از حوزه را رد کنید.

پاسخ به صورت JSON:
{{
  "is_relevant": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "دلیل کوتاه",
  "requires_human_referral": true/false
}}"""

        try:
            response = await self._evaluator_llm.apredict(evaluation_prompt)
            # Try to parse JSON response
            import json
            import re
            
            # Extract JSON from response if it's wrapped in text
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                try:
                    evaluation = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # Fallback if JSON is malformed
                    evaluation = self._parse_evaluation_fallback(response)
            else:
                # Fallback parsing
                evaluation = self._parse_evaluation_fallback(response)
                
            return evaluation
            
        except Exception as e:
            # Fallback evaluation on error
            return {
                "is_relevant": True,
                "confidence": 0.5,
                "reasoning": f"Evaluation error: {str(e)}",
                "requires_human_referral": False
            }

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
                    # High confidence answer from knowledge base - priority source
                    answer = kb_result["answer"]
                    query_analysis["confidence_score"] = kb_confidence
                    query_analysis["knowledge_source"] = kb_result.get("source_type", "knowledge_base")
                    query_analysis["requires_human_referral"] = False
                    query_analysis["reasoning"] = "High confidence answer found in knowledge base (priority source)."
                    response_parameters["temperature"] = 0.1  # Low temperature for factual answers
                    
                    # Perform semantic evaluation on KB response
                    evaluation = await self._evaluate_response_relevance(message, answer)
                    if not evaluation.get("is_relevant", True) or evaluation.get("requires_human_referral", False):
                        answer = HUMAN_REFERRAL_MESSAGE
                        query_analysis["requires_human_referral"] = True
                        query_analysis["reasoning"] = f"KB answer failed semantic evaluation: {evaluation.get('reasoning', 'Not relevant')}"
                        query_analysis["confidence_score"] = 0.0
                        query_analysis["knowledge_source"] = "none"
                        
                else:
                    # Low KB confidence but domain-related - try general knowledge as fallback
                    conversation = self._get_or_create_session(user_id)
                    
                    # Enhanced context-aware prompt for general knowledge
                    context_prompt = f"""
{settings.SYSTEM_PROMPT}

شما یک مشاور تخصصی در حوزه‌های سلامت، زیبایی و کشاورزی هستید. 
اولویت با پایگاه دانش داخلی است، اما می‌توانید از دانش عمومی نیز استفاده کنید.

دستورالعمل‌ها:
1. اگر سوال کاملاً خارج از حوزه تخصص شماست، بگویید نیاز به کارشناس دارد
2. اگر سوال مرتبط است، پاسخ مفید و دقیق ارائه دهید
3. از دانش عمومی معتبر استفاده کنید اما در چارچوب تخصص خود
4. اگر مطمئن نیستید، صراحت بگویید و به کارشناس ارجاع دهید

سوال کاربر: {message}
                    """
                    
                    # Get response using general knowledge
                    response = conversation.predict(input=context_prompt)
                    
                    # Check if the model indicated it needs human referral
                    referral_indicators = [
                        "نیاز به بررسی توسط کارشناس",
                        "به کارشناس مراجعه کنید",
                        "خارج از حوزه تخصص",
                        "نمی‌توانم پاسخ دهم",
                        "نیاز به کارشناس",
                        "مطمئن نیستم"
                    ]
                    
                    if any(indicator in response for indicator in referral_indicators):
                        answer = HUMAN_REFERRAL_MESSAGE
                        query_analysis["requires_human_referral"] = True
                        query_analysis["reasoning"] = "Model determined the query requires specialist attention."
                    else:
                        # Perform semantic evaluation on general knowledge response
                        evaluation = await self._evaluate_response_relevance(message, response)
                        
                        if evaluation.get("is_relevant", True) and not evaluation.get("requires_human_referral", False):
                            answer = response
                            query_analysis["confidence_score"] = min(0.7, evaluation.get("confidence", 0.5))  # Cap at 0.7 for general knowledge
                            query_analysis["knowledge_source"] = "general_knowledge"
                            query_analysis["requires_human_referral"] = False
                            query_analysis["reasoning"] = f"General knowledge answer passed semantic evaluation: {evaluation.get('reasoning', 'Relevant and helpful')}"
                            response_parameters["temperature"] = 0.3  # Moderate temperature for general knowledge
                        else:
                            answer = HUMAN_REFERRAL_MESSAGE
                            query_analysis["requires_human_referral"] = True
                            query_analysis["reasoning"] = f"General knowledge answer failed semantic evaluation: {evaluation.get('reasoning', 'Not relevant or requires specialist')}"
                            query_analysis["confidence_score"] = 0.0
                            query_analysis["knowledge_source"] = "none"

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