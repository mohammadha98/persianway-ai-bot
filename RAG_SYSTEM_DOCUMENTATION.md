# مستند کامل سیستم RAG (Retrieval-Augmented Generation)

## 📋 فهرست مطالب

1. [مقدمه و معماری کلی](#مقدمه-و-معماری-کلی)
2. [اجزای اصلی سیستم](#اجزای-اصلی-سیستم)
3. [جریان کار (Workflow)](#جریان-کار-workflow)
4. [جزئیات فنی هر جزء](#جزئیات-فنی-هر-جزء)
5. [بهبودهای اعمال شده](#بهبودهای-اعمال-شده)
6. [مثال‌های عملی](#مثال‌های-عملی)
7. [پیکربندی و تنظیمات](#پیکربندی-و-تنظیمات)

---

## مقدمه و معماری کلی

### 🎯 هدف سیستم

سیستم RAG (Retrieval-Augmented Generation) پرشین وی یک سیستم هوشمند پاسخ‌دهی است که:

- از پایگاه دانش (Knowledge Base) برای پاسخ‌دهی دقیق استفاده می‌کند
- با استفاده از Embeddings و Vector Search، مرتبط‌ترین اطلاعات را پیدا می‌کند
- با استفاده از LLM، پاسخ‌های طبیعی و دقیق تولید می‌کند
- از Context و تاریخچه گفتگو برای درک بهتر سوالات استفاده می‌کند

### 🏗️ معماری کلی

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query (سوال کاربر)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              ChatService (سرویس اصلی چت)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Intent Detection (تشخیص قصد کاربر)              │   │
│  │     - PUBLIC: سوالات درباره شرکت                     │   │
│  │     - PRIVATE: سوالات تخصصی (کشاورزی، سلامت، زیبایی)│   │
│  │     - OFF_TOPIC: سوالات غیرمرتبط                    │   │
│  │     - NEEDS_CLARIFICATION: سوالات مبهم              │   │
│  └──────────────────────────────────────────────────────┘   │
│                        │                                     │
│                        ▼                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  2. Domain Check (بررسی ارتباط با دامنه)             │   │
│  │     - فیلتر کردن سوالات غیرمرتبط                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                        │                                     │
│                        ▼                                     │
└───────────────────────┼─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         KnowledgeBaseService (سرویس پایگاه دانش)             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Query Expansion (گسترش و بازنویسی سوال)          │   │
│  │     - بازنویسی با توجه به تاریخچه گفتگو              │   │
│  │     - ایجاد نسخه‌های جایگزین از سوال                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                        │                                     │
│                        ▼                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  2. Multi-Query Search (جستجوی چندگانه)              │   │
│  │     - جستجو با query اصلی (وزن بیشتر)                 │   │
│  │     - جستجو با query های گسترش یافته (وزن کمتر)       │   │
│  │     - استفاده از MMR برای تنوع نتایج                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                        │                                     │
│                        ▼                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  3. Similarity Threshold Filtering                    │   │
│  │     - فیلتر کردن نتایج با similarity پایین            │   │
│  └──────────────────────────────────────────────────────┘   │
│                        │                                     │
│                        ▼                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  4. Deduplication (حذف تکراری‌ها)                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                        │                                     │
│                        ▼                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  5. QA Chain (زنجیره پاسخ‌دهی)                        │   │
│  │     - ترکیب context با سوال                           │   │
│  │     - تولید پاسخ با LLM                                │   │
│  └──────────────────────────────────────────────────────┘   │
│                        │                                     │
│                        ▼                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  6. Confidence Calculation (محاسبه اطمینان)         │   │
│  │     - Multi-factor confidence (چند فاکتوری)           │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              ChromaDB Vector Store (پایگاه داده برداری)     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  - ذخیره Embeddings اسناد                            │   │
│  │  - جستجوی Semantic Similarity                        │   │
│  │  - MMR (Maximal Marginal Relevance)                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## اجزای اصلی سیستم

### 1. ChatService (سرویس اصلی چت)

**مسئولیت‌ها:**
- مدیریت جلسات گفتگو (Conversation Sessions)
- تشخیص قصد کاربر (Intent Detection)
- بررسی ارتباط سوال با دامنه (Domain Check)
- هماهنگی بین Knowledge Base و General Knowledge
- مدیریت Memory و تاریخچه گفتگو

**کلاس اصلی:** `ChatService`

**متدهای کلیدی:**

#### `process_message()`
- نقطه ورود اصلی برای پردازش پیام کاربر
- جریان کار:
  1. بررسی Intent (PUBLIC/PRIVATE/OFF_TOPIC/NEEDS_CLARIFICATION)
  2. بررسی Domain Related بودن
  3. فراخوانی Knowledge Base
  4. Fallback به General Knowledge در صورت نیاز
  5. مدیریت Memory

#### `detect_query_intent()`
- تشخیص قصد کاربر با استفاده از LLM
- دسته‌بندی‌ها:
  - **PUBLIC**: سوالات درباره شرکت پرشین وی
  - **PRIVATE**: سوالات تخصصی (کشاورزی، سلامت، زیبایی)
  - **OFF_TOPIC**: سوالات غیرمرتبط
  - **NEEDS_CLARIFICATION**: سوالات مبهم

#### `_is_topic_related_to_domain()`
- بررسی ارتباط سوال با دامنه تخصصی
- فیلتر کردن کلمات کلیدی غیرمرتبط (سیاست، ورزش، فناوری، ...)
- استفاده از Word Boundary برای دقت بیشتر

#### `_get_or_create_session()`
- مدیریت جلسات گفتگو
- ایجاد Memory برای هر کاربر
- تنظیم System Prompt

---

### 2. KnowledgeBaseService (سرویس پایگاه دانش)

**مسئولیت‌ها:**
- مدیریت افزودن محتوا به پایگاه دانش
- Query Expansion و Contextual Rewriting
- جستجوی Multi-Query با Weighted Scoring
- استفاده از MMR برای تنوع
- محاسبه Confidence Score
- تولید پاسخ با QA Chain

**کلاس اصلی:** `KnowledgeBaseService`

**متدهای کلیدی:**

#### `query_knowledge_base()`
- متد اصلی برای جستجو در پایگاه دانش
- جریان کار:
  1. استخراج و فیلتر کردن تاریخچه گفتگو
  2. Query Expansion (بازنویسی و گسترش)
  3. Multi-Query Search با MMR
  4. Similarity Threshold Filtering
  5. Deduplication
  6. تولید پاسخ با QA Chain
  7. محاسبه Confidence

#### `expand_query_with_context()`
- بازنویسی سوال با توجه به تاریخچه گفتگو
- ایجاد نسخه‌های جایگزین از سوال
- استفاده از GPT-4o-mini برای بازنویسی
- بازگشت: `{rewritten_query, expanded_queries, all_queries}`

#### `add_knowledge_contribution()`
- افزودن محتوای جدید به پایگاه دانش
- پشتیبانی از:
  - متن ساده
  - فایل PDF
  - فایل DOCX
  - فایل Excel (QA Pairs)
- ذخیره در Vector Store و MongoDB

#### `_calculate_confidence_score()`
- محاسبه Multi-factor Confidence:
  - **60%**: Best Document Score
  - **30%**: Score Consistency (انحراف معیار)
  - **10%**: Coverage (تعداد documents مرتبط)

---

### 3. DocumentProcessor (پردازشگر اسناد)

**مسئولیت‌ها:**
- پردازش فایل‌های PDF و DOCX
- تقسیم متن به Chunks
- ایجاد Embeddings با OpenAI
- مدیریت ChromaDB Vector Store

**کلاس اصلی:** `DocumentProcessor`

**ویژگی‌ها:**
- پشتیبانی از فارسی با جداکننده‌های مناسب
- Chunk Size: 1000 کاراکتر (انگلیسی) / 500 کاراکتر (فارسی)
- Chunk Overlap: 200 کاراکتر (انگلیسی) / 50 کاراکتر (فارسی)

---

### 4. ExcelQAProcessor (پردازشگر Excel)

**مسئولیت‌ها:**
- پردازش فایل‌های Excel با ساختار QA
- تبدیل سوال-جواب‌ها به Documents
- افزودن به Vector Store

**ساختار مورد انتظار Excel:**
- ستون‌های: `Title`, `Question`, `Answer`

---

## جریان کار (Workflow)

### 📊 نمودار جریان کامل

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│  ChatService.process_message()          │
└───────────────┬───────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  1. Intent Detection                   │
│     detect_query_intent()              │
│     └─> PUBLIC / PRIVATE / OFF_TOPIC   │
└───────────────┬───────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  2. Domain Check                        │
│     _is_topic_related_to_domain()      │
│     └─> Filter unrelated keywords      │
└───────────────┬───────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  3. Knowledge Base Query                │
│     KnowledgeBaseService.query_kb()     │
└───────────────┬───────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  3.1. Extract Conversation History      │
│       _extract_conversation_history()   │
└───────────────┬───────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  3.2. Query Expansion                  │
│       expand_query_with_context()       │
│       ├─> Rewrite with context         │
│       └─> Generate expanded queries    │
└───────────────┬───────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  3.3. Multi-Query Search               │
│       For each query:                   │
│       ├─> MMR Search (diversity)        │
│       ├─> Similarity Search (scores)    │
│       ├─> Match & Weight                │
│       └─> Collect results               │
└───────────────┬───────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  3.4. Filtering & Deduplication        │
│       ├─> Similarity Threshold         │
│       └─> Remove duplicates            │
└───────────────┬───────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  3.5. QA Chain                          │
│       ├─> Normalize documents           │
│       ├─> Build context                 │
│       └─> Generate answer               │
└───────────────┬───────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  3.6. Confidence Calculation           │
│       _calculate_confidence_score()     │
│       └─> Multi-factor confidence      │
└───────────────┬───────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  4. Response Decision                   │
│     ├─> High Confidence → Use KB Answer│
│     ├─> Low Confidence → General LLM   │
│     └─> Very Low → Human Referral       │
└───────────────┬───────────────────────┘
                │
                ▼
         Final Response
```

---

## جزئیات فنی هر جزء

### 1. Query Expansion (گسترش سوال)

**هدف:** بهبود دقت جستجو با بازنویسی و گسترش سوال

**فرآیند:**

```python
# 1. بازنویسی با Context
if conversation_history:
    # استفاده از تاریخچه برای درک بهتر
    rewritten_query = rewrite_with_context(query, history)
else:
    rewritten_query = query

# 2. ایجاد نسخه‌های جایگزین
expanded_queries = [
    "نسخه جایگزین 1 با مترادف‌ها",
    "نسخه جایگزین 2 با اصطلاحات مختلف",
    "نسخه جایگزین 3 با عبارات متفاوت"
]

# 3. ترکیب همه queries
all_queries = [rewritten_query] + expanded_queries
```

**مثال:**

```
سوال اصلی: "کود برای گندم"
بازنویسی شده: "بهترین کود برای کاشت و رشد گندم چیست"
نسخه‌های جایگزین:
  - "انواع کود مناسب برای گندم"
  - "کوددهی گندم و روش‌های آن"
  - "کودهای شیمیایی و ارگانیک برای گندم"
```

---

### 2. Multi-Query Search با Weighted Scoring

**هدف:** جستجوی جامع با اولویت‌بندی نتایج

**فرآیند:**

```python
# برای هر query:
for idx, query in enumerate(all_queries):
    # تعیین وزن
    if idx == 0:  # Query اصلی
        weight = 1.0  # وزن بیشتر
    else:  # Query های گسترش یافته
        weight = 0.7  # وزن کمتر
    
    # 1. MMR Search برای تنوع
    mmr_docs = vector_store.max_marginal_relevance_search(
        query,
        k=5,
        fetch_k=15,  # 3x بیشتر برای انتخاب
        lambda_mult=0.3  # تعادل relevance/diversity
    )
    
    # 2. Similarity Search برای Score
    similarity_results = vector_store.similarity_search_with_score(
        query,
        k=15
    )
    
    # 3. Match کردن و اعمال وزن
    for doc in mmr_docs:
        score = find_score_in_similarity_results(doc, similarity_results)
        weighted_score = score / weight  # کمتر = بهتر
        results.append((doc, weighted_score))
```

**مزایا:**
- Query اصلی اولویت بیشتری دارد
- Query های جایگزین coverage بیشتری ایجاد می‌کنند
- MMR از redundancy جلوگیری می‌کند

---

### 3. MMR (Maximal Marginal Relevance)

**هدف:** تعادل بین Relevance و Diversity

**پارامترها:**
- `k`: تعداد نتایج نهایی (مثلاً 5)
- `fetch_k`: تعداد کاندیداهای اولیه (مثلاً 15)
- `lambda_mult`: تعادل (0.0 = فقط diversity، 1.0 = فقط relevance)

**فرمول:**
```
MMR = λ × Relevance - (1-λ) × Similarity_to_selected
```

**مثال:**
```
lambda_mult = 0.3:
- 30% اهمیت به relevance
- 70% اهمیت به diversity

نتیجه: نتایج متنوع‌تر اما همچنان مرتبط
```

---

### 4. Similarity Threshold Filtering

**هدف:** حذف نتایج با similarity پایین

**فرآیند:**

```python
# فیلتر کردن بر اساس threshold
filtered_docs = [
    (doc, score) 
    for doc, score in all_results
    if score <= similarity_threshold  # 1.5 به طور پیش‌فرض
]

# اگر همه فیلتر شدند، graceful degradation
if not filtered_docs:
    filtered_docs = top_k_results  # نگه داشتن بهترین‌ها
```

**نکته:** در L2 Distance، score کمتر = similarity بیشتر

---

### 5. Multi-Factor Confidence Calculation

**هدف:** محاسبه دقیق‌تر confidence با در نظر گیری چند فاکتور

**فرمول:**

```python
confidence = (
    best_score_confidence × 0.6 +      # 60% وزن
    consistency_score × 0.3 +           # 30% وزن
    coverage_score × 0.1                # 10% وزن
)
```

**جزئیات:**

1. **Best Score Confidence (60%)**
   ```python
   # تبدیل distance score به confidence
   best_confidence = 1 / (1 + exp(5.0 × (score - 1.5)))
   ```

2. **Consistency Score (30%)**
   ```python
   # انحراف معیار کمتر = consistency بیشتر
   std = np.std(top_scores)
   consistency = 1 / (1 + std × 2.0)
   ```

3. **Coverage Score (10%)**
   ```python
   # تعداد documents مرتبط
   coverage = min(num_docs / top_n, 1.0)
   ```

**مثال:**

```
Top 3 Scores: [0.5, 0.6, 0.7]
Best Score: 0.5 → Confidence: 0.85
Consistency: std=0.08 → Score: 0.86
Coverage: 3/3 = 1.0

Final Confidence = 0.85×0.6 + 0.86×0.3 + 1.0×0.1 = 0.87
```

---

### 6. QA Chain (زنجیره پاسخ‌دهی)

**هدف:** تولید پاسخ با استفاده از Context و LLM

**فرآیند:**

```python
# 1. Normalize Documents
normalized_docs = _normalize_documents_for_context(docs)
# - حذف metadata اضافی
# - محدود کردن طول محتوا
# - پاکسازی متن

# 2. Build Context
context = "\n\n".join([
    f"سند {i+1}:\n{doc.page_content}" 
    for i, doc in enumerate(normalized_docs)
])

# 3. Create Prompt
prompt = f"""
{system_prompt}

{rag_template}

اطلاعات مرجع:
{context}

سوال: {rewritten_query}

پاسخ:
"""

# 4. Generate Answer
answer = llm.invoke(prompt)
```

**Template Structure:**
```
System Prompt (نقش و دستورالعمل‌ها)
    +
RAG Template (دستورالعمل پاسخ‌دهی)
    +
Context (اسناد مرتبط)
    +
Query (سوال کاربر)
    =
Final Prompt
```

---

## بهبودهای اعمال شده

### ✅ 1. Weighted Multi-Query Search

**قبل:**
- همه queries وزن یکسان داشتند
- Query اصلی و expanded queries تفاوتی نداشتند

**بعد:**
- Query اصلی: وزن 1.0
- Expanded queries: وزن 0.7
- نتایج query اصلی اولویت بیشتری دارند

---

### ✅ 2. MMR برای تنوع

**قبل:**
- فقط Similarity Search (نتایج مشابه)
- احتمال redundancy بالا

**بعد:**
- MMR Search برای تنوع
- `lambda_mult=0.3` برای تعادل مناسب
- نتایج متنوع‌تر و کامل‌تر

---

### ✅ 3. Similarity Threshold Filtering

**قبل:**
- همه نتایج استفاده می‌شدند
- حتی نتایج با similarity پایین

**بعد:**
- فیلتر کردن بر اساس threshold (1.5)
- فقط نتایج مرتبط استفاده می‌شوند
- Graceful degradation در صورت فیلتر شدن همه

---

### ✅ 4. Multi-Factor Confidence

**قبل:**
- فقط Best Score برای confidence
- عدم توجه به consistency و coverage

**بعد:**
- 3 فاکتور: Best Score (60%), Consistency (30%), Coverage (10%)
- محاسبه دقیق‌تر confidence
- تصمیم‌گیری بهتر برای human referral

---

### ✅ 5. Query Expansion با Context

**قبل:**
- فقط query اصلی جستجو می‌شد
- عدم استفاده از تاریخچه گفتگو

**بعد:**
- بازنویسی query با توجه به تاریخچه
- ایجاد نسخه‌های جایگزین
- Coverage بهتر از زوایای مختلف سوال

---

## مثال‌های عملی

### مثال 1: سوال ساده

**ورودی:**
```
User: "بهترین کود برای گندم چیست؟"
History: []
```

**فرآیند:**

1. **Intent Detection:**
   ```
   Intent: PRIVATE
   is_public: False
   ```

2. **Query Expansion:**
   ```
   Rewritten: "بهترین کود برای کاشت و رشد گندم چیست"
   Expanded:
     - "انواع کود مناسب برای گندم"
     - "کوددهی گندم و روش‌های آن"
     - "کودهای شیمیایی و ارگانیک برای گندم"
   ```

3. **Multi-Query Search:**
   ```
   Query 1 (weight=1.0): 5 results
   Query 2 (weight=0.7): 3 results
   Query 3 (weight=0.7): 2 results
   Query 4 (weight=0.7): 1 result
   Total: 11 candidates
   ```

4. **Filtering:**
   ```
   Similarity Threshold: 1.5
   Filtered: 8 results (3 removed)
   ```

5. **Deduplication:**
   ```
   Unique: 5 results
   ```

6. **QA Chain:**
   ```
   Context: 5 documents about fertilizers for wheat
   Answer: "بهترین کود برای گندم شامل..."
   ```

7. **Confidence:**
   ```
   Best Score: 0.4 → Confidence: 0.92
   Consistency: 0.88
   Coverage: 1.0
   Final: 0.91
   ```

**خروجی:**
```json
{
  "answer": "بهترین کود برای گندم شامل...",
  "confidence_score": 0.91,
  "source_type": "knowledge_base",
  "requires_human_support": false
}
```

---

### مثال 2: سوال با Context

**ورودی:**
```
User: "چطور استفاده کنم؟"
History: [
  {"role": "user", "content": "بهترین کود برای گندم چیست؟"},
  {"role": "assistant", "content": "بهترین کود برای گندم..."}
]
```

**فرآیند:**

1. **Query Expansion با Context:**
   ```
   Rewritten: "چطور از کود گندم استفاده کنم"
   (با توجه به تاریخچه، "کود" به "کود گندم" تبدیل شد)
   ```

2. **جستجو:**
   ```
   Query با context: نتایج مرتبط‌تر
   ```

**نتیجه:** پاسخ دقیق‌تر با توجه به context

---

### مثال 3: سوال غیرمرتبط

**ورودی:**
```
User: "بهترین تیم فوتبال کدومه؟"
```

**فرآیند:**

1. **Domain Check:**
   ```
   Keyword: "فوتبال" → Unrelated
   Result: False
   ```

2. **Intent Detection:**
   ```
   Intent: OFF_TOPIC
   ```

**خروجی:**
```json
{
  "answer": "درود! 🌹\n\nمتأسفانه این سوال خارج از حوزه تخصص ماست...",
  "confidence_score": 0.3,
  "source_type": "off_topic_redirect",
  "requires_human_referral": true
}
```

---

## پیکربندی و تنظیمات

### RAG Settings (در ConfigService)

```python
RAGSettings(
    # Confidence Threshold
    knowledge_base_confidence_threshold: 0.5,  # حداقل confidence
    
    # Search Settings
    search_type: "mmr",  # نوع جستجو
    top_k_results: 5,  # تعداد نتایج
    
    # Similarity Threshold
    similarity_threshold: 1.5,  # حداکثر distance score
    
    # MMR Settings
    mmr_diversity_score: 0.3,  # تعادل relevance/diversity
    fetch_k_multiplier: 3,  # ضریب fetch_k
    
    # Query Weights
    original_query_weight: 1.0,  # وزن query اصلی
    expanded_query_weight: 0.7,  # وزن query های جایگزین
    
    # LLM Settings
    temperature: 0.1,  # دما برای پاسخ‌های دقیق
    
    # Prompts
    system_prompt: "...",  # System prompt
    prompt_template: "...",  # RAG template
    human_referral_message: "..."  # پیام human referral
)
```

### تنظیمات پیشنهادی

**برای دقت بیشتر:**
```python
similarity_threshold: 1.2  # سخت‌گیرانه‌تر
mmr_diversity_score: 0.2   # تنوع کمتر، relevance بیشتر
original_query_weight: 1.2  # وزن بیشتر برای query اصلی
```

**برای coverage بیشتر:**
```python
top_k_results: 7  # نتایج بیشتر
fetch_k_multiplier: 4  # pool بزرگتر
mmr_diversity_score: 0.4  # تنوع بیشتر
```

---

## خلاصه و نکات مهم

### ✅ نقاط قوت سیستم

1. **Multi-Query Search**: Coverage بهتر از زوایای مختلف
2. **MMR**: نتایج متنوع و غیرتکراری
3. **Weighted Scoring**: اولویت‌بندی صحیح
4. **Multi-Factor Confidence**: محاسبه دقیق‌تر
5. **Context Awareness**: استفاده از تاریخچه گفتگو
6. **Graceful Degradation**: Fallback در صورت خطا

### ⚠️ نکات مهم

1. **Embeddings**: نیاز به OpenAI API Key برای embeddings
2. **Vector Store**: ChromaDB باید initialize شده باشد
3. **Memory**: هر کاربر session جداگانه دارد
4. **Config**: تنظیمات از MongoDB خوانده می‌شوند

### 🔄 جریان داده

```
User Query
    ↓
ChatService (Intent + Domain Check)
    ↓
KnowledgeBaseService (Query Expansion)
    ↓
ChromaDB (Vector Search)
    ↓
QA Chain (Answer Generation)
    ↓
Confidence Calculation
    ↓
Response Decision
    ↓
Final Answer
```

---

## نتیجه‌گیری

سیستم RAG پرشین وی یک سیستم پیشرفته و بهینه برای پاسخ‌دهی بر اساس پایگاه دانش است که با استفاده از:

- **Query Expansion** برای بهبود جستجو
- **Multi-Query Search** برای coverage بهتر
- **MMR** برای تنوع نتایج
- **Multi-Factor Confidence** برای تصمیم‌گیری دقیق
- **Context Awareness** برای درک بهتر سوالات

پاسخ‌های دقیق و مرتبط تولید می‌کند.

---

**تاریخ ایجاد:** 2024  
**نسخه:** 1.0  
**نگهدارنده:** تیم توسعه پرشین وی

