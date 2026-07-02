# OFF_TOPIC Intent - Feature Summary

## خلاصه تغییرات (Persian Summary)

به سیستم تشخیص intent یک دسته‌بندی جدید اضافه شد: **`OFF_TOPIC`**

### هدف
سیستم حالا می‌تونه سوالات بی‌ربط و نامرتبط رو تشخیص بده و به جای اینکه بخواد جواب بده، کاربر رو به موضوعات اصلی هدایت می‌کنه.

### حوزه‌های تخصصی پرشین وی
سیستم فقط در این حوزه‌ها جواب میده:
1. 🏢 **اطلاعات شرکت** - درباره پرشین وی
2. 🌱 **کشاورزی** - کاشت، داشت، کود، آفات، خاک
3. 💊 **سلامت** - تغذیه، ویتامین، محصولات سلامتی
4. 💄 **زیبایی** - پوست، مو، محصولات آرایشی

### سوالات خارج از حوزه (OFF_TOPIC)
سوالاتی که در این دسته قرار می‌گیرن:
- ⚽ **ورزش**: فوتبال، بسکتبال، المپیک
- 🎬 **سرگرمی**: فیلم، موسیقی، سینما
- 💻 **تکنولوژی**: برنامه‌نویسی، کامپیوتر، موبایل
- 💰 **مالی**: قیمت دلار، بورس، بانک
- 🏛️ **سیاست**: انتخابات، دولت، سیاستمداران
- 🏠 **املاک**: آپارتمان، ویلا، رهن

### پاسخ به سوالات OFF_TOPIC

وقتی سوالی خارج از حوزه باشه، سیستم این پیام رو میده:

```
درود! 🌹

متأسفانه این سوال خارج از حوزه تخصص ماست. پرشین وی در حوزه‌های زیر آماده کمک به شماست:

🌱 **کشاورزی**: کاشت، داشت، کود، آبیاری، مبارزه با آفات
💊 **سلامت**: تغذیه، ویتامین‌ها، محصولات سلامتی
💄 **زیبایی**: مراقبت از پوست، محصولات آرایشی و بهداشتی
🏢 **اطلاعات شرکت**: درباره پرشین وی، خدمات و محصولات

چطور می‌تونم در این زمینه‌ها بهتون کمک کنم؟
```

### مهم: سیستم سخت‌گیر نیست! ⚠️

سیستم **با تسامح** عمل می‌کنه:
- فقط سوالاتی که **کاملاً و واضح** بی‌ربط هستند OFF_TOPIC میشن
- اگر کوچکترین احتمالی وجود داشته باشه که سوال مرتبط باشه، سیستم سعی می‌کنه جواب بده
- وقتی تردید هست، سیستم سوال رو PRIVATE طبقه‌بندی می‌کنه و جستجو می‌کنه

---

## Technical Implementation (English)

### Changes Made

1. **Updated Intent Classification**
   - Added `OFF_TOPIC` as the 4th intent category
   - System now has: PUBLIC, PRIVATE, NEEDS_CLARIFICATION, OFF_TOPIC

2. **Enhanced LLM Prompt**
   - Clearly defines PersianWay's three main expertise areas
   - Provides explicit examples for each intent
   - Emphasizes lenient classification (only obvious off-topic questions)

3. **New Response Handler**
   - Handles OFF_TOPIC intent in `process_message()`
   - Sends friendly redirect message
   - Skips knowledge base search (saves resources)
   - Maintains positive, helpful tone

4. **Return Value Updated**
   ```python
   {
       "intent": "OFF_TOPIC",  # New intent type
       "is_public": False,
       "explanation": "Question about sports/politics/etc",
       "off_topic_message": "Custom redirect message in Persian"
   }
   ```

### Example Queries

#### OFF_TOPIC (will get redirect message)
```python
"بهترین تیم فوتبال کدومه؟"        # Sports
"چطور برنامه نویسی یاد بگیرم؟"     # Technology  
"قیمت دلار امروز چقدره؟"           # Finance
"فیلم خوب پیشنهاد بده"             # Entertainment
"نظرت درباره انتخابات چیه؟"       # Politics
```

#### PRIVATE (will search knowledge base)
```python
"بهترین کود برای گندم؟"            # Agriculture
"چه ویتامینی برای پوست خوبه؟"     # Health
"کرم ضد آفتاب معرفی کن"            # Beauty
"خدمات شما چیه؟"                   # Services
```

### Configuration

**LLM Settings for Intent Classification:**
- Model: `gpt-4o-mini`
- Temperature: `0.1`
- Top P: `0.1`

**Response Settings:**
- Confidence Score: `0.3` (for OFF_TOPIC)
- Knowledge Source: `"off_topic_redirect"`
- Requires Human Referral: `False`

### Key Design Decisions

1. **Lenient Approach**
   - Only classify as OFF_TOPIC if question is CLEARLY unrelated
   - Default to PRIVATE when in doubt
   - Prevents false rejections of potentially relevant questions

2. **Friendly Tone**
   - Uses polite Persian language
   - Explains what we DO cover instead of just saying "no"
   - Invites user to ask relevant questions
   - Maintains positive brand image

3. **Resource Optimization**
   - Skips expensive vector DB searches
   - Fast response time (no KB query needed)
   - Reduces LLM token usage

4. **Clear Boundaries**
   - Establishes scope of service
   - Sets user expectations
   - Guides users to relevant topics

### Files Modified

1. **`app/services/chat_service.py`**
   - Updated `detect_query_intent()` docstring and return type
   - Enhanced classifier prompt with OFF_TOPIC examples
   - Added OFF_TOPIC handler in `process_message()`
   - Updated return value validation

2. **`test_intent_clarification.py`**
   - Added OFF_TOPIC test cases
   - Updated test output to show off_topic_message

3. **`CLARIFICATION_INTENT_FEATURE.md`**
   - Updated documentation with OFF_TOPIC information
   - Added examples and use cases
   - Updated process flow diagram

### Testing

Run the test script:
```bash
python test_intent_clarification.py
```

Expected results:
- Sports/entertainment/politics → OFF_TOPIC
- Agriculture/health/beauty → PRIVATE
- Company questions → PUBLIC
- Vague questions → NEEDS_CLARIFICATION

### Benefits

1. ✅ **Better UX**: Clear guidance on what we cover
2. ✅ **Cost Savings**: No wasted KB searches on irrelevant topics
3. ✅ **Brand Clarity**: Establishes expertise areas
4. ✅ **User Guidance**: Redirects to relevant topics
5. ✅ **Positive Tone**: Friendly, not rejecting

---

## Version History

- **v1.1** (2025-11-10): Added OFF_TOPIC intent
- **v1.0** (2025-11-10): Initial release with NEEDS_CLARIFICATION

---

## Quick Reference

### 4 Intent Types

| Intent | Description | Response |
|--------|-------------|----------|
| **PUBLIC** | Company info questions | Search public docs |
| **PRIVATE** | Agriculture/health/beauty | Search all docs |
| **NEEDS_CLARIFICATION** | Unclear questions | Ask for details |
| **OFF_TOPIC** | Unrelated questions | Friendly redirect |

### Decision Tree

```
Is question clear? 
├─ No → NEEDS_CLARIFICATION
└─ Yes → Is it about our expertise areas?
          ├─ Clearly NO → OFF_TOPIC
          └─ Yes/Maybe → Is it about the company?
                         ├─ Yes → PUBLIC
                         └─ No → PRIVATE
```

---

**Status:** ✅ Implemented and Ready  
**Author:** PersianWay AI Team  
**Date:** 2025-11-10

