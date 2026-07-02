# Fix: Query Rewriting Race Condition

## مشکل (Problem Description)

گاهی اوقات وقتی کاربر پیام جدیدی می‌فرستاد، سیستم به پیام قبلی جواب می‌داد به جای پیام جدید.

**مثال:**
1. کاربر: "بهترین کود برای گندم چیه؟" → سیستم پاسخ درست میده ✅
2. کاربر: "قیمت دلار امروز چنده؟" → سیستم درباره **کود گندم** جواب میده ❌

## علت مشکل (Root Cause)

### 1. شامل شدن پیام فعلی در History

در متد `query_knowledge_base()` در فایل `knowledge_base.py`:

```python
extracted_history = self._extract_conversation_history(conversation_history)
if extracted_history:
    query = await self.rewrite_query_with_context(extracted_history, query)
```

**مشکل**: اگر `conversation_history` که از API میاد شامل پیام فعلی کاربر هم باشه، سیستم اون رو هم توی context قرار میده و LLM گیج میشه!

```
History شامل: 
  - user: "بهترین کود برای گندم چیه؟"
  - assistant: "..."
  - user: "قیمت دلار امروز چنده؟"  ← پیام فعلی که نباید در history باشه!

Query: "قیمت دلار امروز چنده؟"
```

وقتی LLM می‌بینه پیام فعلی هم در history هست، ممکنه به جای اینکه اون رو standalone query بکنه، با context قبلی مرتبطش کنه.

### 2. استفاده از تاریخچه طولانی

- بدون محدودیت در تعداد message‌های history
- بدون truncation برای message‌های خیلی بلند
- احتمال token overflow و confusion

## راه‌حل (Solution)

### 1. فیلتر کردن پیام فعلی از History

```python
if extracted_history:
    # Remove the current user message from history if it exists
    filtered_history = []
    for msg in extracted_history:
        # Skip if this message matches the current query
        if msg.get('role') == 'user' and msg.get('content', '').strip() == query.strip():
            logging.info(f"Skipping current query from history: '{query[:50]}...'")
            continue
        filtered_history.append(msg)
    
    # Only rewrite if we have actual previous context
    if filtered_history:
        query = await self.rewrite_query_with_context(filtered_history, query)
```

### 2. محدود کردن تعداد Message‌های History

در `rewrite_query_with_context()`:

```python
# Take only the most recent messages (max_history=4 = last 2 exchanges)
recent_history = history[-max_history:] if len(history) > max_history else history

# Truncate very long messages to avoid token limit
truncated_history = []
for msg in recent_history:
    content = msg.get('content', '')
    if len(content) > 300:
        content = content[:300] + "..."
    truncated_history.append({
        'role': msg['role'],
        'content': content
    })
```

### 3. اضافه کردن Logging برای Debug

```python
logging.info(f"[KB Query] Original query: '{query[:100]}...'")
logging.debug(f"[KB Query] Extracted {len(extracted_history)} messages from conversation history")
logging.debug(f"[KB Query] After filtering: {len(filtered_history)} messages remain for context")
logging.info(f"[KB Query] Query rewritten: '{original_query[:50]}...' -> '{query[:100]}...'")
```

## تغییرات (Changes Made)

### File: `app/services/knowledge_base.py`

#### 1. در متد `query_knowledge_base()` (خطوط 642-670):
- ✅ اضافه شدن فیلتر برای حذف پیام فعلی از history
- ✅ اضافه شدن logging برای debug
- ✅ بررسی اینکه بعد از filter کردن، واقعاً context قبلی وجود داره

#### 2. در متد `rewrite_query_with_context()` (خطوط 555-593):
- ✅ تغییر `max_history` از 3 به 4 (2 exchange کامل)
- ✅ محدود کردن به آخرین N message
- ✅ truncate کردن message‌های خیلی بلند (>300 کاراکتر)
- ✅ بهبود documentation

## نحوه تست (How to Test)

### تست دستی:

1. پیام اول را بفرستید:
   ```
   "بهترین کود برای گندم چیه؟"
   ```
   - سیستم باید جواب درست بده ✅

2. بلافاصله پیام دوم (نامرتبط) را بفرستید:
   ```
   "قیمت دلار امروز چنده؟"
   ```
   - سیستم باید تشخیص بده که OFF_TOPIC است ✅
   - **نباید** درباره کود گندم صحبت کنه ❌

3. لاگ‌ها را چک کنید:
   ```
   [KB Query] Original query: 'قیمت دلار امروز چنده؟'
   [KB Query] Extracted X messages from conversation history
   [KB Query] Skipping current query from history: 'قیمت دلار...'
   [KB Query] After filtering: Y messages remain for context
   ```

### تست Contextual (برای اطمینان از عملکرد صحیح context):

1. پیام اول:
   ```
   "بهترین کود برای گندم چیه؟"
   ```

2. پیام دوم (مرتبط با اولی):
   ```
   "چه زمانی باید بزنم؟"
   ```
   - سیستم باید context را درک کنه و درباره **زمان زدن کود برای گندم** صحبت کنه ✅

## مزایا (Benefits)

1. ✅ **رفع مشکل Race Condition**: پیام فعلی در context خودش استفاده نمیشه
2. ✅ **بهبود دقت**: Query rewriting فقط با context واقعی قبلی انجام میشه
3. ✅ **کاهش Token Usage**: محدود کردن تعداد و طول message‌ها
4. ✅ **بهتر شدن Debug**: logging های دقیق برای trace کردن مشکلات
5. ✅ **جلوگیری از Confusion**: LLM با context واضح‌تر کار می‌کنه

## نکات مهم (Important Notes)

### ⚠️ فرانت‌اند باید دقت کنه:

API هنگام ارسال پیام جدید نباید همان پیام را در `conversation_history` بفرستد.

**درست:**
```json
{
  "message": "قیمت دلار چنده؟",
  "conversation_history": [
    {"role": "user", "content": "بهترین کود برای گندم چیه؟"},
    {"role": "assistant", "content": "..."}
  ]
}
```

**غلط:**
```json
{
  "message": "قیمت دلار چنده؟",
  "conversation_history": [
    {"role": "user", "content": "بهترین کود برای گندم چیه؟"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "قیمت دلار چنده؟"}  ← ❌ نباید باشه!
  ]
}
```

اگرچه ما الان server-side این مشکل رو handle می‌کنیم، ولی بهتره فرانت‌اند هم درست عمل کنه.

### 📊 Context Window Size:

- `max_history=4`: یعنی آخرین 4 message (2 user + 2 assistant)
- این کافیه برای contextual queries مثل "دیگه چی؟" یا "چطور بزنم؟"
- اگه conversation خیلی بلند بشه، فقط آخرین exchanges استفاده میشه

## Backward Compatibility

✅ این تغییرات **backward compatible** هستند:
- اگر history از قبل درست بود، همچنان کار می‌کنه
- اگه history شامل current message بود، حالا filter میشه
- اگه history خیلی بلند باشه، truncate میشه

## Future Improvements

برای آینده می‌تونیم:

1. **Session Locking**: برای جلوگیری از race condition در concurrent requests
2. **Message ID Tracking**: استفاده از unique ID برای هر message
3. **Timestamp Validation**: بررسی اینکه message‌های history قبل از message فعلی باشند
4. **Client-Side Validation**: اضافه کردن validation در فرانت‌اند

---

**Status:** ✅ Implemented and Tested  
**Date:** 2025-11-10  
**Version:** 1.1

