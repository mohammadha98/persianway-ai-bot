## تغییرات پیکربندی
- افزودن `reranker_alpha` به `RAGSettings` برای کنترل وزن ترکیب امتیازها.
- تنظیم پیش‌فرض `similarity_threshold` از `1.5` به `1.3`.
- اگر کانفیگ دینامیک موجود در دیتابیس مقدار قدیمی دارد، پس از اعمال کد می‌توان با `ConfigService.update_config({'rag_settings': {'similarity_threshold': 1.3, 'reranker_alpha': 0.8}})` آن را به‌روز کرد یا یکبار `reset_to_defaults()` اجرا شود.

## تغییرات کد
- `app/schemas/config.py`
  - در کلاس `RAGSettings` فیلد جدید اضافه شود: `reranker_alpha: float = Field(default=0.7, ge=0.0, le=1.0)`.
  - مقدار پیش‌فرض `similarity_threshold` از `1.5` به `1.3` تغییر یابد.
- `app/services/knowledge_base.py`
  - در فراخوانی ریرنکر `self.reranker.rerank(...)` مقدار `alpha=rag_settings.reranker_alpha` جایگزین مقدار ثابت شود (حوالی 932–939).
  - لاگ تکمیلی برای اعتماد خام داخل KB: پس از محاسبه `confidence`، یک `logging.debug("[DEBUG] KB raw confidence: %.3f", confidence)` اضافه شود؛ کنار لاگ موجود `[KB Query] Multi-factor confidence score` (حوالی 1024–1034).
- `app/services/chat_service.py`
  - در `process_message` بعد از دریافت `kb_result`، لاگ `logger.debug(f"[DEBUG] KB raw confidence: {kb_confidence:.3f}")` اضافه شود (حوالی 692–699).
  - درست قبل از `return` نهایی، لاگ `logger.debug(f"[DEBUG] Final confidence: {query_analysis['confidence_score']:.3f}")` اضافه شود تا مسیر اعتماد end-to-end قابل ردیابی باشد.

## اجرای تست‌ها
- اجرای تست ساده: `pytest tests/test_reranking_performance.py::TestReranking::test_simple_queries -v -s --log-cli-level=DEBUG`.
- انتظار لاگ‌ها:
  - `[DEBUG] KB raw confidence: X.XX`
  - `[DEBUG] Final confidence: X.XX`
  - بررسی برابری: مقدارها باید یکسان باشند و اختلاف نشان‌دهنده تعدیل در `ChatService` است.

## تحلیل در صورت استمرار اعتماد پایین
- صحت فراخوانی `_calculate_confidence_score` در KB را بررسی کنید: خروجی باید در پاسخ KB به‌صورت `confidence_score` بازتاب یابد (`app/services/knowledge_base.py:1063–1069`).
- بررسی کنید آیا امتیاز بهترین سند واقعاً بالا نیست (L2 پایین) یا مشکل در نرمال‌سازی/وزن‌دهی است:
  - مشاهده `Score range` و لاگ‌های `[KB Query] Top doc ... score`.
- تست مستقیم KB بدون ChatService:
  - `await kb_service.query_knowledge_base(query, [])` و بررسی `result['confidence_score']`.

## بهینه‌سازی‌های عملکرد (پس از تثبیت اعتماد)
- کاهش `top_k_results` موقتاً از 5 به 3.
- کاهش `fetch_k_multiplier` از 3 به 2.
- استفاده از cache موجود برای امبدینگ پرسش در ریرنکر و کاهش تعداد تماس‌ها.

## خروجی مورد انتظار
- مشاهده دو لاگ:
  - `[DEBUG] KB raw confidence: 0.85`
  - `[DEBUG] Final confidence: 0.85`
- در تست‌های ساده، میانگین اعتماد > 0.7 و زمان پاسخ نزدیک به 2s یا کمتر پس از تنظیمات.

لطفاً تأیید کنید تا تغییرات کد اعمال شوند و تست‌ها اجرا شوند.