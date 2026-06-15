from typing import List, Dict, Any
import asyncio
import logging
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from app.services.chat_service import get_llm

MODEL_SUMMARIZER = "google/gemma-2-9b-it"

def _build_prompt(query: str, doc_content: str, doc_id: str) -> str:
    return (
        f"سؤال کاربر: «{query}»\n"
        f"متن سند:\n\n" 
        f"\"\"\"{doc_content}\"\"\"\n\n"
        f"دستور:\n"
        f"با توجه به سؤال کاربر، خلاصه‌ای دقیق و فشرده از محتوای مرتبط در متن بالا بنویس.\n"
        f"اگر سند شامل مراحل یا فهرست است، ترتیب آن‌ها را حفظ کن.\n"
        f"پاسخ را در قالب JSON زیر برگردان:\n"
        f"{{ \"source_id\": \"{doc_id}\", \"summary\": \"<خلاصهٔ ۳ تا ۵ جمله‌ای از اطلاعات مرتبط>\" }}"
    )

def _safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.!?\n])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def _fallback_summary(doc_content: str) -> str:
    sentences = _split_sentences(doc_content)
    if not sentences:
        return ""
    return " ".join(sentences[:5])

async def summarize_doc(doc: Any, query: str, model_name: str = None, temperature: float = 0.3, max_tokens: int = 300) -> Dict[str, Any]:
    doc_id = str(getattr(doc, "metadata", {}).get("id") or getattr(doc, "metadata", {}).get("source") or "unknown")
    content = (getattr(doc, "page_content", "") or "").strip()
    llm = await get_llm(model_name=model_name or MODEL_SUMMARIZER, temperature=temperature, max_tokens=max_tokens)
    sys = SystemMessage(content=(
        "یک خلاصه‌ساز دقیق برای متن‌های فارسی و انگلیسی هستی."
        " فقط از محتوای سند و سؤال کاربر استفاده کن."
        " ترتیب مراحل و فهرست‌ها را حفظ کن."
        " خروجی باید JSON معتبر باشد و فقط شامل کلیدهای source_id و summary باشد."
        " طول خلاصه ۳ تا ۵ جمله باشد."
    ))
    prompt = _build_prompt(query, content, doc_id)
    hm = HumanMessage(content=prompt)
    try:
        result = await llm.ainvoke([sys, hm])
        text = result if isinstance(result, str) else getattr(result, "content", "")
        parsed = _safe_parse_json(text)
        summary = parsed.get("summary")
        if not summary:
            summary = _fallback_summary(content)
        return {"source_id": doc_id, "summary": summary}
    except Exception as e:
        logging.warning(f"[Context Condenser] Summarization failed: {str(e)}")
        return {"source_id": doc_id, "summary": _fallback_summary(content)}

async def batch_condense(retrieved_docs: List[Any], query: str, model_name: str = None) -> List[Dict[str, Any]]:
    tasks = [summarize_doc(doc, query, model_name=model_name) for doc in retrieved_docs]
    return await asyncio.gather(*tasks)

