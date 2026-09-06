import json
import os
import sys
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.services.document_processor import DocumentProcessor

def run():
    dp = DocumentProcessor()
    vs = dp.get_vector_store()
    if vs is None:
        print(json.dumps({"error": "vector_store_not_available"}, ensure_ascii=False))
        return
    data = vs._collection.get()
    metas = data.get("metadatas") or []
    types = [m.get("entry_type") for m in metas if isinstance(m, dict)]
    existing = list(set(types))
    counts = dict(Counter(types))
    print(json.dumps({"existing_types": existing, "counts": counts}, ensure_ascii=False))

if __name__ == "__main__":
    run()