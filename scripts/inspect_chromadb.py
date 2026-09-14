"""READ-ONLY ChromaDB inspector (Phase 1 verification).

Prints, for every collection directory under ./vectordb:
  - total document (chunk) count
  - distribution of metadata["entry_type"] values with counts
  - count of documents missing the "entry_type" key
  - same distribution for metadata["is_public"]
  - 3 sample chunk texts per entry_type (first 200 chars each)

No writes. Uses chromadb.PersistentClient in read-only fashion.
"""
import os
import sys
from collections import Counter

import chromadb
from chromadb.config import Settings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECDB_ROOT = os.path.join(PROJECT_ROOT, "vectordb")


def inspect_collection(path: str) -> None:
    name = os.path.basename(path)
    print(f"\n{'=' * 70}")
    print(f"COLLECTION DIR: {name}")
    print(f"{'=' * 70}")
    try:
        client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        collections = client.list_collections()
    except Exception as e:
        print(f"  [ERROR] cannot open client: {e}")
        return
    if not collections:
        print("  [no collections]")
        return
    for col in collections:
        print(f"\n  collection name: {col.name}")
        try:
            count = col.count()
        except Exception as e:
            print(f"    [ERROR] count failed: {e}")
            continue
        print(f"    total chunks: {count}")
        if count == 0:
            continue
        try:
            get_all = col.get(include=["metadatas", "documents"], limit=100000)
        except Exception as e:
            print(f"    [ERROR] get failed: {e}")
            continue
        metadatas = get_all.get("metadatas") or []
        documents = get_all.get("documents") or []

        entry_counter = Counter()
        missing_entry = 0
        public_counter = Counter()
        samples = {}

        for md, doc in zip(metadatas, documents):
            if md is None:
                missing_entry += 1
                continue
            et = md.get("entry_type")
            if et is None:
                missing_entry += 1
            else:
                entry_counter[str(et)] += 1
                if et not in samples:
                    samples[et] = []
                if len(samples[et]) < 3 and doc:
                    samples[et].append(doc[:200])
            pub = md.get("is_public")
            public_counter["<missing>" if pub is None else str(pub)] += 1

        print(f"    entry_type distribution: {dict(entry_counter)}")
        print(f"    chunks missing entry_type: {missing_entry}")
        print(f"    is_public distribution: {dict(public_counter)}")
        for et, texts in samples.items():
            print(f"    --- samples for entry_type={et} ---")
            for i, t in enumerate(texts):
                t = t.replace("\n", " ")
                print(f"      [{i + 1}] {t}")
        # also show a couple of missing-entry_type samples
        miss_samples = []
        for md, doc in zip(metadatas, documents):
            if (md is None or md.get("entry_type") is None) and doc:
                miss_samples.append(doc[:200].replace("\n", " "))
                if len(miss_samples) >= 3:
                    break
        if miss_samples:
            print("    --- samples for chunks missing entry_type ---")
            for i, t in enumerate(miss_samples):
                print(f"      [{i + 1}] {t}")


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else VECDB_ROOT
    if not os.path.isdir(root):
        print(f"vectordb root not found: {root}")
        return 1
    dirs = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
    print(f"vectordb root: {root}")
    print(f"subdirectories: {dirs}")
    for d in dirs:
        inspect_collection(os.path.join(root, d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
