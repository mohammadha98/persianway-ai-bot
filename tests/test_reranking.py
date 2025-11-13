import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain.schema import Document


def test_reranking_query_selection():
    all_queries = [
        "پرشین وی چیست",
        "شرکت پرشین وی Persian Way",
        "محصولات پرشین وی",
    ]

    filtered_docs = [
        (Document(page_content="محصولات پرشین وی شامل...", metadata={}), 0.5, all_queries[2], "expanded"),
        (Document(page_content="لیست محصولات...", metadata={}), 0.6, all_queries[2], "expanded"),
        (Document(page_content="پرشین وی یک شرکت...", metadata={}), 0.8, all_queries[0], "original"),
        (Document(page_content="Persian Way company...", metadata={}), 1.2, all_queries[1], "expanded"),
    ]

    expanded_queries_info = []
    for q in all_queries[1:]:
        query_docs = [(doc, s) for doc, s, sq, _ in filtered_docs if sq == q]
        if query_docs:
            avg_score = sum(s for _, s in query_docs) / len(query_docs)
            expanded_queries_info.append((q, avg_score, len(query_docs)))

    expanded_queries_info.sort(key=lambda x: x[1])
    best_expanded = expanded_queries_info[0][0] if expanded_queries_info else ""

    assert best_expanded == all_queries[2], f"Expected '{all_queries[2]}' but got '{best_expanded}'"
