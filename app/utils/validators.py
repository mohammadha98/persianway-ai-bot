def validate_rag_settings(settings: dict) -> dict:
    validated = settings.copy()
    changes = []
    def rec(key, val):
        changes.append(f"{key}: {val}")
    if 'top_k_results' in validated:
        original = validated['top_k_results']
        validated['top_k_results'] = max(1, min(int(original), 20))
        if validated['top_k_results'] != original:
            rec('top_k_results', f"{original} → {validated['top_k_results']}")
    if 'temperature' in validated:
        original = validated['temperature']
        validated['temperature'] = max(0.0, min(float(original), 2.0))
        if abs(validated['temperature'] - original) > 0.01:
            rec('temperature', f"{original} → {validated['temperature']}")
    if 'similarity_threshold' in validated:
        original = validated['similarity_threshold']
        validated['similarity_threshold'] = max(0.0, float(original))
        if abs(validated['similarity_threshold'] - original) > 0.01:
            rec('similarity_threshold', f"{original} → {validated['similarity_threshold']}")
    if 'knowledge_base_confidence_threshold' in validated:
        original = validated['knowledge_base_confidence_threshold']
        validated['knowledge_base_confidence_threshold'] = max(0.0, min(float(original), 1.0))
        if abs(validated['knowledge_base_confidence_threshold'] - original) > 0.01:
            rec('confidence_threshold', f"{original} → {validated['knowledge_base_confidence_threshold']}")
    for weight_key in ['original_query_weight', 'expanded_query_weight']:
        if weight_key in validated:
            original = validated[weight_key]
            validated[weight_key] = max(0.1, float(original))
            if abs(validated[weight_key] - original) > 0.01:
                rec(weight_key, f"{original} → {validated[weight_key]}")
    if 'mmr_diversity_score' in validated:
        original = validated['mmr_diversity_score']
        validated['mmr_diversity_score'] = max(0.0, min(float(original), 1.0))
        if abs(validated['mmr_diversity_score'] - original) > 0.01:
            rec('mmr_diversity_score', f"{original} → {validated['mmr_diversity_score']}")
    if 'fetch_k_multiplier' in validated:
        original = validated['fetch_k_multiplier']
        validated['fetch_k_multiplier'] = max(1, int(original))
        if validated['fetch_k_multiplier'] != original:
            rec('fetch_k_multiplier', f"{original} → {validated['fetch_k_multiplier']}")
    if 'reranker_alpha' in validated:
        original = validated['reranker_alpha']
        validated['reranker_alpha'] = max(0.0, min(float(original), 1.0))
        if abs(validated['reranker_alpha'] - original) > 0.01:
            rec('reranker_alpha', f"{original} → {validated['reranker_alpha']}")
    return validated
