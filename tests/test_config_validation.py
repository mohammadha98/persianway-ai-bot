from app.utils.validators import validate_rag_settings


def test_config_validation():
    invalid = {
        "top_k_results": 50,
        "temperature": 3.0,
        "similarity_threshold": -1.0,
        "knowledge_base_confidence_threshold": 1.5,
        "mmr_diversity_score": 2.0,
        "original_query_weight": -0.5,
    }
    validated = validate_rag_settings(invalid)
    assert validated["top_k_results"] == 20
    assert validated["temperature"] == 2.0
    assert validated["similarity_threshold"] == 0.0
    assert validated["knowledge_base_confidence_threshold"] == 1.0
    assert validated["mmr_diversity_score"] == 1.0
    assert validated["original_query_weight"] == 0.1
