import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.services.config_service import ConfigService


@pytest.mark.asyncio
async def test_weighted_search_normalization():
    service = ConfigService()
    await service._load_config()
    rag_settings = await service.get_rag_settings()

    all_queries = [
        "query اصلی",
        "expanded query 1",
        "expanded query 2",
        "expanded query 3",
    ]

    num_expanded = len(all_queries) - 1
    total_weight = rag_settings.original_query_weight + (
        rag_settings.expanded_query_weight * num_expanded
    )

    weights_sum = 0.0
    for i, _ in enumerate(all_queries):
        if i == 0:
            normalized_weight = rag_settings.original_query_weight / total_weight
        else:
            normalized_weight = rag_settings.expanded_query_weight / total_weight
        weights_sum += normalized_weight

    assert abs(weights_sum - 1.0) < 1e-4, f"Normalized weights sum to {weights_sum}, expected 1.0"
