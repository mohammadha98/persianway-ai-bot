import pytest
import time
import numpy as np
from typing import List, Dict, Any
import logging
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.chat_service import ChatService
from app.services.knowledge_base import KnowledgeBaseService


class TestReranking:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.chat_service = ChatService()
        self.kb_service = KnowledgeBaseService()
        self.test_results = []
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @pytest.mark.asyncio
    async def test_simple_queries(self):
        queries = [
            "بهترین کود برای گندم چیست؟",
            "شماره تماس پرشین وی",
            "خدمات شرکت چیست؟",
            "محصولات کشاورزی",
        ]

        for query in queries:
            result = await self._run_query_test(query, category="simple")
            self.test_results.append(result)

        avg_conf = np.mean([r['confidence'] for r in self.test_results[-4:] if r.get('success')]) if self.test_results[-4:] else 0.0
        assert avg_conf > 0.7, f"Average confidence too low: {avg_conf}"

    @pytest.mark.asyncio
    async def test_contextual_queries(self):
        conversations = [
            {
                "history": [
                    {"role": "user", "content": "بهترین کود برای گندم چیست؟"},
                    {"role": "assistant", "content": "بهترین کود برای گندم اوره است..."},
                ],
                "query": "چطور استفاده کنم؟",
            },
            {
                "history": [
                    {"role": "user", "content": "محصولات شما چیه؟"},
                    {"role": "assistant", "content": "محصولات ما شامل کودهای..."},
                ],
                "query": "قیمت چقدره؟",
            },
        ]

        for conv in conversations:
            result = await self._run_query_test(
                conv['query'], history=conv['history'], category="contextual"
            )
            self.test_results.append(result)

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        edge_queries = [
            "محصولات",
            "بهترین راه برای افزایش بازدهی تولید گندم در شرایط خشکسالی و کمبود آب با استفاده از کودهای ارگانیک",
            "کشاورزی مدرن تکنولوژی",
            "به نظرت اگه الان کود بدم مشکلی پیش میاد؟",
        ]

        for query in edge_queries:
            result = await self._run_query_test(query, category="edge_case")
            self.test_results.append(result)

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self):
        query = "بهترین کود برای گندم چیست؟"
        times = []

        for _ in range(10):
            start = time.time()
            await self.kb_service.query_knowledge_base(query, [])
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = float(np.mean(times)) if times else 0.0
        self.logger.info(f"Average query time: {avg_time:.3f}s")
        assert avg_time < 2.0, f"Query too slow: {avg_time}s"

    @pytest.mark.asyncio
    async def test_reranking_quality(self):
        query = "کاربرد کود اوره در کشاورزی"
        result = await self.kb_service.query_knowledge_base(query, [])
        assert result['confidence_score'] > 0.0, "No documents retrieved"

    async def _run_query_test(self, query: str, history: List = None, category: str = "general") -> Dict[str, Any]:
        start_time = time.time()
        try:
            resp = await self.chat_service.process_message(
                user_id="test_user",
                message=query,
                conversation_history=history or [],
            )
            elapsed = time.time() - start_time
            qa = resp.get('query_analysis', {})
            return {
                'query': query,
                'category': category,
                'confidence': qa.get('confidence_score', 0.0),
                'source_type': qa.get('knowledge_source'),
                'response_time': elapsed,
                'answer_length': len(resp.get('answer', '')),
                'success': True,
            }
        except Exception as e:
            self.logger.error(f"Query failed: {query} - {e}")
            return {
                'query': query,
                'category': category,
                'confidence': 0.0,
                'error': str(e),
                'success': False,
            }

    def test_generate_report(self):
        report = {
            'total_tests': len(self.test_results),
            'successful': sum(1 for r in self.test_results if r.get('success')),
            'failed': sum(1 for r in self.test_results if not r.get('success')),
            'avg_confidence': float(np.mean([r['confidence'] for r in self.test_results if r.get('success')])) if any(r.get('success') for r in self.test_results) else 0.0,
            'avg_response_time': float(np.mean([r['response_time'] for r in self.test_results if r.get('success')])) if any(r.get('success') for r in self.test_results) else 0.0,
            'by_category': {},
        }

        for category in ['simple', 'contextual', 'edge_case']:
            cat_results = [r for r in self.test_results if r.get('category') == category]
            if cat_results:
                report['by_category'][category] = {
                    'count': len(cat_results),
                    'avg_confidence': float(np.mean([r['confidence'] for r in cat_results])),
                    'success_rate': sum(1 for r in cat_results if r.get('success')) / len(cat_results),
                }

        print("\n" + "=" * 60)
        print("📊 RERANKING TEST REPORT")
        print("=" * 60)
        print(f"Total Tests: {report['total_tests']}")
        print(f"Success Rate: {report['successful']}/{report['total_tests']}")
        print(f"Average Confidence: {report['avg_confidence']:.3f}")
        print(f"Average Response Time: {report['avg_response_time']:.3f}s")
        print("\nBy Category:")
        for cat, stats in report['by_category'].items():
            print(f"  {cat}: Conf={stats['avg_confidence']:.3f}, Success={stats['success_rate']:.1%}")
        print("=" * 60)

        return report

