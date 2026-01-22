"""Benchmark test suite."""
from typing import List, Dict, Any
import time
from src.evaluation.test_cases import get_test_cases
from src.evaluation.metrics import EvaluationMetrics
from src.llm.chain import RAGChain
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import VectorStore


class BenchmarkSuite:
    """Run benchmark tests."""
    
    def __init__(self, rag_chain: RAGChain):
        """Initialize benchmark suite."""
        self.rag_chain = rag_chain
        self.metrics = EvaluationMetrics()
    
    def run_benchmark(self, test_cases: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run benchmark tests."""
        if test_cases is None:
            test_cases = get_test_cases()
        
        results = []
        
        for test_case in test_cases:
            question = test_case['question']
            
            # Measure latency
            start_time = time.time()
            try:
                result = self.rag_chain.query(question, n_context=5)
                latency_ms = (time.time() - start_time) * 1000
                
                answer = result.get('answer', '')
                evaluation = self.metrics.evaluate_answer(test_case, answer, latency_ms)
                results.append(evaluation)
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                results.append({
                    'test_id': test_case['id'],
                    'question': question,
                    'answer': f"Error: {str(e)}",
                    'overall_score': 0.0,
                    'latency_ms': latency_ms,
                    'passed': False,
                    'error': str(e)
                })
        
        # Calculate overall metrics
        overall_metrics = self.metrics.calculate_accuracy(results)
        
        return {
            'results': results,
            'metrics': overall_metrics,
            'test_cases_run': len(test_cases)
        }
    
    def run_latency_benchmark(self, num_queries: int = 10) -> Dict[str, Any]:
        """Run latency benchmark."""
        test_cases = get_test_cases()[:num_queries]
        latencies = []
        
        for test_case in test_cases:
            question = test_case['question']
            start_time = time.time()
            
            try:
                self.rag_chain.query(question, n_context=5)
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
        
        if latencies:
            return {
                'avg_latency_ms': sum(latencies) / len(latencies),
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0
            }
        else:
            return {
                'avg_latency_ms': 0.0,
                'min_latency_ms': 0.0,
                'max_latency_ms': 0.0,
                'p95_latency_ms': 0.0
            }
