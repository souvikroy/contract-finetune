"""Evaluation metrics for legal document analysis."""
from typing import List, Dict, Any
import re


class EvaluationMetrics:
    """Calculate evaluation metrics."""
    
    @staticmethod
    def calculate_keyword_match(answer: str, expected_keywords: List[str]) -> float:
        """Calculate keyword match score."""
        answer_lower = answer.lower()
        matched_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
        return matched_keywords / len(expected_keywords) if expected_keywords else 0.0
    
    @staticmethod
    def calculate_answer_length_score(answer: str, min_length: int = 50, max_length: int = 2000) -> float:
        """Calculate score based on answer length."""
        length = len(answer)
        if length < min_length:
            return length / min_length
        elif length > max_length:
            return max(0.0, 1.0 - (length - max_length) / max_length)
        else:
            return 1.0
    
    @staticmethod
    def calculate_relevance_score(answer: str, question: str) -> float:
        """Calculate relevance score based on question-answer overlap."""
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        
        if not question_words:
            return 0.0
        
        overlap = len(question_words.intersection(answer_words))
        return overlap / len(question_words)
    
    @staticmethod
    def evaluate_answer(test_case: Dict[str, Any], answer: str, latency_ms: float) -> Dict[str, Any]:
        """Evaluate an answer against a test case."""
        expected_keywords = test_case.get('expected_keywords', [])
        
        keyword_score = EvaluationMetrics.calculate_keyword_match(answer, expected_keywords)
        length_score = EvaluationMetrics.calculate_answer_length_score(answer)
        relevance_score = EvaluationMetrics.calculate_relevance_score(answer, test_case['question'])
        
        # Combined score (weighted)
        overall_score = (
            keyword_score * 0.4 +
            length_score * 0.2 +
            relevance_score * 0.4
        )
        
        return {
            'test_id': test_case['id'],
            'question': test_case['question'],
            'answer': answer,
            'keyword_score': keyword_score,
            'length_score': length_score,
            'relevance_score': relevance_score,
            'overall_score': overall_score,
            'latency_ms': latency_ms,
            'passed': overall_score >= 0.6  # Threshold for passing
        }
    
    @staticmethod
    def calculate_accuracy(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate accuracy metrics from evaluation results."""
        if not results:
            return {
                'accuracy': 0.0,
                'avg_score': 0.0,
                'avg_latency_ms': 0.0,
                'pass_rate': 0.0
            }
        
        passed = sum(1 for r in results if r.get('passed', False))
        total = len(results)
        avg_score = sum(r.get('overall_score', 0.0) for r in results) / total
        avg_latency = sum(r.get('latency_ms', 0.0) for r in results) / total
        
        return {
            'accuracy': passed / total,
            'avg_score': avg_score,
            'avg_latency_ms': avg_latency,
            'pass_rate': passed / total,
            'total_tests': total,
            'passed_tests': passed
        }
