"""Results analysis for benchmarks."""
import json
from typing import Dict, Any, List
import pandas as pd


def analyze_results(results_file: str) -> Dict[str, Any]:
    """Analyze benchmark results from file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract metrics
    metrics = results.get('metrics', {})
    test_results = results.get('results', [])
    
    # Create DataFrame for analysis
    df = pd.DataFrame(test_results)
    
    analysis = {
        'overall_accuracy': metrics.get('accuracy', 0.0),
        'average_score': metrics.get('avg_score', 0.0),
        'average_latency_ms': metrics.get('avg_latency_ms', 0.0),
        'pass_rate': metrics.get('pass_rate', 0.0),
        'total_tests': metrics.get('total_tests', 0),
        'passed_tests': metrics.get('passed_tests', 0),
        'score_distribution': {
            'min': df['overall_score'].min() if 'overall_score' in df.columns else 0.0,
            'max': df['overall_score'].max() if 'overall_score' in df.columns else 0.0,
            'mean': df['overall_score'].mean() if 'overall_score' in df.columns else 0.0,
            'median': df['overall_score'].median() if 'overall_score' in df.columns else 0.0
        },
        'latency_distribution': {
            'min': df['latency_ms'].min() if 'latency_ms' in df.columns else 0.0,
            'max': df['latency_ms'].max() if 'latency_ms' in df.columns else 0.0,
            'mean': df['latency_ms'].mean() if 'latency_ms' in df.columns else 0.0,
            'median': df['latency_ms'].median() if 'latency_ms' in df.columns else 0.0
        }
    }
    
    return analysis


def generate_report(analysis: Dict[str, Any]) -> str:
    """Generate text report from analysis."""
    report = "="*60 + "\n"
    report += "BENCHMARK RESULTS ANALYSIS\n"
    report += "="*60 + "\n\n"
    
    report += "Overall Metrics:\n"
    report += f"  Accuracy: {analysis['overall_accuracy']:.2%}\n"
    report += f"  Average Score: {analysis['average_score']:.2f}\n"
    report += f"  Pass Rate: {analysis['pass_rate']:.2%}\n"
    report += f"  Tests Passed: {analysis['passed_tests']}/{analysis['total_tests']}\n\n"
    
    report += "Score Distribution:\n"
    score_dist = analysis['score_distribution']
    report += f"  Min: {score_dist['min']:.2f}\n"
    report += f"  Max: {score_dist['max']:.2f}\n"
    report += f"  Mean: {score_dist['mean']:.2f}\n"
    report += f"  Median: {score_dist['median']:.2f}\n\n"
    
    report += "Latency Distribution:\n"
    latency_dist = analysis['latency_distribution']
    report += f"  Min: {latency_dist['min']:.2f}ms\n"
    report += f"  Max: {latency_dist['max']:.2f}ms\n"
    report += f"  Mean: {latency_dist['mean']:.2f}ms\n"
    report += f"  Median: {latency_dist['median']:.2f}ms\n"
    
    return report
