"""Performance metrics collection."""
from typing import Dict, List, Optional
from collections import defaultdict, deque
import time
from datetime import datetime
from src.config import settings


class MetricsCollector:
    """Collect and track performance metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.latencies: deque = deque(maxlen=1000)
        self.errors: deque = deque(maxlen=1000)
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.function_latencies: Dict[str, List[float]] = defaultdict(list)
        self.function_errors: Dict[str, int] = defaultdict(int)
    
    def record_latency(self, function_name: str, latency_ms: float):
        """Record function latency."""
        self.latencies.append(latency_ms)
        self.total_latency += latency_ms
        self.request_count += 1
        
        # Track per-function latency
        self.function_latencies[function_name].append(latency_ms)
        if len(self.function_latencies[function_name]) > 100:
            self.function_latencies[function_name].pop(0)
        
        # Check alert threshold
        if latency_ms > settings.alert_threshold_latency_ms:
            self._trigger_latency_alert(function_name, latency_ms)
    
    def record_error(self, function_name: str):
        """Record function error."""
        self.errors.append(time.time())
        self.error_count += 1
        self.function_errors[function_name] += 1
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        if not self.latencies:
            avg_latency = 0.0
            min_latency = 0.0
            max_latency = 0.0
            p95_latency = 0.0
        else:
            sorted_latencies = sorted(self.latencies)
            avg_latency = sum(sorted_latencies) / len(sorted_latencies)
            min_latency = min(sorted_latencies)
            max_latency = max(sorted_latencies)
            p95_index = int(len(sorted_latencies) * 0.95)
            p95_latency = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else max_latency
        
        error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0.0
        
        return {
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate_percent': error_rate,
            'avg_latency_ms': avg_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'p95_latency_ms': p95_latency,
            'function_stats': {
                func: {
                    'avg_latency': sum(lats) / len(lats) if lats else 0.0,
                    'error_count': self.function_errors[func]
                }
                for func, lats in self.function_latencies.items()
            }
        }
    
    def _trigger_latency_alert(self, function_name: str, latency_ms: float):
        """Trigger latency alert."""
        # In production, this would send to alerting system
        print(f"ALERT: High latency detected - {function_name}: {latency_ms:.2f}ms")
    
    def reset(self):
        """Reset all metrics."""
        self.latencies.clear()
        self.errors.clear()
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.function_latencies.clear()
        self.function_errors.clear()


# Global metrics collector instance
metrics_collector = MetricsCollector()
