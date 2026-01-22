"""Latency tracking utilities."""
import time
from functools import wraps
from typing import Callable, Any
from src.monitoring.metrics import MetricsCollector


def track_latency(func: Callable) -> Callable:
    """Decorator to track function latency."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000
            
            # Record metric if metrics collector is available
            try:
                metrics = MetricsCollector()
                metrics.record_latency(func.__name__, latency_ms)
            except:
                pass
            
            return result
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            try:
                metrics = MetricsCollector()
                metrics.record_error(func.__name__)
            except:
                pass
            raise e
    
    return wrapper


class LatencyTracker:
    """Track latency for operations."""
    
    def __init__(self):
        """Initialize latency tracker."""
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start tracking."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """Stop tracking and return latency in milliseconds."""
        if self.start_time is None:
            return 0.0
        
        self.end_time = time.time()
        latency_ms = (self.end_time - self.start_time) * 1000
        return latency_ms
    
    def reset(self):
        """Reset tracker."""
        self.start_time = None
        self.end_time = None
