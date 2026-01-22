"""API optimization utilities."""
from typing import List, Dict, Any, Optional
from functools import wraps
import time
import hashlib
import json


def track_latency(func):
    """Decorator to track function latency."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        return result, latency
    return wrapper


def generate_cache_key(query: str, context: Optional[str] = None) -> str:
    """Generate cache key from query and optional context."""
    key_data = {'query': query}
    if context:
        key_data['context'] = context[:500]  # Limit context length
    
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()


def optimize_prompt(prompt: str, max_length: Optional[int] = None) -> str:
    """Optimize prompt by removing unnecessary whitespace."""
    # Remove extra whitespace
    optimized = ' '.join(prompt.split())
    
    # Truncate if max_length specified
    if max_length and len(optimized) > max_length:
        optimized = optimized[:max_length] + "..."
    
    return optimized


def batch_queries(queries: List[str], batch_size: int = 10) -> List[List[str]]:
    """Batch queries for processing."""
    batches = []
    for i in range(0, len(queries), batch_size):
        batches.append(queries[i:i+batch_size])
    return batches
