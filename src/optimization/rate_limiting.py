"""Rate limiting and queuing."""
import time
from collections import deque
from typing import Optional
from threading import Lock
from src.config import settings


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, max_requests: int = 50, time_window: int = 60):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = Lock()
    
    def acquire(self) -> bool:
        """Try to acquire a request slot."""
        with self.lock:
            now = time.time()
            
            # Remove old requests outside time window
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()
            
            # Check if we can make a new request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def wait_time(self) -> float:
        """Get time to wait before next request can be made."""
        with self.lock:
            if len(self.requests) < self.max_requests:
                return 0.0
            
            oldest_request = self.requests[0]
            wait_time = self.time_window - (time.time() - oldest_request)
            return max(0.0, wait_time)
    
    def reset(self):
        """Reset rate limiter."""
        with self.lock:
            self.requests.clear()


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=50, time_window=60)
