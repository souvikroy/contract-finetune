"""Multi-level caching system."""
from typing import Optional, Dict, Any
import json
import hashlib
import time
from src.config import settings

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class Cache:
    """Multi-level cache (memory + optional Redis)."""
    
    def __init__(self):
        """Initialize cache."""
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.redis_client = None
        
        if settings.redis_enabled and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
            except Exception as e:
                print(f"Redis connection failed: {e}. Using memory cache only.")
                self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Try Redis first if available
        if self.redis_client:
            try:
                cached = self.redis_client.get(key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                print(f"Redis get error: {e}")
        
        # Fall back to memory cache
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            # Check expiration
            if entry.get('expires_at', float('inf')) > time.time():
                return entry.get('value')
            else:
                del self.memory_cache[key]
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL."""
        cache_entry = {
            'value': value,
            'expires_at': time.time() + ttl
        }
        
        # Store in Redis if available
        if self.redis_client:
            try:
                self.redis_client.setex(
                    key,
                    ttl,
                    json.dumps(value)
                )
            except Exception as e:
                print(f"Redis set error: {e}")
        
        # Also store in memory cache
        self.memory_cache[key] = cache_entry
    
    def delete(self, key: str):
        """Delete key from cache."""
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                print(f"Redis delete error: {e}")
        
        if key in self.memory_cache:
            del self.memory_cache[key]
    
    def clear(self):
        """Clear all cache."""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                print(f"Redis clear error: {e}")
        
        self.memory_cache.clear()
    
    def generate_key(self, query: str, context: Optional[str] = None) -> str:
        """Generate cache key."""
        key_data = {'query': query}
        if context:
            key_data['context'] = context[:500]
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


# Global cache instance
cache = Cache()
