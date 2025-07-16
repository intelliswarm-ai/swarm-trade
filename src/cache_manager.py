"""
Simple caching manager to reduce API latency
"""

import time
import json
from typing import Dict, Any, Optional
from functools import wraps


class SimpleCache:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry['expires']:
                return entry['value']
            else:
                # Remove expired entry
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        if ttl is None:
            ttl = self.default_ttl
        
        self.cache[key] = {
            'value': value,
            'expires': time.time() + ttl,
            'created': time.time()
        }
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time >= entry['expires']
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        expired_count = sum(
            1 for entry in self.cache.values()
            if current_time >= entry['expires']
        )
        
        return {
            'total_entries': len(self.cache),
            'expired_entries': expired_count,
            'active_entries': len(self.cache) - expired_count,
            'cache_size_mb': self._estimate_size() / (1024 * 1024)
        }
    
    def _estimate_size(self) -> int:
        """Estimate cache size in bytes"""
        try:
            return len(json.dumps(self.cache, default=str).encode('utf-8'))
        except:
            return 0


# Global cache instances
market_data_cache = SimpleCache(default_ttl=300)  # 5 minutes for market data
news_cache = SimpleCache(default_ttl=1800)        # 30 minutes for news
company_info_cache = SimpleCache(default_ttl=3600) # 1 hour for company info


def cached(cache_instance: SimpleCache, ttl: Optional[int] = None, key_func=None):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


def cache_key_for_stock_data(symbol: str, period: str = "1y", **kwargs) -> str:
    """Generate cache key for stock data"""
    return f"stock_data:{symbol}:{period}"


def cache_key_for_news(symbol: str, days: int = 7, **kwargs) -> str:
    """Generate cache key for news data"""
    return f"news:{symbol}:{days}"


def cache_key_for_company_info(symbol: str, **kwargs) -> str:
    """Generate cache key for company info"""
    return f"company_info:{symbol}"


# Convenience decorators for common use cases
def cache_stock_data(ttl: int = 300):
    """Cache stock data for 5 minutes by default"""
    return cached(market_data_cache, ttl, cache_key_for_stock_data)


def cache_news_data(ttl: int = 1800):
    """Cache news data for 30 minutes by default"""
    return cached(news_cache, ttl, cache_key_for_news)


def cache_company_info(ttl: int = 3600):
    """Cache company info for 1 hour by default"""
    return cached(company_info_cache, ttl, cache_key_for_company_info)


def get_cache_status() -> Dict[str, Any]:
    """Get status of all caches"""
    return {
        'market_data_cache': market_data_cache.get_stats(),
        'news_cache': news_cache.get_stats(),
        'company_info_cache': company_info_cache.get_stats()
    }


def clear_all_caches():
    """Clear all caches"""
    market_data_cache.clear()
    news_cache.clear()
    company_info_cache.clear()


def cleanup_all_caches() -> Dict[str, int]:
    """Clean up expired entries in all caches"""
    return {
        'market_data_expired': market_data_cache.cleanup_expired(),
        'news_expired': news_cache.cleanup_expired(),
        'company_info_expired': company_info_cache.cleanup_expired()
    }