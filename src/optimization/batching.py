"""Request batching logic."""
from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from src.config import settings


class BatchProcessor:
    """Process requests in batches."""
    
    def __init__(self, batch_size: Optional[int] = None, max_workers: int = 5):
        """Initialize batch processor."""
        self.batch_size = batch_size or settings.batch_size
        self.max_workers = max_workers
    
    def process_batch(self, items: List[Any], process_func: Callable) -> List[Any]:
        """Process items in batches."""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i+self.batch_size]
            batch_results = self._process_single_batch(batch, process_func)
            results.extend(batch_results)
        
        return results
    
    def _process_single_batch(self, batch: List[Any], process_func: Callable) -> List[Any]:
        """Process a single batch."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {executor.submit(process_func, item): item for item in batch}
            
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing batch item: {e}")
                    results.append(None)
        
        return results
    
    def process_parallel(self, items: List[Any], process_func: Callable) -> List[Any]:
        """Process items in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {executor.submit(process_func, item): item for item in items}
            
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing item: {e}")
                    results.append(None)
        
        return results
