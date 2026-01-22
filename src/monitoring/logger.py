"""Structured logging."""
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pythonjsonlogger import jsonlogger
from src.config import settings


class StructuredLogger:
    """Structured JSON logger."""
    
    def __init__(self, name: str = "contract_chatbot"):
        """Initialize logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, settings.log_level))
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Create JSON formatter
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_request(self, query: str, latency_ms: float, success: bool, 
                   error: Optional[str] = None, **kwargs):
        """Log API request."""
        log_data = {
            'event': 'api_request',
            'query': query[:200],  # Truncate long queries
            'latency_ms': latency_ms,
            'success': success,
            **kwargs
        }
        
        if error:
            log_data['error'] = error
            self.logger.error(json.dumps(log_data))
        else:
            self.logger.info(json.dumps(log_data))
    
    def log_query(self, query: str, answer: str, sources: list, latency_ms: float):
        """Log query and response."""
        log_data = {
            'event': 'query',
            'query': query[:200],
            'answer_length': len(answer),
            'num_sources': len(sources),
            'latency_ms': latency_ms
        }
        self.logger.info(json.dumps(log_data))
    
    def log_error(self, error: str, context: Optional[Dict[str, Any]] = None):
        """Log error."""
        log_data = {
            'event': 'error',
            'error': str(error),
            'context': context or {}
        }
        self.logger.error(json.dumps(log_data))
    
    def log_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Log metric."""
        log_data = {
            'event': 'metric',
            'metric_name': metric_name,
            'value': value,
            'tags': tags or {}
        }
        self.logger.info(json.dumps(log_data))


# Global logger instance
logger = StructuredLogger()
