"""Alert configuration and management."""
import time
from typing import Callable, Optional, List
from src.config import settings
from src.monitoring.metrics import metrics_collector


class AlertManager:
    """Manage alerts for system metrics."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alert_handlers: List[Callable] = []
        self.latency_threshold = settings.alert_threshold_latency_ms
    
    def register_handler(self, handler: Callable):
        """Register an alert handler."""
        self.alert_handlers.append(handler)
    
    def check_latency(self, latency_ms: float, function_name: str):
        """Check if latency exceeds threshold."""
        if latency_ms > self.latency_threshold:
            self.trigger_alert(
                'high_latency',
                f"Function {function_name} exceeded latency threshold: {latency_ms:.2f}ms > {self.latency_threshold}ms",
                {'function': function_name, 'latency_ms': latency_ms}
            )
    
    def check_error_rate(self, error_rate: float):
        """Check if error rate is too high."""
        if error_rate > 10.0:  # 10% error rate threshold
            self.trigger_alert(
                'high_error_rate',
                f"Error rate exceeded threshold: {error_rate:.2f}%",
                {'error_rate': error_rate}
            )
    
    def trigger_alert(self, alert_type: str, message: str, context: dict):
        """Trigger an alert."""
        alert_data = {
            'type': alert_type,
            'message': message,
            'context': context,
            'timestamp': time.time()
        }
        
        # Call all registered handlers
        for handler in self.alert_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                print(f"Error in alert handler: {e}")
        
        # Default: print to console
        print(f"ALERT [{alert_type}]: {message}")
    
    def check_metrics(self):
        """Check all metrics and trigger alerts if needed."""
        stats = metrics_collector.get_stats()
        
        # Check error rate
        if stats['error_rate_percent'] > 10.0:
            self.check_error_rate(stats['error_rate_percent'])
        
        # Check average latency
        if stats['avg_latency_ms'] > self.latency_threshold:
            self.trigger_alert(
                'high_avg_latency',
                f"Average latency exceeded threshold: {stats['avg_latency_ms']:.2f}ms",
                {'avg_latency_ms': stats['avg_latency_ms']}
            )


# Global alert manager instance
alert_manager = AlertManager()
