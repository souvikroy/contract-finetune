"""Evaluation metrics display."""
import streamlit as st
from src.monitoring.metrics import metrics_collector
from src.monitoring.dashboard import render_metrics_dashboard


def render_evaluation_dashboard():
    """Render evaluation metrics dashboard."""
    st.header("📊 Performance Metrics")
    
    # Render metrics dashboard
    render_metrics_dashboard()
    
    # Additional evaluation metrics
    st.subheader("System Health")
    stats = metrics_collector.get_stats()
    
    # Health indicators
    col1, col2 = st.columns(2)
    
    with col1:
        if stats['error_rate_percent'] < 5.0:
            st.success(f"✅ Error Rate: {stats['error_rate_percent']:.2f}%")
        elif stats['error_rate_percent'] < 10.0:
            st.warning(f"⚠️ Error Rate: {stats['error_rate_percent']:.2f}%")
        else:
            st.error(f"❌ Error Rate: {stats['error_rate_percent']:.2f}%")
    
    with col2:
        if stats['avg_latency_ms'] < 200:
            st.success(f"✅ Avg Latency: {stats['avg_latency_ms']:.2f}ms")
        else:
            st.warning(f"⚠️ Avg Latency: {stats['avg_latency_ms']:.2f}ms")
