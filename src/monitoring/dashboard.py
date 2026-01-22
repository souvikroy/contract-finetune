"""Metrics visualization dashboard."""
from typing import Dict, List
import streamlit as st
from src.monitoring.metrics import metrics_collector


def render_metrics_dashboard():
    """Render metrics dashboard in Streamlit."""
    stats = metrics_collector.get_stats()
    
    st.subheader("Performance Metrics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Requests", stats['total_requests'])
    
    with col2:
        st.metric("Error Rate", f"{stats['error_rate_percent']:.2f}%")
    
    with col3:
        st.metric("Avg Latency", f"{stats['avg_latency_ms']:.2f}ms")
    
    with col4:
        st.metric("P95 Latency", f"{stats['p95_latency_ms']:.2f}ms")
    
    # Latency distribution
    if stats['min_latency_ms'] > 0:
        st.subheader("Latency Statistics")
        st.write(f"Min: {stats['min_latency_ms']:.2f}ms | "
                f"Max: {stats['max_latency_ms']:.2f}ms | "
                f"Average: {stats['avg_latency_ms']:.2f}ms")
    
    # Function-level stats
    if stats['function_stats']:
        st.subheader("Function-Level Statistics")
        function_data = []
        for func_name, func_stats in stats['function_stats'].items():
            function_data.append({
                'Function': func_name,
                'Avg Latency (ms)': func_stats['avg_latency'],
                'Error Count': func_stats['error_count']
            })
        
        if function_data:
            import pandas as pd
            df = pd.DataFrame(function_data)
            st.dataframe(df, use_container_width=True)
