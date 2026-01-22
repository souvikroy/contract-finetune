"""Main Streamlit application."""
import streamlit as st
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ui.chat_interface import ChatInterface
from src.ui.document_viewer import DocumentViewer
from src.ui.metrics_dashboard import render_evaluation_dashboard
from src.ui.components import render_query_suggestions
from src.llm.chain import RAGChain
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import VectorStore
from src.optimization.cache import cache
from src.monitoring.logger import logger
from src.monitoring.metrics import metrics_collector
from src.utils.timing import LatencyTracker
from src.config import settings


# Initialize components
@st.cache_resource
def initialize_rag_chain():
    """Initialize RAG chain (cached)."""
    try:
        vector_store = VectorStore()
        retriever = HybridRetriever(vector_store)
        rag_chain = RAGChain(retriever)
        return rag_chain
    except Exception as e:
        st.error(f"Error initializing RAG chain: {e}")
        return None


def main():
    """Main application."""
    st.set_page_config(
        page_title="Legal Contract Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize components
    rag_chain = initialize_rag_chain()
    chat_interface = ChatInterface()
    document_viewer = DocumentViewer()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Query suggestions
        suggestions = [
            "What are the key obligations of the contractor?",
            "What is the defect liability period?",
            "What are the payment terms?",
            "What happens in case of termination?",
            "What are the performance security requirements?"
        ]
        
        selected_suggestion = render_query_suggestions(suggestions)
        if selected_suggestion:
            st.session_state.user_query = selected_suggestion
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            chat_interface.clear_history()
            st.rerun()
        
        # Export chat button
        if st.button("💾 Export Chat History"):
            history_text = chat_interface.export_history()
            st.download_button(
                label="Download History",
                data=history_text,
                file_name="chat_history.txt",
                mime="text/plain"
            )
        
        st.divider()
        
        # Metrics link
        if st.button("📊 View Metrics"):
            st.session_state.show_metrics = True
    
    # Main content area
    if st.session_state.get('show_metrics', False):
        render_evaluation_dashboard()
        if st.button("← Back to Chat"):
            st.session_state.show_metrics = False
            st.rerun()
    else:
        # Chat interface
        chat_interface.render()
        
        # Get user input
        user_query = chat_interface.get_user_input() or st.session_state.get('user_query')
        
        if user_query and rag_chain:
            # Clear the query from session state
            if 'user_query' in st.session_state:
                del st.session_state.user_query
            
            # Add user message
            chat_interface.add_message('user', user_query)
            
            # Check cache first
            cache_key = cache.generate_key(user_query)
            cached_response = cache.get(cache_key)
            
            if cached_response and settings.cache_enabled:
                # Use cached response
                answer = cached_response.get('answer', '')
                sources = cached_response.get('sources', [])
                latency_ms = cached_response.get('latency_ms', 0)
                
                chat_interface.add_message('assistant', answer, sources)
                st.rerun()
            else:
                # Process query
                with st.spinner("Thinking..."):
                    tracker = LatencyTracker()
                    tracker.start()
                    
                    try:
                        # Query RAG chain
                        result = rag_chain.query(user_query, n_context=5)
                        
                        latency_ms = tracker.stop()
                        
                        answer = result.get('answer', '')
                        sources = result.get('sources', [])
                        
                        # Cache the response
                        if settings.cache_enabled:
                            cache.set(cache_key, {
                                'answer': answer,
                                'sources': sources,
                                'latency_ms': latency_ms
                            }, ttl=3600)
                        
                        # Log metrics
                        metrics_collector.record_latency('rag_query', latency_ms)
                        logger.log_query(user_query, answer, sources, latency_ms)
                        
                        # Add assistant message
                        chat_interface.add_message('assistant', answer, sources)
                        
                        # Show sources
                        if sources:
                            document_viewer.render_source_documents(sources)
                        
                        st.rerun()
                    
                    except Exception as e:
                        latency_ms = tracker.stop()
                        error_msg = str(e)
                        
                        metrics_collector.record_error('rag_query')
                        logger.log_error(error_msg, {'query': user_query})
                        
                        st.error(f"Error processing query: {error_msg}")
                        chat_interface.add_message('assistant', f"I encountered an error: {error_msg}")
                        st.rerun()
        
        elif user_query and not rag_chain:
            st.error("RAG chain not initialized. Please check the logs.")
        
        # Show initial message
        if not st.session_state.messages:
            st.info("👋 Welcome! Ask me anything about the contract document.")
    
    # Footer with metrics summary
    with st.container():
        st.divider()
        stats = metrics_collector.get_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"Total Requests: {stats['total_requests']}")
        with col2:
            st.caption(f"Avg Latency: {stats['avg_latency_ms']:.2f}ms")
        with col3:
            st.caption(f"Error Rate: {stats['error_rate_percent']:.2f}%")


if __name__ == "__main__":
    main()
