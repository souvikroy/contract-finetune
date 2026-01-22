"""Reusable UI components for Streamlit."""
import streamlit as st
from typing import List, Dict, Any


def render_message(message: Dict[str, Any], is_user: bool = True):
    """Render a chat message."""
    if is_user:
        with st.chat_message("user"):
            st.write(message.get('content', ''))
    else:
        with st.chat_message("assistant"):
            st.write(message.get('content', ''))
            
            # Show sources if available
            if 'sources' in message and message['sources']:
                with st.expander("Sources"):
                    for i, source in enumerate(message['sources'], 1):
                        st.write(f"**Source {i}:**")
                        st.write(f"- Page: {source.get('page', 'Unknown')}")
                        if source.get('section'):
                            st.write(f"- Section: {source['section']}")
                        if source.get('clause_id'):
                            st.write(f"- Clause: {source['clause_id']}")
                        st.write(f"- Preview: {source.get('text_preview', '')[:200]}...")


def render_query_suggestions(suggestions: List[str]):
    """Render query suggestions."""
    st.subheader("💡 Try asking:")
    for suggestion in suggestions:
        if st.button(suggestion, key=f"suggestion_{suggestion[:20]}"):
            return suggestion
    return None


def render_document_viewer(document: Dict[str, Any]):
    """Render document viewer."""
    st.subheader("📄 Document View")
    
    if 'section' in document:
        st.write(f"**Section:** {document['section']}")
    
    if 'clause_id' in document:
        st.write(f"**Clause:** {document['clause_id']}")
    
    st.write(f"**Page:** {document.get('page', 'Unknown')}")
    
    st.text_area("Content", document.get('text', ''), height=200, disabled=True)
