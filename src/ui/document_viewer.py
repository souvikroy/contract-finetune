"""Document viewer component."""
import streamlit as st
from typing import List, Dict, Any


class DocumentViewer:
    """Document viewer for contract sections."""
    
    def render_source_documents(self, sources: List[Dict[str, Any]]):
        """Render source documents from RAG retrieval."""
        if not sources:
            return
        
        st.subheader("📚 Relevant Document Sections")
        
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i} - Page {source.get('page', 'Unknown')}"):
                if source.get('section'):
                    st.write(f"**Section:** {source['section']}")
                
                if source.get('clause_id'):
                    st.write(f"**Clause:** {source['clause_id']}")
                
                if source.get('type'):
                    st.write(f"**Type:** {source['type']}")
                
                st.text_area(
                    "Content",
                    source.get('text_preview', ''),
                    height=150,
                    key=f"source_{i}",
                    disabled=True
                )
    
    def render_full_document_section(self, section: Dict[str, Any]):
        """Render a full document section."""
        st.subheader("📄 Document Section")
        
        if 'header' in section:
            st.write(f"**{section['header']}**")
        
        if 'content' in section:
            for item in section['content']:
                st.write(item.get('text', ''))
        
        if 'page' in section:
            st.caption(f"Page {section['page']}")
