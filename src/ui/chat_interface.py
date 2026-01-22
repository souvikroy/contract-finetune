"""Chat interface component."""
import streamlit as st
from typing import List, Dict, Any, Optional
from src.ui.components import render_message


class ChatInterface:
    """Chat interface manager."""
    
    def __init__(self):
        """Initialize chat interface."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def render(self):
        """Render chat interface."""
        st.title("🤖 Legal Contract Chatbot")
        st.markdown("Ask questions about the contract document")
        
        # Display chat history
        for message in st.session_state.messages:
            is_user = message.get('role') == 'user'
            render_message(message, is_user=is_user)
    
    def add_message(self, role: str, content: str, sources: Optional[List[Dict]] = None):
        """Add message to chat."""
        message = {
            'role': role,
            'content': content,
            'sources': sources or []
        }
        st.session_state.messages.append(message)
        st.session_state.chat_history.append(message)
    
    def get_user_input(self) -> Optional[str]:
        """Get user input from chat input."""
        if prompt := st.chat_input("Ask a question about the contract..."):
            return prompt
        return None
    
    def clear_history(self):
        """Clear chat history."""
        st.session_state.messages = []
        st.session_state.chat_history = []
    
    def export_history(self) -> str:
        """Export chat history as text."""
        history_text = "Chat History\n" + "="*50 + "\n\n"
        
        for msg in st.session_state.chat_history:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            history_text += f"{role}: {content}\n\n"
            
            if msg.get('sources'):
                history_text += "Sources:\n"
                for source in msg['sources']:
                    history_text += f"  - Page {source.get('page')}: {source.get('text_preview', '')[:100]}\n"
                history_text += "\n"
        
        return history_text
