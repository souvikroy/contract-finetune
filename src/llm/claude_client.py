"""Claude API client wrapper with LangChain."""
from typing import Optional, List, Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from src.config import settings


class ClaudeClient:
    """Wrapper for Claude API using LangChain."""
    
    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None):
        """Initialize Claude client."""
        self.model_name = model or settings.claude_model
        self.temperature = temperature or settings.temperature
        
        self.llm = ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=settings.max_tokens,
            anthropic_api_key=settings.anthropic_api_key
        )
    
    def invoke(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Invoke Claude API with messages."""
        langchain_messages = []
        
        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'user':
                langchain_messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                langchain_messages.append(AIMessage(content=content))
            elif role == 'system':
                langchain_messages.append(SystemMessage(content=content))
        
        response = self.llm.invoke(langchain_messages)
        return response.content
    
    def stream(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None):
        """Stream response from Claude API."""
        langchain_messages = []
        
        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'user':
                langchain_messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                langchain_messages.append(AIMessage(content=content))
        
        for chunk in self.llm.stream(langchain_messages):
            yield chunk.content
