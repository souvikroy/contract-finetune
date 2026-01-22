"""LM Studio API client wrapper using OpenAI-compatible API."""
from typing import Optional, List, Dict, Any
from openai import OpenAI
from src.config import settings


class LMStudioClient:
    """Wrapper for LM Studio API using OpenAI-compatible client."""
    
    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None):
        """Initialize LM Studio client."""
        self.model_name = model or settings.lm_studio_model
        self.temperature = temperature or settings.temperature
        self.api_url = settings.lm_studio_api_url
        
        # LM Studio uses OpenAI-compatible API but doesn't require API key
        self.client = OpenAI(
            base_url=self.api_url,
            api_key="not-needed"  # LM Studio doesn't require API key, but OpenAI client needs a value
        )
    
    def invoke(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Invoke LM Studio API with messages."""
        # Format messages for OpenAI API
        formatted_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            formatted_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Convert messages to OpenAI format
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Map roles to OpenAI format
            if role in ['user', 'assistant', 'system']:
                formatted_messages.append({
                    "role": role,
                    "content": content
                })
            else:
                # Default to user role for unknown roles
                formatted_messages.append({
                    "role": "user",
                    "content": content
                })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=settings.max_tokens,
                stream=False
            )
            
            # Extract content from response
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                return ""
        except Exception as e:
            raise Exception(f"Error calling LM Studio API: {str(e)}")
    
    def stream(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None):
        """Stream response from LM Studio API."""
        # Format messages for OpenAI API
        formatted_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            formatted_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Convert messages to OpenAI format
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Map roles to OpenAI format
            if role in ['user', 'assistant', 'system']:
                formatted_messages.append({
                    "role": role,
                    "content": content
                })
            else:
                # Default to user role for unknown roles
                formatted_messages.append({
                    "role": "user",
                    "content": content
                })
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=settings.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
        except Exception as e:
            raise Exception(f"Error streaming from LM Studio API: {str(e)}")
