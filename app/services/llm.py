import litellm
from typing import List, Dict, AsyncGenerator
import asyncio
import logging
from .prompts import prepare_name_generation_prompt, prepare_query_system_prompt, prepare_query_user_prompt

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, model: str):
        self.model = model
        self.fallback_model = "gpt-4.1"  # Fallback model
        
    
    async def stream_with_context(
        self, 
        query: str, 
        context: List[Dict],
        chat_history: List[Dict] = None
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response with provided context and chat history"""
        # Build prompt with context
        context_text = None
        """  if context:
            context_items = []
            for c in context:
                if "metadata" in c and "title" in c["metadata"] and c["metadata"]["title"]:
                    content_str = f"Title:{c['metadata']['title']}\n\nContent:{c['content']}"
                    if "link" in c["metadata"] and c["metadata"]["link"]:
                        content_str += f"\nLink:{c['metadata']['link']}"
                    context_items.append(content_str)
                else:
                    context_items.append(c["content"])
            context_text = "\n\n".join(context_items) """
        
        
        # Use default system prompt
        system_message = prepare_query_system_prompt()
        
        user_message = prepare_query_user_prompt(query, context_text, chat_history)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Try main model first, then fallback
        models_to_try = [self.model, self.fallback_model]
        
        for model in models_to_try:
            try:
                logger.info(f"Attempting to use model: {model}")
                
                # Stream response
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    stream=True,
                    max_tokens=250,
                )
                
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return  # Success, exit the function
                        
            except Exception as e:
                logger.warning(f"Model {model} failed: {str(e)}")
                if model == self.fallback_model:
                    # If even fallback fails, yield error message
                    yield f"Error: Both main model ({self.model}) and fallback model ({self.fallback_model}) failed. Last error: {str(e)}"
                else:
                    # Continue to try fallback model
                    logger.info(f"Falling back to {self.fallback_model}")
                    continue
    
    async def get_chat_completion(
        self,
        messages: List[Dict],
        stream: bool = False
    ) -> str:
        """Get a complete chat response (non-streaming) with fallback support"""
        models_to_try = [self.model, self.fallback_model]
        
        for model in models_to_try:
            try:
                logger.info(f"Attempting to use model: {model}")
                
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    stream=stream
                )
                
                if stream:
                    full_response = ""
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                    return full_response
                else:
                    return response.choices[0].message.content
                    
            except Exception as e:
                logger.warning(f"Model {model} failed: {str(e)}")
                if model == self.fallback_model:
                    # If even fallback fails, return error message
                    return f"Error: Both main model ({self.model}) and fallback model ({self.fallback_model}) failed. Last error: {str(e)}"
                else:
                    # Continue to try fallback model
                    logger.info(f"Falling back to {self.fallback_model}")
                    continue
    
    async def generate_chat_name(self, user_query: str, ai_response: str) -> str:
        """Generate a concise chat name based on user query and AI response using gpt-4o-mini with fallback"""
        models_to_try = ["gpt-4o-mini", self.fallback_model]
        
        for model in models_to_try:
            try:
                prompt = prepare_name_generation_prompt(user_query, ai_response)

                response = await litellm.acompletion(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=20,
                    temperature=0.7
                )
                
                title = response.choices[0].message.content.strip()
                # Remove quotes if present and limit length
                title = title.strip('"\'').strip()
                return title[:50]  # Limit to 50 characters
                
            except Exception as e:
                logger.warning(f"Chat name generation with {model} failed: {str(e)}")
                if model == self.fallback_model:
                    # Fallback to a simple name based on user query
                    words = user_query.split()[:3]
                    return " ".join(words).title() or "New Chat"
                else:
                    continue 