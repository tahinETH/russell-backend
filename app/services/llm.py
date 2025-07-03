import litellm
from typing import List, Dict, AsyncGenerator, Optional
import asyncio
import logging
from .prompts import prepare_name_generation_prompt, prepare_query_system_prompt, prepare_query_user_prompt, prepare_blackholes_lesson_prompt, prepare_image_generation_prompt
import os

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, model: str):
        self.model = model
        self.fallback_model = "gpt-4.1"  # Fallback model
        self._blackholes_content = None  # Cache for blackholes content
        
    def _load_blackholes_content(self) -> str:
        """Load blackholes.txt content (cached after first load)"""
        if self._blackholes_content is None:
            try:
                # Get the directory where this file is located
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # Go up one level to app directory
                app_dir = os.path.dirname(current_dir)
                blackholes_path = os.path.join(app_dir, 'blackholes.txt')
                
                with open(blackholes_path, 'r', encoding='utf-8') as f:
                    self._blackholes_content = f.read()
            except Exception as e:
                logger.error(f"Error loading blackholes.txt: {e}")
                self._blackholes_content = ""
        return self._blackholes_content
    
    async def stream_with_context(
        self, 
        query: str, 
        context: List[Dict],
        chat_history: List[Dict] = None,
        lesson: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response with provided context and chat history"""
        
        # Check if this is a black holes lesson
        if lesson == "blackholes":
            # Load blackholes content
            blackholes_content = self._load_blackholes_content()
            
            # Use the specialized black holes prompt
            system_message = prepare_query_system_prompt()
            user_message = prepare_blackholes_lesson_prompt(query, blackholes_content, chat_history)
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
        else:
            # Regular flow with context
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

    async def generate_image_prompt(self, user_query: str, ai_response: str) -> Optional[str]:
        """Generate an image generation prompt based on user query and AI response using gpt-4o"""
        try:
            prompt = prepare_image_generation_prompt(user_query, ai_response)
            
            response = await litellm.acompletion(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            image_prompt = response.choices[0].message.content.strip()
            logger.info(f"Generated image prompt: {image_prompt}")
            return image_prompt
            
        except Exception as e:
            logger.error(f"Failed to generate image prompt: {str(e)}")
            # Try with fallback model
            try:
                response = await litellm.acompletion(
                    model=self.fallback_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                
                image_prompt = response.choices[0].message.content.strip()
                logger.info(f"Generated image prompt with fallback: {image_prompt}")
                return image_prompt
                
            except Exception as fallback_error:
                logger.error(f"Fallback image prompt generation also failed: {str(fallback_error)}")
                return None 