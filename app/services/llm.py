import litellm
from typing import List, Dict, AsyncGenerator
import asyncio
from .prompts import prepare_name_generation_prompt, prepare_query_system_prompt, prepare_query_user_prompt

class LLMService:
    def __init__(self, model: str):
        self.model = model
        
    
    async def stream_with_context(
        self, 
        query: str, 
        context: List[Dict],
        chat_history: List[Dict] = None,
        custom_system_prompt: str = None
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response with provided context and chat history"""
        # Build prompt with context
        context_text = None
        if context:
            context_items = []
            for c in context:
                if "metadata" in c and "title" in c["metadata"] and c["metadata"]["title"]:
                    content_str = f"Title:{c['metadata']['title']}\n\nContent:{c['content']}"
                    if "link" in c["metadata"] and c["metadata"]["link"]:
                        content_str += f"\nLink:{c['metadata']['link']}"
                    context_items.append(content_str)
                else:
                    context_items.append(c["content"])
            context_text = "\n\n".join(context_items)
        
        
        # Use custom system prompt if provided, otherwise use default
        if custom_system_prompt:
            system_message = custom_system_prompt
        else:
            system_message = prepare_query_system_prompt()
        
        user_message = prepare_query_user_prompt(query, context_text, chat_history)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            # Stream response
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0])
                    yield chunk.choices[0].delta.content
        except Exception as e:
            # Handle errors gracefully
            yield f"Error generating response: {str(e)}"
    
    async def get_chat_completion(
        self,
        messages: List[Dict],
        stream: bool = False
    ) -> str:
        """Get a complete chat response (non-streaming)"""
        try:
            response = await litellm.acompletion(
                model=self.model,
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
            return f"Error: {str(e)}"
    
    async def generate_chat_name(self, user_query: str, ai_response: str) -> str:
        """Generate a concise chat name based on user query and AI response using gpt-4o-mini"""
        try:
            prompt = prepare_name_generation_prompt(user_query, ai_response)

            response = await litellm.acompletion(
                model="gpt-4o-mini",
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
            # Fallback to a simple name based on user query
            words = user_query.split()[:3]
            return " ".join(words).title() or "New Chat" 