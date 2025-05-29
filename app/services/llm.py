import litellm
from typing import List, Dict, AsyncGenerator
import asyncio

class LLMService:
    def __init__(self, model: str):
        self.model = model
        
    
    async def stream_with_context(
        self, 
        query: str, 
        context: List[Dict]
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response with provided context"""
        # Build prompt with context
        if context:
            context_text = "\n".join([c["content"] for c in context])
            system_message = f"""Use the following context to answer the user's question. 
If the context doesn't contain relevant information, say so clearly.

Context:
{context_text}"""
        else:
            system_message = "Answer the user's question based on your knowledge."
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
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