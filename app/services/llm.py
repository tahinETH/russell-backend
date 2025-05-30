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
    
    async def generate_chat_name(self, user_query: str, ai_response: str) -> str:
        """Generate a concise chat name based on user query and AI response using gpt-4o-mini"""
        try:
            prompt = f"""Based on the following conversation, generate a short, descriptive title (2-6 words) that captures the main topic or question. The title should be clear and specific.

User: {user_query}
Assistant: {ai_response[:1000]}...

Generate only the title, nothing else. Make it concise and descriptive."""

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