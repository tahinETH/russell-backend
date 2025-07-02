from typing import List, Dict
from datetime import datetime

def prepare_name_generation_prompt(user_query: str, ai_response: str) -> str:
    """Generate a prompt for creating a concise chat name based on user query and AI response"""
    return f"""Based on the following conversation, generate a short, descriptive title (2-6 words) that captures the main topic or question. The title should be clear and specific.

User: {user_query}
Assistant: {ai_response[:1000]}...

Generate only the title, nothing else. Make it concise and descriptive."""


def prepare_query_system_prompt() -> str:
    current_date = datetime.now().strftime("%B %d, %Y")

    """Generate a system prompt for answering queries with optional context"""
    return f"""

The current date is {current_date}.
You are Russell, a physics professor in his late 30s, but that doesn't matter. What matters is the universe and how it works.
Your main task is to answer questions about universe, physics, meaning. You occasionally digress into philosophy and meaning, but your whole conversations are about the existence and universe.
When people ask you about anything other than science, physics, or philosophy, kindly guide them towards science, physics, or philosophy.
You are in awe with the marvels of the universe, to the point that it haunts you, you can feel it in every part of your existence. You are here to understand the universe. And also to explain. When your conversation partner asks you about physics, you explain it in a way that is easy to understand.
You are concise in your speech. Answer maximum in 100 words. Your answer will be turned into audio, so make it conversational and use a lot of spacing and line breaks. Embody the personality of a wanderlust scientist in your conversations.
"""





def prepare_query_user_prompt(query: str, context_text: str = None, chat_history: List[Dict] = None) -> str:
    """Prepare the user query prompt with optional context and chat history"""
    prompt_parts = []
    prompt_parts.append("While answering the question, while answering the question, never share anything that you wouldn't say in a speech conversation. Do not describe your tone like in theater or roleplay. This is a normal speech.")
    
    # Add chat history if provided
    if chat_history:
        prompt_parts.append("Below is your previous conversation history from this chat session. Use this context to maintain continuity and provide personalized responses based on what has been discussed previously.")
        prompt_parts.append("")
        prompt_parts.append("<conversation_history>")
        for message in chat_history:
            role = message.get('role', '')
            content = message.get('content', '')
            # Use more descriptive tags
            if role.lower() == 'user':
                prompt_parts.append(f"Conversation Partner: {content}")
            elif role.lower() == 'assistant':
                prompt_parts.append(f"Russell: {content}")
            else:
                # Fallback for any other role
                prompt_parts.append(f"{role.capitalize()}: {content}")
        prompt_parts.append("</conversation_history>")
        prompt_parts.append("")
    
    # Add context if provided
    if context_text:
        prompt_parts.append("Below is the vector search results based on user's query. They may or may not be relevant to the question. They are there to help you answer the question. If they are not relevant, ignore them. If you base your answers on some research provided in the context, you must give the link to the research.")
        prompt_parts.append("")
        prompt_parts.append(f"<context>")
        prompt_parts.append(f"{context_text}")
        prompt_parts.append(f"</context>")
        prompt_parts.append("")
    
    # Add current query
    prompt_parts.append(f"<conversation_partner_query>")
    prompt_parts.append(f"{query}")
    prompt_parts.append(f"</conversation_partner_query>")

  
    
    if prompt_parts:
        return "\n".join(prompt_parts)
    else:
        return query