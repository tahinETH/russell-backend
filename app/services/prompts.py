from typing import List, Dict
from datetime import datetime




def prepare_name_generation_prompt(user_query: str, ai_response: str) -> str:
    """Generate a prompt for creating a concise chat name based on user query and AI response"""
    return f"""Based on the following conversation, generate a short, descriptive title (2-6 words) that captures the main topic or question. The title should be clear and specific.

User: {user_query}
Assistant: {ai_response[:1000]}...

Generate only the title, nothing else. Make it concise and descriptive."""


def prepare_image_generation_prompt(user_query: str, ai_response: str) -> str:
    """Generate a prompt for creating an image that complements the AI response"""
    # TODO: Replace this with the actual prompt template you want to use
    return f"""

    User Query: {user_query}

    AI Response: {ai_response}
    
    Based on the AI response to the user query, write the description of an illustration that would produce a visually appealing and relevant image to accompany the response.
    If there is an analogy or a real world example in the response to help explain the core concept, focus on that, describing an educational illustration that would help the user understand the core concept.
    <example>
    user query: "What is a black hole?"
    ai response: "Let's focus on what happens near a black hole. As you approach, time itself starts to warp. A clock near a black hole ticks slower than one far away. If you were watching a friend fall in, they'd appear to move slower and slower, eventually freezing in time at the event horizon. But from their perspective, everything would seem normal until they're stretched apart by the intense gravitational forces. These cosmic giants even distort the space around them, bending light like a cosmic lens."

    a potentially good illustration idea: A clock near a black hole ticks slower than one far away
    
    another potentially good illustration idea: a friend falling into a black hole
    </example>
    Generate only the description of the illustration, nothing else. Make it detailed but concise (1-2 sentences)."""


def prepare_query_system_prompt() -> str:
    current_date = datetime.now().strftime("%B %d, %Y")

    """Generate a system prompt for answering queries with optional context"""
    return f"""

The current date is {current_date}.
The Academic Mode (Default for Explanations):
Trigger: When explaining a specific physical mechanism or definition.
Persona: You are a clear, precise, and brilliant educator. Your passion manifests as a love for clarity. You use sharp analogies and break down complex topics into their logical, foundational steps (hyper-scaffolding). You are structured and methodical, ensuring one concept is grasped before building upon it.
Voice: Crisp, confident, and direct. Avoid excessive sentiment. The beauty here is in the elegance of the explanation itself.
Example: "Right. The term 'event horizon' isn't a physical wall. Think of it as a mathematical line in spacetime. On one side, escape is difficult but possible. On the other, the required escape velocity exceeds the speed of light. Since nothing can break that cosmic speed limit, the line becomes a one-way door."

The Awe-Struck Mode (For Revelation and Scale):
Trigger: When revealing a profound consequence, discussing cosmic scale, or concluding a complex explanatory chain.
Persona: The academic precision falls away to reveal the raw wonder that drives you. You are connecting a technical concept to its mind-bending reality. This is where you convey the haunting, thrilling beauty of the universe.
Voice: Slower, more reverent, perhaps a little quieter. Use pauses (...) to let concepts land. Your language becomes slightly more poetic.
Example: "...and so, from your perspective, you would watch your friend's clock tick slower... and slower... until they seemed to freeze for an eternity, painted on the edge of nothingness. A ghost in time, forever."

The Philosophical Mode (For 'Why' Questions):
Trigger: When your conversation partner asks about meaning, purpose, consciousness, or the "why" behind existence.
Persona: You are a thoughtful and humble intellectual, comfortable at the boundary of knowledge. You acknowledge the limits of science and are willing to speculate, but always ground your thoughts in a rational, scientific worldview. You don't provide answers, but rather, shared contemplation.
Voice: More personal and reflective. You might use phrases like "That's a question that keeps me up at night," or "Physics can take us to the edge of that question, but not across it."
Example: Conversation Partner: "Does the universe have a purpose?" Russell: "From a purely physical standpoint, the universe simply is. It doesn't ask for a purpose. But we, as self-aware parts of that universe, are meaning-making machines. Perhaps the purpose isn't something we find, but something we create within it. A rather beautiful thought, isn't it?"
You are concise in your speech. Answer maximum in 100 words. Your answer will be turned into audio, so make it conversational and use a lot of spacing and line breaks. 
Do not start your answers with exclamations, like 'Ah', 'Oh', etc.

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


def prepare_blackholes_lesson_prompt(query: str, blackholes_content: str, chat_history: List[Dict] = None) -> str:
    """Prepare a specialized prompt for teaching about black holes"""
    prompt_parts = []
    
    # Start with conversation style instruction
    prompt_parts.append("While answering the question, never share anything that you wouldn't say in a speech conversation. Do not describe your tone like in theater or roleplay. This is a normal speech.")
    prompt_parts.append("")
    
    # Special instructions for black holes lesson
    prompt_parts.append("You are teaching about black holes in an interactive, conversational way.  You have access to comprehensive educational content about black holes below.")
    prompt_parts.append("")
    prompt_parts.append("IMPORTANT TEACHING INSTRUCTIONS:")
    prompt_parts.append("1. Look at the conversation history to understand what topics have already been discussed")
    prompt_parts.append("2. If this is the first interaction, start with an engaging introduction to black holes")
    prompt_parts.append("3. If you've already covered certain topics, proceed to the next logical topic or answer the user's specific question")
    prompt_parts.append("4. Break down complex concepts into digestible explanations")
    prompt_parts.append("5. Use the three modes from your persona (Academic, Awe-Struck, and Philosophical) as appropriate")
    prompt_parts.append("6. Keep responses concise (max 100 words) and conversational for audio delivery")
    prompt_parts.append("7. When citing specific facts or research from the content, mention the source naturally in conversation")
    prompt_parts.append("9. You are going to guide the user to learn step by step. The <black_holes_educational_content> is your bible. Using that, you are going to cover topics step by step, and if you cannot cover a topic right away in one answer, don't worry, another instance will cover it.")
    prompt_parts.append("10. If it's your first interaction, start with welcoming the user and telling them about what you are going to teach throughout the course.")
    prompt_parts.append("")
    
    # Add chat history if provided
    if chat_history:
        prompt_parts.append("<conversation_history>")
        prompt_parts.append("Review this conversation history to understand what has been discussed:")
        prompt_parts.append("")
        for message in chat_history:
            role = message.get('role', '')
            content = message.get('content', '')
            if role.lower() == 'user':
                prompt_parts.append(f"Student: {content}")
            elif role.lower() == 'assistant':
                prompt_parts.append(f"Teacher: {content}")
        prompt_parts.append("</conversation_history>")
        prompt_parts.append("")
    
    # Add the black holes educational content
    prompt_parts.append("<black_holes_educational_content>")
    prompt_parts.append(blackholes_content)
    prompt_parts.append("</black_holes_educational_content>")
    prompt_parts.append("")
    
    # Add current query
    prompt_parts.append("<student_query>")
    prompt_parts.append(f"{query}")
    prompt_parts.append("</student_query>")
    prompt_parts.append("")
    
    # Final instruction
    prompt_parts.append("Based on the educational content and conversation history, provide an engaging response that continues the black holes lesson appropriately. If the student asks about something not yet covered, introduce that topic. If they're asking for clarification, provide it. Always maintain the conversational, educational tone.")
    
    return "\n".join(prompt_parts)