from typing import AsyncIterator
from videosdk.agents import ConversationFlow, ChatRole
from doc_rag_handler import search_hr_policy_knowledge


class MyConversationFlow(ConversationFlow):
    """Custom conversation flow with RAG integration for HR policy knowledge."""
    
    def __init__(self, agent, stt=None, llm=None, tts=None):
        """
        Initialize the conversation flow.
        
        Args:
            agent: The voice agent instance
            stt: Speech-to-text processor
            llm: Language model
            tts: Text-to-speech processor
        """
        super().__init__(agent, stt, llm, tts)

    async def run(self, transcript: str) -> AsyncIterator[str]:
        """
        Main conversation loop: handle a user turn.
        
        Args:
            transcript: User's speech transcript
            
        Yields:
            str: Response chunks from the LLM
        """
        await self.on_turn_start(transcript)
        
        processed_transcript = transcript.lower().strip()

        # Initialize RAG context
        retrieved_context = None
        try:
            # Perform RAG retrieval for HR policy knowledge
            retrieved_context = await search_hr_policy_knowledge(processed_transcript)
        except Exception as e:
            print(f"Error during RAG retrieval: {e}")
            # Optionally, you could inform the user or LLM that context retrieval failed
            # self.agent.chat_context.add_message(
            #     role=ChatRole.SYSTEM, 
            #     content="[System: HR policy context retrieval failed.]"
            # )

        # Add retrieved context to chat history if available
        if retrieved_context:
            self.agent.chat_context.add_message(
                role=ChatRole.SYSTEM, 
                content=f"Context from HR Policy Manual: {retrieved_context}"
            )

        # Add user message to chat context
        self.agent.chat_context.add_message(role=ChatRole.USER, content=processed_transcript)
        
        # Process with LLM and yield response chunks
        async for response_chunk in self.process_with_llm():
            yield response_chunk

        await self.on_turn_end()

    async def on_turn_start(self, transcript: str) -> None:
        """
        Called at the start of a user turn.
        
        Args:
            transcript: User's speech transcript
        """
        self.is_turn_active = True

    async def on_turn_end(self) -> None:
        """Called at the end of a user turn."""
        self.is_turn_active = False 