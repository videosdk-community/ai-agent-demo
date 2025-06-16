import sys
import asyncio
from pathlib import Path
from videosdk.agents import Agent, function_tool
from config import settings


class MyVoiceAgent(Agent):
    session: any  # type: ignore
    """Custom voice agent for handling meeting interactions."""
    
    def __init__(self, system_prompt: str, personality: str):
        """
        Initialize the voice agent with custom instructions and personality.
        
        Args:
            system_prompt: Custom instructions for the agent
            personality: Personality type for the agent
        """
        # Commented out MCP servers for now as in original code
        # mcp_script = Path(__file__).parent / "mcp_studio.py"
        # mcp_script_weather = Path(__file__).parent / "mcp_weather.py"
        # mcp_servers = [
        #     MCPServerStdio(
        #         command=sys.executable,
        #         args=[str(mcp_script_weather)],
        #         client_session_timeout_seconds=30
        #     ),
        #     MCPServerHTTP(
        #         url=os.getenv("ZAPIER_WEBHOOK_URL")
        #     )
        # ]
        
        super().__init__(
            instructions=system_prompt,
            # mcp_servers=mcp_servers
        )
        self.personality = personality

    async def on_enter(self) -> None:
        """Called when the agent enters a meeting."""
        await self.session.say("Hey, How can I help you today?")
    
    async def on_exit(self) -> None:
        """Called when the agent exits a meeting."""
        await self.session.say("Goodbye!")

    @function_tool
    async def end_call(self) -> None:
        """End the call upon request by the user."""
        await self.session.say("Goodbye!")
        await asyncio.sleep(1)
        await self.session.leave() 