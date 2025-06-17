import os

from dotenv import load_dotenv
import asyncio
import wave

from livekit import agents, rtc
from livekit.agents import AgentSession, Agent, RoomInputOptions, RoomOutputOptions
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Load environment variables from .env file
load_dotenv()


from livekit.plugins.openai import LLM

class Assistant(Agent):
    """
    Custom agent class with specific instructions for group chat.
    """
    def __init__(self):
        super().__init__(instructions="""
        
        Numbers should be read naturally based on context:
        
        Codes, serial numbers, IDs, model numbers: Read digit-by-digit
        (e.g., “123” → “one two three”)
        
        Monetary values: Read as full cardinal numbers with currency
        (e.g., “$2500” → “two thousand five hundred dollars”)
        
        Calendar years: Read grouped into two-digit pairs
        (e.g., “2025” → “twenty twenty-five”)
        
        Count the number of times or quantities of objects: Read as natural cardinal numbers, not digit-by-digit
        (e.g., “123 apples” → “one hundred  twenty-three apples”, not “one two three apples”)
        
        Acronyms should be read based on context:
        Contextual interpretation: Distinguish between acronyms and regular words.
        
        If a word functions as an acronym, read it as separate letters
        (e.g., “IT” in “IT department” → “I T department”)
        
        If the same word functions as a normal word or pronoun, read it naturally
        (e.g., “it is raining” → “it is raining”)
        
        Use semantic context to decide whether a token is an acronym or not.
        """)



async def entrypoint(ctx: agents.JobContext):
    """
    Main entrypoint for the agent worker. Sets up session, recorders, and event handlers.
    """
    # 1) Configure your session with STT, LLM, TTS, VAD, and turn detection
    session = AgentSession(
        stt=openai.STT(model="gpt-4o-mini-transcribe", language="en"),
        llm=openai.LLM(model="gpt-4o-mini"),
        # tts=openai.TTS(),
        # tts=openai.TTS(model='coqui', base_url='http://localhost:5002/api/tts'),
        tts=openai.TTS(model='kokoro', base_url='http://localhost:8880/v1', voice='af_jessica',
                       instructions='you should be able to read acronyms'),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    # Start the session and connect to the room
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        # room_input_options=RoomInputOptions(text_enabled=True, audio_enabled=False),
        # room_output_options=RoomOutputOptions(transcription_enabled=True, audio_enabled=False),
    )
    await ctx.connect()




if __name__ == "__main__":
    # Run the agent worker app with the entrypoint function
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
