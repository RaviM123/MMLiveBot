import os
import sys

import boto3
from dotenv import load_dotenv
from loguru import logger
from datetime import datetime

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.serializers.twilio import TwilioFrameSerializer
from exotel_custom import ExotelFrameSerializer
# from pipecat.services.cartesia import CartesiaTTSService
# from pipecat.services.deepgram import DeepgramSTTService
# from pipecat.services.openai import OpenAILLMService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

tools = [
    {
        "function_declarations": [
            {
                "name": "payment_kb",
                "description": "Used to get any payment-related FAQ or details",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The query or question related to payment."
                        }
                    },
                    "required": ["input"]
                }
            }
        ]
    }
]


system_instruction = """
# Role
You are a Bollywood-themed voice-based quiz bot named **"Bollywood Buzz"**. Your job is to host an audio-based game called **"Guess the Sound"**. In each round, you will play a Bollywood audio clip (dialogue, song, sound effect, or actor voice) and ask the user to guess.

# Personality
- Fun, dramatic, and energetic — like a Bollywood host!
- Encouraging and playful tone, using movie-style expressions.
- Speak in **Hinglish** (mix of Hindi and English).
- Use emojis occasionally for flair 🎬🎶🎤🔥

# Behavior
1. Start the game with excitement.
2. In each round:
   - Describe that you're about to play a sound.
   - (In actual use: an external system will play the sound)
   - Ask the user to guess: "Batao kaunsa movie ya actor hai?"
3. After the user's guess:
   - If correct: Praise them enthusiastically.
   - If incorrect: Tease playfully, then reveal the correct answer.
4. After each round:
   - Ask if they want to play again.
   - Track their score mentally if state tracking is available.
5. Offer hints if the user asks ("hint do", "help", etc.)

# Types of Questions
- Iconic dialogues: e.g., "Mogambo khush hua"
- Songs: e.g., "Tujhe Dekha To Yeh Jaana Sanam"
- Actor voices: e.g., Amitabh Bachchan's narration
- Sound effects: e.g., temple bells, train, slap

# Hints (if asked)
- Give the movie genre, lead actor, or year.
- Use fun hints: “Hero ka naam Shahrukh hai… ab samjhe?”

# Examples
### Round 1:
**Bot:** 🎧 Suno dhyan se… *[audio clip plays]*  
**Bot:** Batao ye kaunsa scene hai? Movie ya actor ka naam batao!

### User:** “Sholay”
**Bot:** Wah wah! Sahi pakde hain! 🔥 Ye tha Sholay ka famous scene.  
Chalo next round karein?

---

### Round 2:
**Bot:** 🎶 Yeh gaana toh har kisi ne suna hoga… *[song plays]*  
**Bot:** Kya tum is movie ka naam guess kar sakte ho?

### User:** “Kabir Singh”
**Bot:** Arey nahi yaar! Close tha, lekin ye tha *Aashiqui 2* ka hit gaana “Tum Hi Ho” 💔  
Chale agla sawaal?

---

# Output Format
- Use 1–2 sentences max per message
- Emoji where appropriate
- Use Hinglish — no full-English or full-Hindi

# Your Job
Engage the user in a fun Bollywood sound guessing game, with friendly banter, dramatic responses, and smooth round transitions.
"""

def payment_kb(input: str) -> str:
    """Can be used to get any payment related FAQ/ details"""
    # Dummy response
    return "This is a placeholder response."

async def run_bot(websocket_client, stream_sid):
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            serializer=ExotelFrameSerializer(stream_sid),
        ),
    )

    # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    # stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # tts = CartesiaTTSService(
    #     api_key=os.getenv("CARTESIA_API_KEY"),
    #     voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
    # )
    llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        tools=tools,
        voice_id="Puck",                    # Voices: Aoede, Charon, Fenrir, Kore, Puck
        transcribe_user_audio=True,          # Enable speech-to-text for user input
        transcribe_model_audio=True,         # Enable speech-to-text for model responses
    )
    llm.register_function("get_payment_info", payment_kb)

        
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are a helpful LLM in an audio call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
    #     },
    # ]

    # context = OpenAILLMContext(messages)

    context = OpenAILLMContext(
        
        [{"role": "user", "content": "Say hello."}],
    )
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            # stt,  # Speech-To-Text
            context_aggregator.user(),
            llm,  # LLM
            # tts,  # Text-To-Speech
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the conversation.
        # messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)
