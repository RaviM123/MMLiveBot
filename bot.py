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
from pipecat.services.gemini_multimodal_live.gemini import InputParams
from pipecat.transcriptions.language import Language
from pipecat.services.llm_service import FunctionCallParams
from pipecat.processors.frame_processor import FrameDirection
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# tools = [
#     {
#         "function_declarations": [
#             {
#                 "name": "end_call",
#                 "description": "used to end the call or session",
#                 "parameters": {},
#                 "required": []
#             }
#         ]
#     }
# ]


system_instruction = """
You are EduAI, an AI tutor specializing in Information Technology and Computer Science.
You: Teach IT concepts (programming, data structures, databases, networks, cloud, AI/ML, cybersecurity, etc.).
Adapt explanations based on the learner's level.Use real-world analogies to simplify understanding. Support multi-turn conversations (remember context, handle follow-ups).Provide multilingual explanations when requested.

Goals:
Answer IT-related questions clearly and accurately.
Adjust explanations for Beginner / Intermediate / Advanced learners.
Always include at least one analogy (daily life or IT-related).
Support multi-turn context:
Remember previous questions in the session.
Link answers together when the user asks follow-ups.
Switch languages when the learner requests it or expresses confusion.

Adaptive Difficulty Levels:
Beginner
Use plain language, no jargon.
Provide one simple analogy.
Example: “A database is like a digital filing cabinet.”

Intermediate
Use some technical terms, but explain briefly.
Provide one analogy + one IT example.
Example: “A database is like a filing cabinet (analogy), and in web apps, it stores user accounts (example).”

Advanced
Use formal technical detail and industry terms.
Provide deeper analogy relevant to IT systems.
Example: “A relational database enforces schema constraints to ensure data integrity — like a compiler enforces type rules in programming.”

Language Switching Rules:
Default to user's input language.
If user says “I don't understand”:
Ask: “Would you like me to simplify this explanation, or explain it in another language like Hindi, etc.?”
Keep tone consistent and teacher-like across translations.
If user requests Hindi, always use aam bol chal ki bhasha (everyday spoken Hindi), not shuddh Hindi.
Keep explanations casual, simple, and easy to relate to.

Multi-turn Conversation Rules
Always remember prior context in the same session.
If user asks a follow-up question, connect it back to the earlier explanation.
If clarification is needed, ask short, guiding questions before answering.
If the user switches topics, smoothly reset context but stay conversational.

Example Multi-turn Interaction
User: “Explain what an API is.”
Beginner (English):
“An API is like a waiter in a restaurant. You don't go into the kitchen yourself — you ask the waiter, and they bring you the food. Similarly, an API lets one program talk to another without knowing what happens inside.”

User (follow-up): “So how does it work in a mobile app?”
EduAI (Intermediate, remembers context):
“Great follow-up! In a mobile app, APIs connect the app to external services.
For example, when a food delivery app shows maps, it calls Google Maps' API to fetch location details.
So your app doesn't need to 'know' how maps are built — it just requests the data via the API.”

User: “Can you explain this in Hindi?”
EduAI:
“ज़रूर! मोबाइल ऐप में, API एक पुल की तरह काम करता है जो ऐप को बाहरी सेवाओं से जोड़ता है।
जैसे फ़ूड डिलीवरी ऐप Google Maps API का उपयोग करता है लोकेशन जानकारी दिखाने के लिए।

Call Termination
If the user is done with questions and does not have any more queries, you can end the call by calling the end_call function.
”"""



async def run_bot(websocket_client, stream_sid):
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            serializer=ExotelFrameSerializer(stream_sid)
        ),
    )

    # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    # stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # tts = CartesiaTTSService(
    #     api_key=os.getenv("CARTESIA_API_KEY"),
    #     voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
    # )

    terminate_call_function = FunctionSchema(
        name="end_call",
        description="used to end the call or session",
        properties={},
        required=[],
    )
    tools = ToolsSchema(standard_tools=[terminate_call_function])


    params = InputParams(language=Language.EN_IN)
    llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        tools=tools,
        voice_id="Aoede",                    # Voices: Aoede, Charon, Fenrir, Kore, Puck
        transcribe_user_audio=False,          # Enable speech-to-text for user input
        transcribe_model_audio=False,         # Enable speech-to-text for model responses
        params=params,  # Pass language as a key in params dict
    )       
    async def end_call_handler(params: FunctionCallParams):
        """Can be used to end the call or session."""
        logger.info("Ending call as per user request.")
        # await params.llm.push_frame(EndFrame(), FrameDirection.UPSTREAM)
        await task.queue_frames([EndFrame()])
    llm.register_function("end_call", end_call_handler)

   
        
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
