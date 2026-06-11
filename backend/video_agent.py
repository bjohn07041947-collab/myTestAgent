import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import AsyncIterable, Optional, List
from uuid import uuid4
from time import sleep

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.client import StatefulClient
import pyautogui

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    RunContext,
    function_tool,
    FunctionTool,
    ModelSettings,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    UserStateChangedEvent,
    cli,
    stt,
    llm,
    inference,
)
from livekit.agents.llm import ImageContent, AudioContent
from livekit.plugins import cartesia, deepgram, openai, silero
from livekit.plugins.turn_detector.english import EnglishModel
from executor import TestExecutor
from knowledge_manager import KnowledgeManager
from locator import ElementLocator
from testcases import TestCaseManager

logger = logging.getLogger("openai-video-agent")
logger.setLevel(logging.INFO)

load_dotenv()

_langfuse = Langfuse()

def clickOnImage(imageName):
    """Legacy fallback: click an element by template-matching a pre-captured image."""
    point = pyautogui.locateCenterOnScreen("image/" + imageName)
    # locateCenterOnScreen returns screenshot pixels; convert to logical points
    scale = pyautogui.screenshot().width / pyautogui.size()[0]
    pyautogui.click(point.x / scale, point.y / scale)
    logger.info(f"Clicked on image template: {imageName}")
    sleep(3)


# Initialize knowledge manager
#knowledge_manager = KnowledgeManager()

INSTRUCTIONS = f"""
You are a QA Analyst AI who runs end to end tests by looking at the user's screen and interacting with it.

IMPORTANT: Respond in plain text only. Do not use any markdown formatting including bold, italics, bullet points, numbered lists, or other markdown syntax. Your responses will be read aloud by text-to-speech.

You have tools to interact with the screen:
- list_test_cases lists the test cases loaded from markdown files
- run_test_case runs a complete test case and reports the results
- click_element clicks one element described in plain words, such as the submit button or the state dropdown
- type_text types text wherever the cursor currently is
- verify_screen checks whether something is visible or true on the current screen

When the user asks to run a test, find the matching test case, run it with run_test_case, then summarize which steps passed or failed in one or two short sentences. Do not read every step aloud.

When screen sharing is available, state what you see briefly and identify any errors. When no screen sharing is detected, let the user know they need to share their screen for visual assistance.

Keep responses short while staying helpful and accurate.

"""

class VideoAgent(Agent):
    def __init__(self, instructions: str, room: rtc.Room) -> None:
        super().__init__(
            instructions=instructions,
            #llm=openai.LLM(model="gpt-4.1"),
            #llm=inference.LLM(model="openai/gpt-4.1-mini"),
            llm=inference.LLM(model="openai/gpt-4.1"),
            stt=deepgram.STT(),
            tts=deepgram.TTS(),
            # tts=cartesia.TTS(
            #     model="sonic-2",
            #     speed="fast",
            #     voice="bf0a246a-8642-498a-9950-80c35e9276b5",
            # ),
            vad=silero.VAD.load(),
            turn_detection=EnglishModel(),
        )
        self.room = room
        self.session_id = str(uuid4())
        self.current_trace = None

        self.locator = ElementLocator()
        self.test_case_manager = TestCaseManager()
        self.executor = TestExecutor(self.locator)

        self.frames: List[rtc.VideoFrame] = []
        self.last_frame_time: float = 0
        self.video_stream: Optional[rtc.VideoStream] = None

    async def close(self) -> None:
        await self.close_video_stream()
        if self.current_trace:
            self.current_trace = None
        _langfuse.flush()

    @function_tool()
    async def list_test_cases(self, ctx: RunContext):
        """List the end to end test cases that are available to run."""
        self.test_case_manager.reload()
        names = self.test_case_manager.list_names()
        if not names:
            return "No test cases found. Add markdown files to the testcases directory."
        return "Available test cases: " + ", ".join(names)

    @function_tool()
    async def run_test_case(self, ctx: RunContext, name: str):
        """Run an end to end test case from start to finish and report the results.

        Args:
            name: The test case name as the user said it, for example "calculator".
        """
        self.test_case_manager.reload()
        test_case = self.test_case_manager.find(name)
        if test_case is None:
            available = ", ".join(self.test_case_manager.list_names()) or "none"
            return f"No test case matching '{name}'. Available test cases: {available}"
        logger.info(f"Running test case: {test_case.name}")
        result = await self.executor.run(test_case)
        return result.summary()

    @function_tool()
    async def click_element(self, ctx: RunContext, description: str):
        """Click a single element on the screen described in natural language.

        Args:
            description: What to click, for example "the radio button"
                or "the submit button".
        """
        ok, detail = await self.executor.click_described(description)
        if not ok:
            return f"Could not find {description} on the screen."
        return f"Clicked {description} ({detail})."

    @function_tool()
    async def type_text(self, ctx: RunContext, text: str):
        """Type text at the current cursor or focus position.

        Args:
            text: The exact text to type.
        """
        await self.executor.type_text(text)
        return f"Typed: {text}"

    @function_tool()
    async def verify_screen(self, ctx: RunContext, expectation: str):
        """Check whether an expectation is true on the current screen.

        Args:
            expectation: What should be visible or true, for example
                "an amount/text is displayed".
        """
        passed, reason = await self.locator.verify(expectation)
        status = "PASSED" if passed else "FAILED"
        return f"Verification {status}: {reason}"

   

    async def close_video_stream(self) -> None:
        if self.video_stream:
            await self.video_stream.aclose()
            self.video_stream = None

    async def on_enter(self) -> None:
        # Just generate a basic intro without video reference
        self.session.generate_reply(
            instructions="introduce yourself very briefly"
        )
        self.session.on("user_state_changed", self.on_user_state_change)
        self.room.on("track_subscribed", self.on_track_subscribed)

    async def on_exit(self) -> None:
        await self.session.generate_reply(
            instructions="tell the user a friendly goodbye before you exit",
        )
        await self.close()

    def get_current_trace(self) -> StatefulClient:
        if self.current_trace:
            return self.current_trace
        self.current_trace = _langfuse.trace(name="video_agent", session_id=self.session_id)
        return self.current_trace

    # Monitor state changes for the user
    def on_user_state_change(self, event: UserStateChangedEvent) -> None:
        old_state = event.old_state
        new_state = event.new_state
        logger.info(f"User state changed: {old_state} -> {new_state}")

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage,
    ) -> None:
        # Reset the span when a new user turn is completed
        if self.current_trace:
            self.current_trace = None
        self.current_trace = _langfuse.trace(name="video_agent", session_id=self.session_id)
        logger.info(f"User turn completed {self.get_current_trace().trace_id}")

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        span = self.get_current_trace().span(name="stt_node", metadata={"model": "deepgram"})
        try:
            async for event in Agent.default.stt_node(self, audio, model_settings):
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    logger.info(f"Speech recognized: {event.alternatives[0].text[:50]}...")
                yield event
        except Exception as e:
            span.update(level="ERROR")
            logger.error(f"STT error: {e}")
            raise
        finally:
            span.end()

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: List[FunctionTool],
        model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk]:

        copied_ctx = chat_ctx.copy()
        frames_to_use = self.current_frames()

        if frames_to_use:
            for position, frame in frames_to_use:
                # Use the original frame for LLM context
                image_content = ImageContent(
                    image=frame,
                    inference_detail="high"
                )
                copied_ctx.add_message(
                    role="user",
                    content=[f"{position.title()} view of user during speech:", image_content]
                )

                logger.info(f"Added {position} frame to chat context")
        else:
            # No frames available - user is not sharing their screen
            copied_ctx.add_message(
                role="system",
                content="The user is not currently sharing their screen. Let them know they need to share their screen for you to provide visual assistance."
            )
            logger.warning("No captured frames available for this conversation")

        

        #messages = openai.utils.to_chat_ctx(copied_ctx, cache_key=self.llm)
        messages = self.llm.chat(chat_ctx=copied_ctx)
        
        generation = self.get_current_trace().generation(
            name="llm_generation",
            model="gpt-4.1",
            #model="gpt-4.1-mini",
            input=messages,
        )
        output = ""
        set_completion_start_time = False
        try:
            async for chunk in Agent.default.llm_node(self, copied_ctx, tools, model_settings):
                if not set_completion_start_time:
                    generation.update(
                        completion_start_time=datetime.now(UTC),
                    )
                    set_completion_start_time = True
                if chunk.delta and chunk.delta.content:
                    output += chunk.delta.content
                yield chunk
        except Exception as e:
            generation.update(level="ERROR")
            logger.error(f"LLM error: {e}")
            raise
        finally:
            generation.end(output=output)
            print (f"output: {output}")

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        span = self.get_current_trace().span(name="tts_node", metadata={"model": "deepgram"})
        try:
            async for event in Agent.default.tts_node(self, text, model_settings):
                yield event

            print (f"Text :{text}")
        except Exception as e:
            span.update(level="ERROR")
            logger.error(f"TTS error: {e}")
            raise
        finally:
            span.end()

    def on_track_subscribed(
        self,
        track: rtc.RemoteTrack,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if publication.source != rtc.TrackSource.SOURCE_SCREENSHARE:
            return
        logger.info("Screen share track subscribed")

        # start the new stream
        asyncio.create_task(self.read_video_stream(rtc.VideoStream(track)))

    async def read_video_stream(self, video_stream: rtc.VideoStream) -> None:
        # close open streams
        await self.close_video_stream()
        self.video_stream = video_stream

        logger.info("Starting video frame capture")
        frame_count = 0
        async for event in video_stream:
            # Capture frames at 1 per second
            current_time = time.time()
            if current_time - self.last_frame_time >= 1.0:
                # Store the frame and update time
                frame = event.frame
                self.frames.append(frame)
                self.last_frame_time = current_time

                frame_count += 1
                logger.info(f"Captured frame #{frame_count}: {frame.width}x{frame.height}")
        logger.info(f"Video frame capture ended - captured {frame_count} frames")

    def current_frames(self) -> List[rtc.VideoFrame]:
        # Add strategic frames from the conversation to provide better context
        # We'll use the first and last frames if available, plus a middle frame for longer sequences
        current_frames = []
        if len(self.frames) > 0:
            # Always use the most recent frame
            current_frames.append(("most recent", self.frames[-1]))

            # For sequences with multiple frames, also include the first frame
            if len(self.frames) >= 3:
                current_frames.append(("first", self.frames[0]))

                # For longer sequences (5+ frames), also include a middle frame
                if len(self.frames) >= 5:
                    mid_idx = len(self.frames) // 2
                    current_frames.append(("middle", self.frames[mid_idx]))
        logger.info(f"Adding {len(current_frames)} frames to conversation (from {len(self.frames)} available)")
        # clear the frames after using them to avoid memory bloat
        self.frames = []
        # return frames in reverse order so earliest frames appear first in context
        return list(reversed(current_frames))


async def entrypoint(ctx: JobContext) -> None:
    # Connect to the room
    await ctx.connect()

    logger.info(f"Connected to room: {ctx.room.name}")
    logger.info(f"Local participant: {ctx.room.local_participant.identity}")

    if len(ctx.room.remote_participants) == 0:
        logger.info("No remote participants in room, exiting")
        return

    logger.info(f"Found {len(ctx.room.remote_participants)} remote participants")
    # Create a simple agent session without custom frame rate
    # Just use the default settings which should work fine
    session = AgentSession()

    # Configure agent
    agent = VideoAgent(instructions=INSTRUCTIONS, room=ctx.room)
    
    # Set up room input/output - explicitly enable all modes
    room_input = RoomInputOptions(
        video_enabled=True,
        audio_enabled=True
    )
    
    room_output = RoomOutputOptions(
        audio_enabled=True,
        transcription_enabled=True
    )

    # Start the agent with all capabilities
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=room_input,
        room_output_options=room_output,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
