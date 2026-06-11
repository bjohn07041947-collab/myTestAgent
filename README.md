# LiveKit AI Video Agent

A complete AI-powered video conferencing application that combines LiveKit's real-time infrastructure with advanced AI capabilities including speech-to-text, language models, and text-to-speech. This project demonstrates how to build an AI agent that can see, hear, and interact with users through video calls.

## Features

- **Real-time Video & Audio**: Built on LiveKit's robust infrastructure
- **AI-Powered Conversations**: Uses GPT-4 with video context awareness
- **Voice Interaction**: Deepgram STT and Cartesia TTS for natural conversations
- **Knowledge Base Integration**: Built-in knowledge system for domain-specific assistance
- **Screen Sharing Support**: Agent can see and respond to shared screens
- **Observability**: Integrated with Langfuse for monitoring and analytics


## Architecture

The application consists of four main components:

1. **Frontend (React)**: User interface built with LiveKit Components for video conferencing
2. **Backend API (FastAPI)**: REST API server for token generation and room management
3. **LiveKit Server**: Real-time infrastructure for video/audio streaming and room coordination
4. **Video Agent (LiveKit Agent)**: AI-powered agent with STT, LLM, TTS, and knowledge base integration

```mermaid
sequenceDiagram
    participant F as Frontend (React)
    participant API as Backend API (FastAPI)
    participant LK as LiveKit Server
    participant A as Video Agent (LiveKit Agent)

    F->>F: User enters name and clicks Connect
    F->>API: GET /api/get-token?participant=name
    API->>LK: Create room with UUID
    API->>API: Generate access token with room permissions
    API-->>F: Return {token, url}

    F->>LK: Connect to LiveKit room with token
    LK->>A: Agent detects new participant
    A->>A: Initialize VideoAgent with instructions
    A-->>F: Send greeting via TTS

    Note over F,A: Video Conference Session
    F->>LK: Share screen (optional)
    LK->>A: Subscribe to screen share track
    A->>A: Capture video frames (1 fps)

    F->>LK: Send audio (user speech)
    A->>A: STT - Convert speech to text
    A->>A: LLM - Process with video frames
    A->>A: TTS - Convert response to audio
    A-->>F: Send audio response via LiveKit

    F->>LK: Disconnect
    A->>A: Clean up resources and exit
```

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- LiveKit server instance
- API keys for: OpenAI, Deepgram, Cartesia, Langfuse

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd livekit-video
```

2. Set up environment variables:
```bash
# Create .env file in the root directory
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
LIVEKIT_URL=wss://your-livekit-server.com

OPENAI_API_KEY=your_openai_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
CARTESIA_API_KEY=your_cartesia_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=your_langfuse_host
```

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the FastAPI server:
```bash
python api.py
```

5. In a separate terminal, start the LiveKit agent:
```bash
python video_agent.py dev
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser to `http://localhost:3000`

## Usage

### Basic Setup
1. **Join a Room**: Enter your name and click "Connect"
2. **Enable Screen Share**: Share your screen for the agent to see
3. **Start Talking**: The agent will respond using voice and can reference what it sees

## End-to-End Test Execution

The agent can run UI test cases defined in markdown files against the screen.
It locates elements visually using iterative grid refinement: a coarse 10x10
grid on the full screenshot picks the region as a cell range (every cell the
element touches), then that range is cropped, upscaled, and re-gridded —
repeating until the grid is finer than the needed click precision. The click
lands on the center of the final cell range, so it works for small
boundary-straddling targets (radio buttons) and wide ones (full-width inputs)
alike. No per-element template images are required.

> Note: the agent process must run on the same machine whose screen is being
> tested — screenshots and clicks happen locally via pyautogui. On macOS, grant
> the terminal Screen Recording and Accessibility permissions.

### Writing test cases

Add a markdown file to `backend/testcases/`:

```markdown
# Payment Calculator

1. Click the radio button under financing type
2. Enter "20000.00" in the financing amount input field
3. Click the credit score dropdown
4. Click the "Excellent" option in the credit score dropdown list
5. Wait 3 seconds
6. Verify an estimated payment amount is displayed
```

Supported step verbs: `Click/Select/Choose/Press`, `Type/Enter "<text>" [in <field>]`,
`Verify/Check <expectation>`, `Wait <n> seconds`.

### Running tests

Say "run the calculator test" during a session. The agent finds the
matching test case, executes each step (screenshot → locate → click/type →
verify), and reports the results by voice. A full report is written to
`backend/results/<testcase>-<timestamp>.md`, with screenshots saved for any
failed steps.

Agent tools: `list_test_cases`, `run_test_case`, `click_element`, `type_text`,
`verify_screen`.

Set `LOCATOR_DEBUG=1` to save the annotated grid images of the last locate call
to `backend/debug/`.

## Key Components

### Video Agent (`backend/video_agent.py`)

The core AI agent that handles:
- Video frame processing during user speech
- Integration with OpenAI's GPT-4 for visual understanding
- Speech-to-text with Deepgram
- Text-to-speech with Cartesia
- Langfuse observability tracking

Key features:
- Captures frames at 1fps continuously
- Sends multiple frames (first, middle, most recent) for better context
- Includes comprehensive error handling and logging

### API Server (`backend/api.py`)

FastAPI server providing:
- Token generation for LiveKit rooms
- Room creation and management
- CORS configuration for frontend access

### Knowledge Manager (`backend/knowledge_manager.py`)

Manages domain-specific knowledge:
- Loads markdown files from the knowledge directory
- Formats content for AI context
- Supports multiple knowledge domains (application)

### Frontend (`frontend/src/App.tsx`)

React application featuring:
- LiveKit room connection
- Video conference interface
- Token-based authentication
- Real-time video and audio

## Knowledge Base

The application includes a knowledge base system for domain-specific assistance:


## Customization

### Swapping AI Components

The agent uses LiveKit's plugin system, making it easy to swap components:

```python
# Change STT provider
from livekit.plugins import assemblyai
stt=assemblyai.STT()

# Change LLM provider
from livekit.plugins import anthropic
llm=anthropic.LLM(model="claude-3-sonnet")

# Change TTS provider
from livekit.plugins import elevenlabs
tts=elevenlabs.TTS(voice="your-voice-id")
```

### Adding Knowledge Domains

1. Create a new markdown file in `backend/knowledge/`
2. Update the `knowledge_files` mapping in `KnowledgeManager`
3. The agent will automatically include the new knowledge in its context

### Modifying Agent Behavior

Update the `INSTRUCTIONS` constant in `video_agent.py` to change the agent's personality, capabilities, or domain focus.

## Observability

The application integrates with Langfuse for comprehensive observability:

- **Traces**: Complete conversation flows with timing
- **Spans**: Individual component performance (STT, LLM, TTS)
- **Generations**: LLM inputs/outputs with cost tracking
- **Error Tracking**: Detailed error logging and debugging

Access your Langfuse dashboard to monitor agent performance and optimize responses.

## Production Considerations

- **Environment Variables**: Secure API key management
- **Error Handling**: Comprehensive error recovery
- **Rate Limiting**: API quota management
- **Scaling**: LiveKit server cluster configuration
- **Security**: CORS configuration, token validation
- **Monitoring**: Langfuse integration for production insights

## API Dependencies

This project requires API keys from:

- **LiveKit**: Real-time infrastructure
- **OpenAI**: GPT-4 language model
- **Deepgram**: Speech-to-text
- **Cartesia**: Text-to-speech
- **Langfuse**: Observability (optional)

All services offer free tiers for development and testing.

## Business Applications

This technology enables various use cases:

- **Technical Support**: Visual troubleshooting with screen sharing
- **Customer Onboarding**: Interactive product walkthroughs
- **Remote Assistance**: Equipment setup and configuration
- **Training & Education**: Interactive software tutorials
- **Professional Services**: Real-time consulting with visual context

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For questions about this implementation, please [open an issue](link-to-issues).
For LiveKit-specific questions, visit the [LiveKit documentation](https://docs.livekit.io/).