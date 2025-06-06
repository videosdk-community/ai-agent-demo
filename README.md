# AI Voice Agent - Intelligent Real-Time Communication Platform

![AI Voice Agent Interface](https://strapi.videosdk.live/uploads/ai_agent_demo_86ae1b7d09.png)

A powerful **AI voice agent** platform that combines the intelligence of **AI agents LLM** with real-time voice communication capabilities. Build sophisticated voice-enabled applications for customer service, sales automation, and interactive assistants.

## Features

- **Advanced AI Voice Agent**: Powered by Google's Gemini Live model
- **Real-Time Voice Communication**: Seamless audio processing with VideoSDK
- **Customizable Personality**: Configure voice, tone, and behavior
- **AI Outbound Calling**: Perfect for automated sales and support calls
- **AI Cold Calling**: Intelligent conversation starters and follow-ups
- **AI Voice Creator**: Generate voices from samples with various personalities
- **AI Voicemail Generator Free**: Automated voicemail creation and responses
- **AI Executive Assistant**: Handle scheduling, reminders, and administrative tasks
- **Goal Based Agent**: Configure agents with specific objectives and KPIs
- **News Reporter Text to Speech**: Professional broadcasting voice capabilities
- **Text to Speech Old Man**: Various voice profiles including elderly personas

## Technology Stack

- **Frontend**: React + TypeScript + Vite
- **Backend**: Python FastAPI
- **AI Engine**: Google Gemini Live
- **Real-Time Communication**: VideoSDK
- **Voice Processing**: Gemini Realtime API

## Prerequisites

- Python 3.12 or higher
- Node.js 18+ and npm/yarn
- Google API Key (Gemini)
- VideoSDK Token

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/videosdk-community/ai-agent-demo
cd ai-agent-demo
```

### 2. Use Cases

#### Business Applications

- **AI Sales Team**: Automate lead qualification and follow-up calls
- **Customer Support**: 24/7 intelligent voice assistance
- **Appointment Scheduling**: Voice-enabled booking systems
- **Market Research**: Automated survey and feedback collection

#### Communication Enhancement

- **Popular AI Voices**: Multiple voice options for different scenarios
- **Interactive Voice Response**: Smart call routing and handling
- **Voice-Enabled Chatbots**: Seamless text-to-voice conversation

#### AI Agent vs AI Assistant

This platform bridges the gap between traditional **AI assistants** and autonomous **AI agents**, offering:

- Proactive conversation initiation
- Context-aware responses
- Goal-oriented interactions
- Adaptive personality traits

## Server Setup

### Create Virtual Environment and Install Dependencies

```bash
# Create virtual environment (requires Python 3.12+)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and configure
cp .env.example .env
# Edit .env file and add your credentials:
# GOOGLE_API_KEY="your-google-api-key-here"
# PORT="8000"

# Start the server
python server.py
```

## Client Setup

### Install Dependencies and Configure Environment

```bash
# Navigate to client directory
cd client

# Install dependencies
npm install
# or
yarn install

# Copy environment template and configure
cp .env.example .env
# Edit .env file and add:
# VITE_VIDEOSDK_TOKEN="your-videosdk-token-from-app.videosdk.live"
# VITE_API_URL="http://localhost:8000"  # or your server URL

# If testing with local server, use ngrok:
# ngrok http 8000
# Then use the ngrok URL in VITE_API_URL

# Start the client
npm run dev
# or
yarn dev
```

## Configuration Options

### Agent Personality Settings

- **Voice Selection**: Choose from various AI voice profiles
- **Temperature**: Control response creativity (0.0 - 1.0)
- **Top-P & Top-K**: Fine-tune response generation
- **System Prompt**: Define agent behavior and personality
- **Response Modalities**: Configure audio-only or multi-modal responses

### Meeting Configuration

```typescript
interface MeetingConfig {
  meeting_id: string;
  token: string;
  model: string;
  voice: string;
  personality: string;
  temperature: number;
  system_prompt: string;
  topP: number;
  topK: number;
}
```

## API Endpoints

### Join Agent

```http
POST /join-agent
Content-Type: application/json

{
  "meeting_id": "your-meeting-id",
  "token": "your-videosdk-token",
  "model": "gemini-2.0-flash-exp",
  "voice": "Puck",
  "personality": "friendly",
  "temperature": 0.7,
  "system_prompt": "You are a helpful AI assistant...",
  "topP": 0.9,
  "topK": 40
}
```

### Leave Agent

```http
POST /leave-agent
Content-Type: application/json

{
  "meeting_id": "your-meeting-id"
}
```

## Key Components

### Backend Components

- **MyVoiceAgent**: Main agent class handling conversation flow
- **AgentSession**: Manages agent lifecycle and state
- **RealTimePipeline**: Processes real-time audio streams
- **GeminiRealtime**: Integration with Google's Gemini Live model

### Frontend Components

- **AgentMeeting**: Main meeting interface component
- **Toast System**: User feedback and notifications
- **Mobile Responsive**: Optimized for all device types

## Security Considerations

- Environment variables for sensitive API keys
- CORS configuration for secure cross-origin requests
- Token-based authentication for VideoSDK integration
- Input validation and error handling

## Troubleshooting

### Common Issues

1. **Python Version**: Ensure you're using Python 3.12 or higher
2. **API Keys**: Verify your Google API key and VideoSDK token are correct
3. **Network Issues**: Use ngrok for local development if encountering connection issues
4. **Dependencies**: Make sure all pip and npm dependencies are installed

### Debug Mode

Enable debug logging by setting environment variables:

```bash
export DEBUG=true
export LOG_LEVEL=debug
```

## Performance Optimization

- **Connection Pooling**: Efficient WebRTC connection management
- **Background Tasks**: Non-blocking agent operations
- **Session Management**: Optimized memory usage for multiple concurrent sessions
- **Error Recovery**: Automatic reconnection and graceful degradation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:

- Check the [VideoSDK Documentation](https://docs.videosdk.live/)
- Create an issue in this repository

## What's Next?

- Multi-language support
- Advanced analytics and reporting
- Custom voice training capabilities
- Integration with popular CRM systems
- Enhanced AI personality customization

**Ready to revolutionize your communication with AI voice agents?** Get started now and build the future of intelligent voice interactions!
