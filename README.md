# Gemrag - Streamlit Chat Interface

## Overview
A production-ready chat application with:
- Multiple session management
- Real-time response streaming
- Token usage tracking
- OpenRouter API integration

## Installation
```bash
pip install streamlit python-dotenv openai tiktoken
```

## Usage
1. Get OpenRouter API key from [OpenRouter.ai](https://openrouter.ai/)
2. Create `.env` file:
```env
OPENROUTER_API_KEY=your_key_here
```
3. Run the application:
```bash
streamlit run app.py
```

## Features
- 💬 Multiple chat sessions with persistent history
- 🔢 Token counting for cost tracking
- ⚡ Real-time response streaming
- 🗑️ Session management (create/delete)
- 🚨 Error handling with auto-recovery
