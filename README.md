# Yoga Institute AI Assistant 🧘

A professional AI-powered chatbot that helps users find information about yoga institutes, their classes, subscriptions, schedules, and more.

## Features

- 💬 Conversational AI with context memory
- 🔍 RAG (Retrieval Augmented Generation) powered by Qdrant vector database
- 📊 Real-time token usage and cost tracking
- 🎨 Clean and intuitive Streamlit interface
- 🔒 Secure API key management

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4o-mini
- **Vector Database**: Qdrant
- **Embeddings**: OpenAI text-embedding-3-small

## Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys in `.streamlit/secrets.toml` (already configured locally)

4. Run the app:
```bash
streamlit run app.py
```

## Deployment on Streamlit Cloud

1. Push your code to GitHub (make sure `.streamlit/secrets.toml` is in `.gitignore`)

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Click "New app" and connect your GitHub repository

4. Set up secrets in Streamlit Cloud:
   - Go to App settings → Secrets
   - Add the following secrets:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key"
   QDRANT_URL = "your-qdrant-url"
   QDRANT_API_KEY = "your-qdrant-api-key"
   ```

5. Deploy!

## Usage

- Start with a greeting to get an introduction
- Ask about specific yoga institutes
- Request "What institutes are available?" to see all certified institutes
- Ask follow-up questions - the bot remembers your conversation context
- Use "New Chat" button to start a fresh conversation

## Cost Tracking

The app displays real-time token usage and costs:
- GPT-4o-mini pricing: $0.150/1M input tokens, $0.600/1M output tokens
- View per-message costs in expandable sections
- Track total session costs in the sidebar

## License

MIT
