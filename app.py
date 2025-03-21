import os
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
import logging

# Load environment variables
load_dotenv()

# Get Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ API key for Groq is missing. Please set the GROQ_API_KEY in the .env file.")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to a specific domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq API client
client = Groq(api_key=GROQ_API_KEY)

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conversation class to store chat history
class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a useful AI assistant."}
        ]
        self.active: bool = True

# Store conversations in memory (temporary)
conversations: Dict[str, Conversation] = {}

# Pydantic model for user input
class UserInput(BaseModel):
    message: str
    role: str = "user"
    conversation_id: str

# Function to get or create a conversation session
def get_or_create_conversation(conversation_id: str) -> Conversation:
    if conversation_id not in conversations:
        conversations[conversation_id] = Conversation()
    return conversations[conversation_id]

# Function to query Groq API and stream response
async def stream_groq_response(conversation: Conversation):
    """
    Streams response from Groq API in real-time.
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversation.messages,
            temperature=0.8,
            max_tokens=1024,
            top_p=1,
            stream=True,
        )

        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"❌ Error with Groq API: {str(e)}")
        yield "❌ Error generating response."

# Chatbot API Endpoint
@app.post("/chat/")
async def chat(input: UserInput):
    """
    Processes user input and streams the chatbot's response.
    """
    conversation = get_or_create_conversation(input.conversation_id)

    if not conversation.active:
        raise HTTPException(status_code=400, detail="Chat session has ended. Please start a new session.")

    # Append user message to conversation history
    conversation.messages.append({"role": input.role, "content": input.message})

    return StreamingResponse(stream_groq_response(conversation), media_type="text/plain")

# Health Check API
@app.get("/health")
async def health_check():
    return {"status": "✅ Server is running!"}

# Run FastAPI with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
