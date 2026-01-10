import asyncio
import os
import warnings
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.arxiv_toolkit import ArxivToolkit
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ssl

# Fix for SSL: CERTIFICATE_VERIFY_FAILED
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Silence the DDGS library renaming warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="phi.tools.duckduckgo")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API Key (MUST be set via environment variable)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required. Set it in Railway or your .env file")
MODEL_ID = "llama-3.1-8b-instant"  # High TPM limit, fast and free on Groq

# --- AGENT 1: THE RESEARCHER (Updated to use Groq) ---
researcher = Agent(
    name="Researcher",
    model=Groq(id=MODEL_ID, api_key=GROQ_API_KEY),
    tools=[ArxivToolkit()], # Disabling read_arxiv_papers to avoid token blowout
    instructions=[
        "You are an academic researcher.",
        "Search arxiv for the most significant AI paper from the last 7 days.",
        "Crucial: Use the metadata and abstract provided by the search tool. Do NOT attempt to read the full PDF content to avoid token limits.",
        "Return a COMPREHENSIVE report containing:",
        "1. Full Title",
        "2. Arxiv PDF Link (Must be a valid URL starts with http)",
        "3. Detailed Abstract",
        "4. Key Contributions & Methodology (based on abstract)",
        "5. Potential Impact",
        "CRITICAL: Any numeric parameters (like max_results) MUST be passed as numbers, NOT strings.",
        "CRITICAL: Return ONLY the content. Start directly with the Title."
    ]
)

# --- AGENT 2: THE TEACHER ---
teacher = Agent(
    name="Teacher",
    model=Groq(id=MODEL_ID, api_key=GROQ_API_KEY),
    instructions=[
        "You are a witty, charismatic science communicator (think VSauce or Feynman).",
        "Take the technical research provided and explain it like you're talking to a friend at a coffee shop.",
        "Use a vivid, creative analogy (cooking, sports, sci-fi, video games) to strip away the jargon.",
        "Be relatable, maybe a bit funny. Focus on the 'Aha!' moment.",
        "CRITICAL: Jump straight into the story. Don't say 'I will explain this'. Just say something like 'Think of it like...' or 'Imagine you are...'"
    ]
)

# --- AGENT 3: THE DESIGNER ---
designer = Agent(
    name="Designer",
    model=Groq(id=MODEL_ID, api_key=GROQ_API_KEY),
    instructions=[
        "You are a top-tier generic Social Media Manager.",
        "Write a notification that genuinely makes people pause and look.",
        "Don't sound like a bot. Be punchy, intriguing, and human.",
        "Format: '[Hook] - [The Real value]'",
        "Example: 'AI just got a memory upgrade. This new technique lets findings stick forever.'",
        "CRITICAL: Return ONLY the final notification text."
    ]
)

from pydantic import BaseModel

# --- SHARED STATE ---
state = {
    "news": "Ready to search...",
    "link": "#",
    "analogy": "Waiting for news...",
    "notification": "Standby...",
    "status": "Idle",
    "date": "Today",
    "interval_minutes": 5
}

# Default to 5 minutes
UPDATE_INTERVAL = 300 

class Config(BaseModel):
    interval_minutes: int

async def run_agent_team():
    global state, UPDATE_INTERVAL
    import re
    from datetime import datetime

    while True:
        try:
            # Step 1: Research
            state["status"] = "🔍 Agent 1: Scanning Arxiv..."
            
            # Explicitly instructing to use integers for numeric parameters and focus on abstracts
            news_res = researcher.run("Search Arxiv for the top 3 most significant AI research papers from the last 7 days. Summarize the best one based on its abstract.")
            state["news"] = news_res.content
            
            # Helper to extract link
            link_match = re.search(r'http[s]?://arxiv\.org/pdf/[\w\.]+', news_res.content)
            if not link_match:
                link_match = re.search(r'http[s]?://arxiv\.org/abs/[\w\.]+', news_res.content)
            
            
            
            state["link"] = link_match.group(0) if link_match else "https://arxiv.org"

            # Step 2: Teach
            state["status"] = "💡 Agent 2: Creating Analogy..."
            analogy_res = teacher.run(f"Explain this research using an analogy: {news_res.content}")
            state["analogy"] = analogy_res.content

            # Step 3: Design
            state["status"] = "🎨 Agent 3: Aligning Notification..."
            notif_res = designer.run(f"Create a notification based on this Analogy: '{analogy_res.content}'. \n\n But ensure it is grounded in this Real Research: '{news_res.content}'")
            state["notification"] = notif_res.content
            
            # Add formatted Time AFTER notification is complete
            now = datetime.now()
            state["date"] = now.strftime("%b %d, %Y")
            state["time"] = now.strftime("%I:%M %p") # e.g. 06:30 PM

            state["status"] = "✅ Cycle Complete"
            print(f"Cycle Done. Notification: {state['notification']}")
            
            # Wait dynamic interval
            print(f"Waiting {UPDATE_INTERVAL} seconds...")
            await asyncio.sleep(UPDATE_INTERVAL) 

        except Exception as e:
            print(f"❌ Error in Agent Team: {e}")
            state["status"] = f"Error: {e}"
            await asyncio.sleep(30)

@app.on_event("startup")
async def startup():
    asyncio.create_task(run_agent_team())

@app.get("/")
async def root():
    return {"message": "AI_AGENT_IS_LIVE_ON_RENDER", "status": "Ready"}

@app.get("/api/latest")
async def get_latest():
    return state

@app.post("/api/config")
async def set_config(config: Config):
    global UPDATE_INTERVAL, state
    # Update global interval (convert minutes to seconds)
    UPDATE_INTERVAL = config.interval_minutes * 60
    state["interval_minutes"] = config.interval_minutes
    return {"message": "Interval updated", "interval_seconds": UPDATE_INTERVAL}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
