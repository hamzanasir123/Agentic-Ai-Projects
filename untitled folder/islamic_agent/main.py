from agents import Agent, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from tools import search_quran, search_hadith, search_lectures
from prompts import build_prompt_for_query, SYSTEM_PROMPT
from retrieval import vector_db
from moderation import moderate_text
from audio import transcribe_audio, synthesize_tts
import os
import requests  # fallback for manual API calls

# --- Load environment variables ---
load_dotenv()

with open("Instructions/triage_agent.md", "r") as file:
    triage_agent_instructions = file.read()

# Load Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set in environment variables")

# --- FastAPI App ---
app = FastAPI(title="Islamic Knowledge Agent (prototype)")


# --- Request/Response Models ---
class ChatRequest(BaseModel):
    query: str
    language: str = "en"
    include_sources: bool = True
    mode: Optional[str] = "text"  # "text" or "voice"
    prefer_sources: Optional[List[str]] = None  # e.g., ["quran","tafsir"]

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    flagged: bool = False


# --- Gemini Client Setup ---
external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

config = RunConfig(model=model, model_provider=external_client)

triage_agent = Agent(
    name="Triage Agent",
    instructions=triage_agent_instructions,
    model=model
)


# --- LLM Call Wrapper ---
async def call_llm(prompt: str, max_tokens: int = 800) -> str:
    """
    Calls Gemini through the OpenAI-compatible API client.
    """
    try:
        response = await external_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")


# --- Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 1. Moderation pre-check
    flagged, reason = moderate_text(req.query)
    if flagged:
        raise HTTPException(status_code=400, detail=f"Query flagged: {reason}")

    # 2. Retrieval
    filters = {}
    if req.prefer_sources:
        filters["source_type"] = req.prefer_sources

    docs = vector_db.similarity_search(req.query, top_k=8, filters=filters)

    # 3. Build prompt
    prompt = build_prompt_for_query(req.query, docs, SYSTEM_PROMPT, language=req.language)

    # 4. LLM generation
    llm_out = await call_llm(prompt)

    # 5. Post-moderation
    out_flagged, out_reason = moderate_text(llm_out)

    response = ChatResponse(
        answer=llm_out,
        citations=[{"id": d["id"], "type": d["type"], "meta": d.get("meta")} for d in docs],
        confidence=vector_db.estimate_confidence(docs),
        flagged=out_flagged
    )

    # 6. If voice requested, synthesize
    if req.mode == "voice":
        audio_bytes = synthesize_tts(llm_out, lang=req.language)
        response.answer += "\n\n[audio_generated]"

    return response


@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...), language: str = "en"):
    # 1. Transcribe
    audio_bytes = await file.read()
    transcript = transcribe_audio(audio_bytes, language=language)

    # 2. Reuse chat flow
    req = ChatRequest(query=transcript, language=language, mode="voice")
    return await chat(req)
