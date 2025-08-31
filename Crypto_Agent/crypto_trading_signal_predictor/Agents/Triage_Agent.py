from agents import Agent, RunConfig,OpenAIChatCompletionsModel
from Agents.Crypto_Agent import crypto_manager_agent
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os

# --- Load environment variables ---
load_dotenv()

with open("Instructions/triage_agent.md", "r") as file:
    triage_agent_instructions = file.read()

# --- Gemini API Keys ---
GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY1"),
    os.getenv("GEMINI_API_KEY2"),
    os.getenv("GEMINI_API_KEY3")
]

if not GEMINI_KEYS or GEMINI_KEYS == [""]:
    raise ValueError("No Gemini API keys found in .env file.")

current_key_index = 0

api_key = GEMINI_KEYS[current_key_index].strip()
external_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

model = OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=external_client
    )

config = RunConfig(
    model=model, 
    model_provider=external_client
    )

triage_agent = Agent(
            name="Triage Agent",
            instructions=triage_agent_instructions,
            model=model,
            handoffs=[crypto_manager_agent]
        )
