from agents import Agent, RunConfig, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os
import chainlit as cl
from datetime import datetime

# --- Load environment variables ---
load_dotenv()

with open("Instructions/format_agent.md", "r") as file:
    format_agent_instructions = file.read()

# --- Gemini API Keys ---
GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY1"),
    os.getenv("GEMINI_API_KEY2"),
    os.getenv("GEMINI_API_KEY3")
]

if not any(GEMINI_KEYS) or GEMINI_KEYS == [""]:
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

format_agent = Agent(
    name="Format Agent",
    instructions=format_agent_instructions,
    model=model,
)

# --- Chat start ---
@cl.on_chat_start
async def start():
    await cl.Message(content="👋 Hi! The Format Agent is ready. Kindly give me a subject.").send()


# --- Handle user messages ---
@cl.on_message
async def main(message: cl.Message):
    subject = message.content.strip()

    # Get today's date in DD-MMM-YY format
    today = datetime.now().strftime("%d-%b-%y")

    # Build the user prompt with date + subject
    user_prompt = f"""
Date: {today}
Subject: {subject}
"""

    # Run the agent asynchronously
    response = await Runner.run(
        starting_agent=format_agent,
        run_config=config,
        input=user_prompt
    )

    # Display formatted SMS to user
    await cl.Message(content=response.final_output).send()
