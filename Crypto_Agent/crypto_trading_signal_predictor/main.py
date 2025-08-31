import asyncio
from agents import Runner
from Agents.Triage_Agent import triage_agent, config as triage_agent_config


# --- Simple CLI instead of Chainlit ---
conversation_memory = []


async def main():
    print("TRIAGE AGENT INITIALIZING...")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        conversation_memory.append({"sender": "user", "text": user_input})
        conversation_text = "\n".join([f"{m['sender']}: {m['text']}" for m in conversation_memory])
        print("🤖 Thinking...")
        response = await Runner.run(
            starting_agent=triage_agent,
            input=conversation_text,
            run_config=triage_agent_config
        )
        print(response.final_output)
        conversation_memory.append({"sender": "bot", "text": response.final_output})


if __name__ == "__main__":
    asyncio.run(main())
