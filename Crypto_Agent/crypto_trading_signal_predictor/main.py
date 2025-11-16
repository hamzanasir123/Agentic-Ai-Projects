import chainlit as cl
import json
import asyncio
from agents import Runner
from Agents.Triage_Agent import triage_agent, config as triage_agent_config


# -----------------------------------------
# SAFE SERIALIZATION FOR NUMPY/PANDAS TYPES
# -----------------------------------------
def normalize_value(v):
    try:
        if hasattr(v, "item"):     # numpy scalar
            return v.item()
        if hasattr(v, "tolist"):   # numpy array or pandas
            return v.tolist()
        return v
    except:
        return str(v)


# -----------------------------------------
# SUMMARIZER FOR TOOL OUTPUT (NO JSON)
# -----------------------------------------
def summarize_output(output):
    if isinstance(output, dict):
        lines = []
        for key, value in output.items():
            value = normalize_value(value)
            lines.append(f"- **{key}:** {value}")
        return "### 🔎 Tool Summary\n" + "\n".join(lines)

    if isinstance(output, list):
        return "### 🔎 Tool Output:\n" + ", ".join(map(str, output))

    # fallback
    return f"### 🔎 Tool Output:\n{normalize_value(output)}"


# -----------------------------------------
# Think animation
# -----------------------------------------
async def start_animation(msg):
    dots = 0
    while True:
        await asyncio.sleep(0.45)
        dots = (dots + 1) % 4
        msg.content = "🤖 Thinking" + ("." * dots)
        await msg.update()


# -----------------------------------------
# CHAT START
# -----------------------------------------
@cl.on_chat_start
async def start():
    await cl.Message(content="👋 Hi! The Triage Agent is ready.").send()


# -----------------------------------------
# MAIN MESSAGE HANDLER
# -----------------------------------------
@cl.on_message
async def main(message: cl.Message):

    # Build conversation context
    history = cl.user_session.get("history", [])
    history.append({"sender": "user", "text": message.content})
    cl.user_session.set("history", history)

    conversation_text = "\n".join(f"{m['sender']}: {m['text']}" for m in history)

    # Thinking animation message
    thinking_msg = cl.Message(content="🤖 Thinking", author="bot")
    await thinking_msg.send()
    animation_task = asyncio.create_task(start_animation(thinking_msg))

    # Placeholder for streamed model answer
    msg = cl.Message(content="", author="bot")
    await msg.send()

    # Run agent with streaming
    result = Runner.run_streamed(
        starting_agent=triage_agent,
        input=conversation_text,
        run_config=triage_agent_config
    )

    first_token = True

    async for event in result.stream_events():
        event_name = type(event).__name__

        # ----------------------------
        # Model text streaming
        # ----------------------------
        if event_name == "RawResponsesStreamEvent":
            event_type = type(event.data).__name__

            if event_type == "ResponseTextDeltaEvent":
                delta = event.data.delta
                if delta:
                    if first_token:
                        animation_task.cancel()
                        await thinking_msg.remove()
                        first_token = False
                    await msg.stream_token(delta)

            elif event_type == "ResponseCompletedEvent":
                await msg.update()

        # ----------------------------
        # Tool output detected
        # ----------------------------
        elif hasattr(event, "item") and event.item.type == "tool_call_output_item":
            output = event.item.output

            if first_token:
                animation_task.cancel()
                await thinking_msg.remove()
                first_token = False

            summary_text = summarize_output(output)
            await msg.stream_token("\n\n" + summary_text + "\n")

    # Final update
    await msg.update()

    # Save bot response in memory
    history.append({"sender": "bot", "text": msg.content})
    cl.user_session.set("history", history)
