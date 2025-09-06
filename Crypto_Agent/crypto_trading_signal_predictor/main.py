import chainlit as cl
import json
import asyncio
from agents import ItemHelpers, Runner
from Agents.Triage_Agent import triage_agent, config as triage_agent_config


@cl.on_chat_start
async def start():
    await cl.Message(content="👋 Hi! The Triage Agent is ready.").send()


@cl.on_message
async def main(message: cl.Message):
    # Gather conversation history
    conversation = []
    for m in cl.user_session.get("history", []):
        conversation.append(f"{m['sender']}: {m['text']}")
    conversation.append(f"user: {message.content}")
    conversation_text = "\n".join(conversation)

    # Store user input
    history = cl.user_session.get("history", [])
    history.append({"sender": "user", "text": message.content})
    cl.user_session.set("history", history)

    # 🔹 Show "thinking..." placeholder with animated dots
    thinking_msg = cl.Message(content="🤖 Thinking", author="bot")
    await thinking_msg.send()

    async def animate_dots(msg: cl.Message):
        dots = 0
        while True:
            await asyncio.sleep(0.5)
            dots = (dots + 1) % 4
            msg.content = "🤖 Thinking" + ("." * dots)
            await msg.update()

    task = asyncio.create_task(animate_dots(thinking_msg))

    # 🔹 Prepare streaming message (final bot answer)
    msg = cl.Message(content="", author="bot")
    await msg.send()

    # Run with streaming
    result = Runner.run_streamed(
        starting_agent=triage_agent,
        input=conversation_text,
        run_config=triage_agent_config
    )

    first_token = True
    async for event in result.stream_events():
        event_type = type(event).__name__

        if event_type == "RawResponsesStreamEvent":
            data_type = type(event.data).__name__

            if data_type == "ResponseTextDeltaEvent":
                delta = event.data.delta
                if delta:
                    if first_token:
                        task.cancel()
                        await thinking_msg.remove()
                        first_token = False
                    await msg.stream_token(delta)

            elif data_type == "ResponseCompletedEvent":
                await msg.update()

        elif hasattr(event, "item"):
            if event.item.type == "tool_call_output_item":
                output = event.item.output
                if first_token:
                    task.cancel()
                    await thinking_msg.remove()
                    first_token = False

                if isinstance(output, dict):
                    if all(isinstance(v, list) for v in output.values()):
                        await cl.DataTable(
                            name="Tool Result",
                            columns=list(output.keys()),
                            rows=list(zip(*output.values()))
                        ).send()
                    elif set(output.keys()) >= {"x", "y"}:
                        await cl.Chart(
                            name="Tool Chart",
                            type="line",
                            data={
                                "labels": output["x"],
                                "datasets": [{"label": "Values", "data": output["y"]}]
                            }
                        ).send()
                    else:
                        # JSON → code block inside same msg (avoid new message)
                        await msg.stream_token(
                            f"\n```json\n{json.dumps(output, indent=2)}\n```"
                        )
                else:
                    await msg.stream_token(str(output))


    await msg.update()

    # Save final response
    history.append({"sender": "bot", "text": msg.content})
    cl.user_session.set("history", history)
