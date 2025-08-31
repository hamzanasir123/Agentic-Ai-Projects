import asyncio
import os
import re
from agents import Agent, RunConfig, Runner,OpenAIChatCompletionsModel
from dotenv import load_dotenv
from openai import AsyncOpenAI, InternalServerError
from tools.any_info_about_any_coin_tool import any_info_about_any_coin
from tools.news_about_crypto_tool import news_about_crypto
from tools.risk_management_tool import risk_management_tool
from Agents.Signal_Prediction_Agent import signal_predictor_agent
from Agents.Swing_Trading_Agent import swing_trade_agent


# --- Load environment variables ---
load_dotenv()

with open("Instructions/crypto_agent.md", "r") as file:
    crypto_agent_instructions = file.read()

# --- Gemini API Keys ---
GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY1"),
    os.getenv("GEMINI_API_KEY2"),
    os.getenv("GEMINI_API_KEY3")
]
current_key_index = 0

if not GEMINI_KEYS or GEMINI_KEYS == [""]:
    raise ValueError("No Gemini API keys found in .env file.")

# --- Coin Map ---
coin_map = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "dogecoin": "DOGE",
    "litecoin": "LTC",
    "tether": "USDT",
    "pkr": "PKR",
    "usd": "USD"
}


def detect_pair(text: str):
    text = text.lower()

    # Extend coin_map with tickers (reverse lookup)
    extended_map = {**coin_map}
    for name, ticker in coin_map.items():
        extended_map[ticker.lower()] = ticker

    # --- Match explicit pairs (BTC/USD, btc-usdt, ethusd) ---
    pair_match = re.findall(r"\b([a-z]{2,5})[-/ ]?([a-z]{2,5})\b", text)
    if pair_match:
        for base, quote in pair_match:
            if base in extended_map and quote in extended_map:
                return f"{extended_map[base]}/{extended_map[quote]}"

    # --- Single coin fallback (assume USD) ---
    for word in text.split():
        if word in extended_map:
            return f"{extended_map[word]}/USD"

    return None


def detect_intent_and_pair(user_input: str):
    text = user_input.lower()
    general_keywords = ["hello", "hi", "how are you", "good morning", "good evening", "thanks", "thank you"]
    if any(k in text for k in general_keywords):
        return "general", None

    trading_pair = detect_pair(text)

    if any(word in text for word in ["predict", "forecast", "future", "signal"]):
        intent = "prediction"
    elif any(word in text for word in ["price", "rate", "value", "worth", "current"]):
        intent = "price"
    elif any(word in text for word in ["news", "update", "article", "media"]):
        intent = "news"
    elif any(word in text for word in ["swing", "analysis"]):
        intent = "swing_analysis"
    elif any(word in text for word in ["risk", "management", "stoploss", "capital"]):
        intent = "risk_management"
    elif trading_pair:
        intent = "price"
    else:
        intent = "general"

    return intent, trading_pair

# --- Create the agent globally ---
crypto_manager_agent = Agent(
    name="Crypto Manager Agent",
    instructions=crypto_agent_instructions,
    tools=[
        any_info_about_any_coin,
        news_about_crypto,
        risk_management_tool
    ],
    handoffs=[
        signal_predictor_agent,
        swing_trade_agent      
        ],
    model=None  
)

async def run_agent(conversation_text, intent, pair):
    global current_key_index

    while current_key_index < len(GEMINI_KEYS):
        api_key = GEMINI_KEYS[current_key_index].strip()
        external_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        model = OpenAIChatCompletionsModel(
            model="gemini-2.5-flash",
            openai_client=external_client
        )

        config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)

        # ✅ inject model at runtime
        crypto_manager_agent.model = model

        try:
            result = await Runner.run(
                starting_agent=crypto_manager_agent,
                input=f"{conversation_text}\nIntent: {intent}\nPair: {pair}",
                run_config=config
            )
            return result.final_output

        except Exception as e:
            error_text = str(e).lower()
            if "quota" in error_text or "rate limit" in error_text:
                current_key_index += 1
                continue
            elif isinstance(e, InternalServerError) and "503" in str(e):
                await asyncio.sleep(5)
                continue
            else:
                return f"❌ Error: {str(e)}"
    return "❌ All Gemini keys exhausted."