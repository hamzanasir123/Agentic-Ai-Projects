import re
from agents import function_tool
from functions.data_collector_tool import get_coin_details
from functions.apply_indicators_strategies_and_visualize import apply_indicators_strategies_and_visualize

NAME_TO_SYMBOL = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "ada": "ADA",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    # add more as needed
}

DEFAULT_QUOTE = "USD"

def normalize_input(user_input: str) -> str:
    """
    Normalize user input into SYMBOL/QUOTE format.
    Example: "give me prediction on BTC/USD for next 12 hours" -> "BTC/USD"
    """
    match = re.search(r"([a-zA-Z]+)[/\- ]?([a-zA-Z]+)?", user_input)
    if not match:
        return None

    base = match.group(1).lower()
    quote = DEFAULT_QUOTE.lower()

    base_sym = NAME_TO_SYMBOL.get(base, base.upper())
    quote_sym = NAME_TO_SYMBOL.get(quote, quote.upper())

    return f"{base_sym}/{quote_sym}"



@function_tool
async def get_predictions_tool(input: str):
    pair = normalize_input(input)
    if not pair:
        return "Invalid input format. Use format like 'BTC/USDT' or just 'BTC'."
    print(f"Fetching predictions for: {pair}")
    response = await get_coin_details(pair)
    print("OHLCV Data Retrieved:")
    if response.ohlcv is None:
        print("Error:", response.error)
        return
    result = apply_indicators_strategies_and_visualize(response.ohlcv) 
    print("Indicators and Strategies Applied:")
    return result
