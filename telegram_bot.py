"""
🌸 CRYPTO TRADING BOT · TELEGRAM INTERACTIVE BOT
=================================================
Telegram bot that accepts user requests and provides
LLM-powered market analysis with educational context.

Commands:
  /start   - Welcome message
  /analyze <TICKER> - Analyze a specific crypto
  /signal  - Get latest signals from trades.csv
  /help    - Show help

Setup:
  pip install python-telegram-bot pandas requests python-dotenv
  Set OPENROUTER_API_KEY in .env or environment
"""

import os
import json
import logging
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

# python-telegram-bot v20+
from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
TELEGRAM_BOT_TOKEN = "8039573395:AAEQN8zUF-OjUCeTEMdorLtet2HAwsGTYCk"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = "google/gemini-2.5-flash-lite"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
TRADES_CSV = "trades.csv"
FEATURES_CSV = "crypto_features_3months.csv"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================
# LLM ANALYSIS
# ============================================================
def call_llm_explanation(symbol: str, rsi: float, sma_50: float,
                          price: float, sentiment_label: str,
                          sentiment_score: float, news_titles: str,
                          timeframe: str = "1d") -> str:
    """
    Calls OpenRouter LLM to generate a natural language market explanation.
    Educational only — no direct buy/sell advice.
    """
    system_prompt = (
        "You are a cautious trading assistant providing educational analysis only. "
        "Never give direct buy/sell advice. Explain conditions and scenarios."
    )

    user_prompt = f"""Analyze this market data:

Symbol: {symbol}
Timeframe: {timeframe}
RSI: {rsi:.1f}
SMA_50: {sma_50:.4f}
Price: {price:.4f}
Sentiment: {sentiment_label} (score: {sentiment_score:.2f})
Recent News: {news_titles}

Provide:
1. Current market condition (2-3 sentences)
2. Key technical signals (2 bullet points)
3. Sentiment context from news
4. Possible scenarios: bullish/bearish/neutral with brief reasoning
5. Risk factors to watch

Format for mobile reading. Use emojis. Keep under 300 words."""

    if not OPENROUTER_API_KEY:
        return (
            "⚠️ <b>LLM недоступен</b> — не задан OPENROUTER_API_KEY.\n\n"
            f"📊 RSI: {rsi:.1f} | Price: {price:.4f} | Sentiment: {sentiment_label}"
        )

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.5,
            "max_tokens": 600,
        }
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"⚠️ Ошибка LLM: {e}"


def get_market_data(ticker: str):
    """
    Tries to get the latest row for a ticker from features CSV.
    Returns dict with market data or None.
    """
    try:
        df = pd.read_csv(FEATURES_CSV)
        ticker_upper = ticker.upper()
        df_ticker = df[df["ticker"].str.upper() == ticker_upper]
        if df_ticker.empty:
            return None
        row = df_ticker.sort_values("date").iloc[-1]
        return {
            "symbol": row["ticker"],
            "date": str(row["date"]),
            "price": float(row.get("close", 0)),
            "rsi": float(row.get("rsi", 50)),
            "sma_50": float(row.get("ma20", row.get("ma7", 0))),
            "sentiment_score": float(row.get("sentiment_score", 0) if "sentiment_score" in row else 0),
            "sentiment_label": "neutral",
            "news_titles": "No recent news available",
        }
    except Exception as e:
        logger.error(f"Error loading market data: {e}")
        return None


def get_latest_signals(n: int = 5):
    """
    Reads trades.csv and returns the latest N buy/sell signals.
    """
    try:
        df = pd.read_csv(TRADES_CSV)
        signals = df[df["decision"].isin(["buy", "sell"])].tail(n)
        return signals
    except Exception as e:
        logger.error(f"Error reading trades: {e}")
        return None


# ============================================================
# TELEGRAM HANDLERS
# ============================================================
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message."""
    text = (
        "🌸 <b>Crypto Trading Bot · Team 5</b>\n\n"
        "Я образовательный ассистент по крипторынку. "
        "Я анализирую технические индикаторы и новости, "
        "но <b>не даю инвестиционных советов</b>.\n\n"
        "<b>Команды:</b>\n"
        "  /analyze &lt;TICKER&gt; — анализ монеты (напр. <code>/analyze BTC</code>)\n"
        "  /signal — последние торговые сигналы системы\n"
        "  /tickers — список доступных монет\n"
        "  /help — справка\n\n"
        "💡 Просто напиши тикер (напр. <code>ETH</code>) и получи анализ!"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help message."""
    text = (
        "📖 <b>Справка</b>\n\n"
        "<b>/analyze TICKER</b>\n"
        "Получить LLM-анализ монеты: технические сигналы, "
        "настроение рынка, сценарии и риски.\n\n"
        "<b>/signal</b>\n"
        "Последние 5 сигналов бота (buy/sell) из backtesta.\n\n"
        "<b>/tickers</b>\n"
        "Показать все доступные монеты из датасета.\n\n"
        "⚠️ <i>Весь анализ носит образовательный характер. "
        "Не является инвестиционным советом.</i>"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def analyze_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /analyze <TICKER> — fetch market data and generate LLM explanation.
    """
    if not context.args:
        await update.message.reply_text(
            "❓ Укажи тикер, например: <code>/analyze BTC</code>",
            parse_mode="HTML"
        )
        return

    ticker = context.args[0].upper()
    wait_msg = await update.message.reply_text(
        f"🔍 Анализирую <b>{ticker}</b>... ⏳", parse_mode="HTML"
    )

    data = get_market_data(ticker)
    if data is None:
        await wait_msg.edit_text(
            f"❌ Монета <b>{ticker}</b> не найдена в датасете.\n"
            "Используй /tickers чтобы увидеть доступные монеты.",
            parse_mode="HTML"
        )
        return

    # Call LLM
    explanation = call_llm_explanation(
        symbol=data["symbol"],
        rsi=data["rsi"],
        sma_50=data["sma_50"],
        price=data["price"],
        sentiment_label=data["sentiment_label"],
        sentiment_score=data["sentiment_score"],
        news_titles=data["news_titles"],
    )

    # Determine RSI zone emoji
    rsi = data["rsi"]
    if rsi > 70:
        rsi_emoji = "🔴"
    elif rsi < 30:
        rsi_emoji = "🟢"
    else:
        rsi_emoji = "🟡"

    header = (
        f"📊 <b>{data['symbol']}</b> @ <code>${data['price']:.4f}</code>\n"
        f"📅 {data['date']}\n"
        f"{rsi_emoji} RSI: {rsi:.1f} | SMA50: {data['sma_50']:.4f}\n"
        f"{'─' * 30}\n\n"
    )
    footer = (
        f"\n\n<code>⚠️ Образовательный анализ, не инвестиционный совет</code>"
    )

    full_message = header + explanation + footer

    # Telegram limit is 4096 chars
    if len(full_message) > 4096:
        full_message = full_message[:4090] + "…"

    await wait_msg.edit_text(full_message, parse_mode="HTML")


async def signal_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /signal — show latest buy/sell signals from trades.csv
    """
    signals = get_latest_signals(5)

    if signals is None or signals.empty:
        await update.message.reply_text(
            "📭 Сигналов пока нет. Запусти <code>main.py</code> для генерации.",
            parse_mode="HTML"
        )
        return

    lines = ["📡 <b>Последние сигналы системы:</b>\n"]
    for _, row in signals.iterrows():
        decision = str(row.get("decision", "")).upper()
        emoji = "🟢" if decision == "BUY" else "🔴"
        ticker = row.get("ticker", "?")
        price = row.get("price", 0)
        date = row.get("date", "?")
        sentiment = row.get("sentiment", "?")
        rsi = row.get("rsi", "?")

        lines.append(
            f"{emoji} <b>{decision}</b> · {ticker} @ ${float(price):.4f}\n"
            f"   📅 {date} | RSI: {rsi} | Sent: {sentiment}\n"
        )

    lines.append("\n<i>⚠️ Образовательный анализ, не инвестиционный совет</i>")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def tickers_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show available tickers from the dataset."""
    try:
        df = pd.read_csv(FEATURES_CSV)
        tickers = sorted(df["ticker"].unique().tolist())
        text = (
            "💰 <b>Доступные монеты:</b>\n\n"
            + " | ".join(f"<code>{t}</code>" for t in tickers)
            + f"\n\nВсего: {len(tickers)} монет\n"
            "Используй: <code>/analyze TICKER</code>"
        )
        await update.message.reply_text(text, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle plain text messages — treat as ticker if it looks like one,
    otherwise suggest using commands.
    """
    text = update.message.text.strip().upper()

    # If it looks like a ticker (2-6 uppercase letters)
    if text.isalpha() and 2 <= len(text) <= 6:
        # Fake context args and reuse analyze handler
        context.args = [text]
        await analyze_handler(update, context)
        return

    await update.message.reply_text(
        "🤔 Не понял команду.\n\n"
        "💡 Напиши тикер монеты (например <code>BTC</code>) "
        "или используй /help для списка команд.",
        parse_mode="HTML"
    )


# ============================================================
# MAIN
# ============================================================
def main():
    print("🌸 Starting Crypto Trading Telegram Bot · Team 5")
    print(f"   Token: ...{TELEGRAM_BOT_TOKEN[-10:]}")
    print(f"   LLM: {'✅ configured' if OPENROUTER_API_KEY else '⚠️  no API key (set OPENROUTER_API_KEY)'}")
    print(f"   Data: {FEATURES_CSV}")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register commands
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("analyze", analyze_handler))
    app.add_handler(CommandHandler("signal", signal_handler))
    app.add_handler(CommandHandler("tickers", tickers_handler))

    # Handle plain text (e.g. user just types "BTC")
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    print("🚀 Bot is running. Press Ctrl+C to stop.\n")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
