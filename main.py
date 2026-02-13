"""
Crypto Trading Bot - Simplified 4-Component Scoring Strategy
Uses cached LLM analysis with optimized linear weights.
Model: 4 features (sentiment, RSI, BB, LLM action), no MACD/volatility/reversal.
"""

import os
import json
import math
import time
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = "google/gemini-2.5-flash-lite"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

FEATURES_CSV = "crypto_features_3months.csv"
NEWS_CSV = "crypto_news_3months.csv"
OUTPUT_CSV = "trades_log.csv"

STARTING_CAPITAL = 10000
TRANSACTION_FEE = 0.001  # 0.1%

# Optimized weights (scipy.optimize.differential_evolution, 4-component model)
W_SENTIMENT = 17.9413
W_RSI = 2.2901
W_BB = 12.9918
W_ACTION = 19.0590
BUY_THRESH = 3.5803
SELL_THRESH = 4.7507
ALLOC_PCT = 0.10


# ============================================================
# MODULE 1: DATA LOADING
# ============================================================

def load_data():
    print("Loading market data...")
    features = pd.read_csv(FEATURES_CSV)
    print(f"  Market data: {len(features)} rows, {len(features.columns)} columns")

    print("Loading news data...")
    news = pd.read_csv(NEWS_CSV)
    print(f"  News data: {len(news)} rows")

    news_by_date = {}
    for _, row in news.iterrows():
        date = str(row["date"]).strip()
        title = str(row.get("title", ""))
        body = str(row.get("body", ""))
        if date not in news_by_date:
            news_by_date[date] = []
        news_by_date[date].append(f"{title}: {body[:200]}")

    def get_news_text(date):
        articles = news_by_date.get(str(date).strip(), [])
        return "\n\n".join(articles[:3]) if articles else "No significant news today"

    features["news_text"] = features["date"].apply(get_news_text)
    features = features.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"  Date range: {features['date'].min()} to {features['date'].max()}")
    print(f"  Tickers: {sorted(features['ticker'].unique())}")
    print(f"  Rows with news: {(features['news_text'] != 'No significant news today').sum()}")

    return features


# ============================================================
# MODULE 2: LLM PROMPT BUILDER
# ============================================================

def build_prompt(row):
    close = float(row.get("close", 0))
    ma7 = float(row.get("ma7", 0))
    ma20 = float(row.get("ma20", 0)) if pd.notna(row.get("ma20")) else 0
    rsi = float(row.get("rsi", 50))
    macd_hist = float(row.get("macd_hist", 0))
    bb_pos = float(row.get("bb_position", 0.5))
    vol7d = float(row.get("volatility_7d", 0))
    returns = float(row.get("returns", 0))

    rsi_zone = "OVERSOLD" if rsi < 30 else ("OVERBOUGHT" if rsi > 70 else "neutral")
    bb_zone = "NEAR LOWER BAND" if bb_pos < 0.2 else ("NEAR UPPER BAND" if bb_pos > 0.8 else "middle")
    has_news = row["news_text"] != "No significant news today"

    prompt = f"""You are a conservative cryptocurrency risk analyst. Output ONLY valid JSON.

Date: {row['date']} | Coin: {row['ticker']}
Price: ${close} | Return: {returns*100:.2f}% | Volatility: {vol7d*100:.2f}%
RSI: {rsi:.1f} [{rsi_zone}] | MACD: {macd_hist:.6f} | BB: {bb_pos:.3f} [{bb_zone}]
MA7: ${ma7:.4f} | MA20: {f'${ma20:.4f}' if ma20 > 0 else 'N/A'}

News: {row['news_text'] if has_news else 'None - use technicals only.'}

Output JSON: {{"sentiment_score": <-1 to 1>, "market_mood": "<bearish|neutral|bullish>", "trend_strength": <0-1>, "reversal_probability": <0-1>, "risk_level": "<low|medium|high|extreme>", "recommended_action": "<buy|sell|hold>", "confidence": <0-1>, "reasoning": "<brief>"}}"""

    return prompt


# ============================================================
# MODULE 3: OPENROUTER API CLIENT
# ============================================================

def call_llm(prompt, max_retries=3):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return json.loads(data["choices"][0]["message"]["content"])
        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt+1}/{max_retries}: {e} (waiting {2**attempt}s)")
                time.sleep(2 ** attempt)
            else:
                return {"sentiment_score": 0, "market_mood": "neutral", "trend_strength": 0,
                        "reversal_probability": 0, "risk_level": "medium",
                        "recommended_action": "hold", "confidence": 0, "reasoning": "LLM failed"}


# ============================================================
# MODULE 4: TRADING LOGIC (4-Component Linear Scoring)
# ============================================================

def trading_decision(analysis, indicators):
    sentiment = float(analysis.get("sentiment_score", 0) or 0)
    risk_level = str(analysis.get("risk_level", "medium")).lower()
    llm_action = str(analysis.get("recommended_action", "hold")).lower()
    confidence = float(analysis.get("confidence", 0) or 0)

    rsi = float(indicators.get("rsi", 50))
    bb_pos = float(indicators.get("bb_position", 0.5))

    # Risk gate
    if risk_level == "extreme":
        return "sell", "EXTREME RISK", confidence

    # 4-component linear score
    score = (W_SENTIMENT * sentiment
             + W_RSI * (50 - rsi) / 50
             + W_BB * (0.5 - bb_pos) / 0.5)

    if llm_action == "buy": score += W_ACTION
    elif llm_action == "sell": score -= W_ACTION

    # Decision
    decision = "hold"
    reason = f"score={score:.1f}"

    if score >= BUY_THRESH:
        decision = "buy"
        reason = f"BUY: score={score:.1f} (sent={sentiment:.2f}, rsi={rsi:.0f}, bb={bb_pos:.2f}, llm={llm_action})"

    if score <= SELL_THRESH:
        decision = "sell"
        reason = f"SELL: score={score:.1f} (sent={sentiment:.2f}, rsi={rsi:.0f}, bb={bb_pos:.2f}, llm={llm_action})"

    if rsi > 75:
        decision = "sell"
        reason = f"EMERGENCY: RSI={rsi:.1f}"

    return decision, reason, confidence


# ============================================================
# MODULE 5: BACKTESTING ENGINE
# ============================================================

def run_backtest(trades_df):
    print("\n=== BACKTESTING ===")
    capital = STARTING_CAPITAL
    positions = {}
    daily_portfolio_values = []
    trade_results = []

    dates = sorted(trades_df["date"].unique())
    for date in dates:
        day_trades = trades_df[trades_df["date"] == date]
        for _, trade in day_trades.iterrows():
            ticker, price, decision = trade["ticker"], float(trade["price"]), trade["decision"]

            if decision == "buy" and capital > 0:
                alloc = capital * ALLOC_PCT
                fee = alloc * TRANSACTION_FEE
                invest = alloc - fee
                qty = invest / price
                if ticker not in positions:
                    positions[ticker] = {"qty": 0, "avg_price": 0}
                old_qty = positions[ticker]["qty"]
                old_cost = old_qty * positions[ticker]["avg_price"]
                new_qty = old_qty + qty
                positions[ticker]["qty"] = new_qty
                positions[ticker]["avg_price"] = (old_cost + invest) / new_qty if new_qty > 0 else 0
                capital -= alloc
                trade_results.append({"action": "buy", "pnl": 0})

            elif decision == "sell" and ticker in positions and positions[ticker]["qty"] > 0:
                qty = positions[ticker]["qty"]
                proceeds = qty * price
                fee = proceeds * TRANSACTION_FEE
                net = proceeds - fee
                pnl = net - (qty * positions[ticker]["avg_price"])
                capital += net
                trade_results.append({"action": "sell", "pnl": pnl})
                positions[ticker] = {"qty": 0, "avg_price": 0}

        portfolio_value = capital
        for ticker, pos in positions.items():
            if pos["qty"] > 0:
                ticker_row = day_trades[day_trades["ticker"] == ticker]
                if not ticker_row.empty:
                    portfolio_value += pos["qty"] * float(ticker_row.iloc[0]["price"])
        daily_portfolio_values.append({"date": date, "value": portfolio_value})

    pv = pd.DataFrame(daily_portfolio_values)
    if len(pv) > 1:
        pv["daily_return"] = pv["value"].pct_change().dropna()
        pv = pv.dropna()
        avg_return = pv["daily_return"].mean()
        std_return = pv["daily_return"].std()
        sharpe = (avg_return / std_return) * math.sqrt(365) if std_return > 0 else 0
        total_return = (pv["value"].iloc[-1] / STARTING_CAPITAL - 1) * 100
        max_val = pv["value"].cummax()
        drawdown = ((pv["value"] - max_val) / max_val).min() * 100
        sell_trades = [t for t in trade_results if t["action"] == "sell"]
        wins = len([t for t in sell_trades if t.get("pnl", 0) > 0])
        win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0
        buy_trades = [t for t in trade_results if t["action"] == "buy"]

        print(f"  Sharpe Ratio:  {sharpe:.4f}")
        print(f"  Total Return:  {total_return:+.2f}%")
        print(f"  Max Drawdown:  {drawdown:.2f}%")
        print(f"  Win Rate:      {win_rate:.1f}%")
        print(f"  Total Trades:  {len(trade_results)} ({len(buy_trades)} buys, {len(sell_trades)} sells)")
        print(f"  Final Value:   ${pv['value'].iloc[-1]:.2f}")
        return {"sharpe": sharpe, "total_return": total_return, "max_drawdown": drawdown,
                "win_rate": win_rate, "final_value": pv["value"].iloc[-1]}
    return None


# ============================================================
# MODULE 6: MAIN PIPELINE
# ============================================================

def process_row(idx, row, total):
    prompt = build_prompt(row)
    analysis = call_llm(prompt)
    indicators = {"rsi": row.get("rsi", 50), "bb_position": row.get("bb_position", 0.5)}
    decision, reason, confidence = trading_decision(analysis, indicators)

    return {
        "idx": idx,
        "date": row["date"],
        "ticker": row["ticker"],
        "price": float(row["close"]),
        "decision": decision,
        "reason": reason,
        "sentiment": f"{float(analysis.get('sentiment_score', 0)):.3f}",
        "rsi": f"{float(row.get('rsi', 0)):.2f}",
        "confidence": f"{confidence:.3f}",
        "analysis": analysis,
        "indicators": indicators,
    }


def main():
    start_time = time.time()

    # Check if cache exists
    if os.path.exists("llm_cache.json"):
        print("Found llm_cache.json - using cached LLM results (no API calls)")
        with open("llm_cache.json") as f:
            cache = json.load(f)

        results = []
        for item in cache:
            analysis = item["analysis"]
            indicators = item["indicators"]
            indicators["rsi"] = float(indicators.get("rsi", 50))
            indicators["bb_position"] = float(indicators.get("bb_position", 0.5))
            decision, reason, confidence = trading_decision(analysis, indicators)
            results.append({
                "date": item["date"],
                "ticker": item["ticker"],
                "price": item["price"],
                "decision": decision,
                "reason": reason,
                "sentiment": f"{float(analysis.get('sentiment_score', 0)):.3f}",
                "rsi": f"{indicators['rsi']:.2f}",
                "confidence": f"{confidence:.3f}",
            })
    else:
        print("No cache found - calling LLM API...")
        features = load_data()
        total = len(features)
        print(f"\nProcessing {total} rows (20 concurrent workers)...")
        print("=" * 60)

        results = [None] * total
        completed = 0

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {}
            for idx, row in features.iterrows():
                future = executor.submit(process_row, idx, row, total)
                futures[future] = idx

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results[result["idx"]] = result
                    completed += 1
                    if completed % 20 == 0 or completed == total:
                        print(f"  Progress: {completed}/{total} ({completed/total*100:.0f}%)")
                except Exception as e:
                    idx = futures[future]
                    row = features.iloc[idx]
                    results[idx] = {
                        "idx": idx, "date": row["date"], "ticker": row["ticker"],
                        "price": float(row["close"]), "decision": "hold",
                        "reason": f"Error: {e}", "sentiment": "0.000",
                        "rsi": f"{float(row.get('rsi', 0)):.2f}", "confidence": "0.000",
                    }
                    completed += 1

        # Save cache
        cache_data = []
        for r in results:
            if r:
                cache_data.append({
                    "idx": r.get("idx", 0), "date": r["date"], "ticker": r["ticker"],
                    "price": r["price"], "analysis": r.get("analysis", {}),
                    "indicators": r.get("indicators", {}),
                })
                r.pop("idx", None)
                r.pop("analysis", None)
                r.pop("indicators", None)

        with open("llm_cache.json", "w") as f:
            json.dump(cache_data, f)
        print(f"\nSaved cache to llm_cache.json")

    # Export
    trades_df = pd.DataFrame(results)
    trades_df.to_csv(OUTPUT_CSV, index=False)
    total = len(trades_df)
    buy_c = len(trades_df[trades_df["decision"] == "buy"])
    sell_c = len(trades_df[trades_df["decision"] == "sell"])
    hold_c = len(trades_df[trades_df["decision"] == "hold"])
    print(f"\nExported {total} trades to {OUTPUT_CSV}")
    print(f"  Buy:  {buy_c} ({buy_c/total*100:.1f}%)")
    print(f"  Sell: {sell_c} ({sell_c/total*100:.1f}%)")
    print(f"  Hold: {hold_c} ({hold_c/total*100:.1f}%)")

    metrics = run_backtest(trades_df)
    print(f"\nCompleted in {time.time()-start_time:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
