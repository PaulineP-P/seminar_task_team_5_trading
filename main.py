"""
🌸 CRYPTO TRADING BOT · 4-COMPONENT LINEAR SCORING
======================================================
Uses cached LLM analysis + optimized weights (from optimize.py)
Features: sentiment, RSI, Bollinger Bands, LLM action
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
 
# 💗 CONFIGURATION (optimized by differential evolution)
# ============================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = "google/gemini-2.5-flash-lite"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
 
# # input/output files (simplified names)
FEATURES_CSV = "features.csv"        # renamed from crypto_features_3months.csv
NEWS_CSV = "news.csv"                 # renamed from crypto_news_3months.csv
OUTPUT_CSV = "trades.csv"              # renamed from trades_log.csv
 
# # backtest settings
STARTING_CAPITAL = 10000
TRANSACTION_FEE = 0.001  # 0.1%
 
# # OPTIMIZED WEIGHTS (from optimize.py)
# # these maximize sharpe ratio on historical data
W_SENTIMENT = 17.9413   # llm sentiment dominates
W_RSI = 2.2901          # rsi adds small signal
W_BB = 12.9918          # bollinger bands matter
W_ACTION = 19.0590      # llm action is crucial
BUY_THRESH = 3.5803     # buy if score ≥ this
SELL_THRESH = 4.7507    # sell if score ≤ this
ALLOC_PCT = 0.10        # 10% of capital per buy
 
 
# 📦 MODULE 1: DATA LOADING
# ============================================================
def load_data():
    """load market features + news, merge by date"""
    print("🌸 loading market data...")
    features = pd.read_csv(FEATURES_CSV)
    print(f"   → {len(features)} rows, {len(features.columns)} columns")
 
    print("🌸 loading news data...")
    news = pd.read_csv(NEWS_CSV)
    print(f"   → {len(news)} articles")
 
    # # group news by date for quick lookup
    news_by_date = {}
    for _, row in news.iterrows():
        date = str(row["date"]).strip()
        title = str(row.get("title", ""))
        body = str(row.get("body", ""))
        if date not in news_by_date:
            news_by_date[date] = []
        news_by_date[date].append(f"{title}: {body[:200]}")
 
    def get_news_text(date):
        """get up to 3 news snippets for a given date"""
        articles = news_by_date.get(str(date).strip(), [])
        return "\n\n".join(articles[:3]) if articles else "No significant news today"
 
    # # attach news to market data
    features["news_text"] = features["date"].apply(get_news_text)
    features = features.sort_values(["date", "ticker"]).reset_index(drop=True)
 
    print(f"   📅 date range: {features['date'].min()} → {features['date'].max()}")
    print(f"   💰 tickers: {sorted(features['ticker'].unique())}")
    print(f"   📰 rows with news: {(features['news_text'] != 'No significant news today').sum()} ({features['news_text'].ne('No significant news today').mean()*100:.1f}%)")
 
    return features
 
 
# 🤖 MODULE 2: LLM PROMPT BUILDER
# ============================================================
def build_prompt(row):
    """
    create structured prompt for gemini 2.5 flash
    includes technical indicators + news (if any)
    """
    close = float(row.get("close", 0))
    ma7 = float(row.get("ma7", 0))
    ma20 = float(row.get("ma20", 0)) if pd.notna(row.get("ma20")) else 0
    rsi = float(row.get("rsi", 50))
    macd_hist = float(row.get("macd_hist", 0))
    bb_pos = float(row.get("bb_position", 0.5))
    vol7d = float(row.get("volatility_7d", 0))
    returns = float(row.get("returns", 0))
 
    # # human-readable zones (helps llm understand context)
    rsi_zone = "OVERSOLD" if rsi < 30 else ("OVERBOUGHT" if rsi > 70 else "neutral")
    bb_zone = "NEAR LOWER BAND" if bb_pos < 0.2 else ("NEAR UPPER BAND" if bb_pos > 0.8 else "middle")
    has_news = row["news_text"] != "No significant news today"
 
    # # prompt designed for structured JSON output
    prompt = f"""You are a conservative cryptocurrency risk analyst. Output ONLY valid JSON.
 
Date: {row['date']} | Coin: {row['ticker']}
Price: ${close} | Return: {returns*100:.2f}% | Volatility: {vol7d*100:.2f}%
RSI: {rsi:.1f} [{rsi_zone}] | MACD: {macd_hist:.6f} | BB: {bb_pos:.3f} [{bb_zone}]
MA7: ${ma7:.4f} | MA20: {f'${ma20:.4f}' if ma20 > 0 else 'N/A'}
 
News: {row['news_text'] if has_news else 'None - use technicals only.'}
 
Output JSON: {{"sentiment_score": <-1 to 1>, "market_mood": "<bearish|neutral|bullish>", "trend_strength": <0-1>, "reversal_probability": <0-1>, "risk_level": "<low|medium|high|extreme>", "recommended_action": "<buy|sell|hold>", "confidence": <0-1>, "reasoning": "<brief>"}}"""
 
    return prompt
 
 
# 🌐 MODULE 3: OPENROUTER API CLIENT
# ============================================================
def call_llm(prompt, max_retries=3):
    """
    call gemini 2.5 flash via openrouter
    includes retry logic with exponential backoff
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,           # low temp = consistent
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
                wait = 2 ** attempt
                print(f"   ⚠️ retry {attempt+1}/{max_retries}: {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                # # fallback response if all retries fail
                return {
                    "sentiment_score": 0, 
                    "market_mood": "neutral", 
                    "trend_strength": 0,
                    "reversal_probability": 0, 
                    "risk_level": "medium",
                    "recommended_action": "hold", 
                    "confidence": 0, 
                    "reasoning": "LLM failed"
                }
 
 
# 📐 MODULE 4: TRADING LOGIC (4-COMPONENT LINEAR SCORING)
# ============================================================
def trading_decision(analysis, indicators):
    """
    convert llm output + indicators into buy/sell/hold
    uses 4-component linear model with optimized weights
    """
    sentiment = float(analysis.get("sentiment_score", 0) or 0)
    risk_level = str(analysis.get("risk_level", "medium")).lower()
    llm_action = str(analysis.get("recommended_action", "hold")).lower()
    confidence = float(analysis.get("confidence", 0) or 0)
 
    rsi = float(indicators.get("rsi", 50))
    bb_pos = float(indicators.get("bb_position", 0.5))
 
    # # RISK GATE #1: extreme risk → force sell
    if risk_level == "extreme":
        return "sell", "EXTREME RISK (forced sell)", confidence
 
    # # 4-component linear score
    # # each component normalized to roughly -1..+1
    score = (W_SENTIMENT * sentiment                     # sentiment: -1..+1
             + W_RSI * (50 - rsi) / 50                   # rsi: 0..100 → -1..+1
             + W_BB * (0.5 - bb_pos) / 0.5)              # bb: 0..1 → -1..+1
 
    # # llm action: buy → +weight, sell → -weight, hold → 0
    if llm_action == "buy": 
        score += W_ACTION
    elif llm_action == "sell": 
        score -= W_ACTION
 
    # # apply thresholds
    decision = "hold"
    reason = f"score={score:.1f}"
 
    if score >= BUY_THRESH:
        decision = "buy"
        reason = f"BUY signal: score={score:.1f} (sent={sentiment:.2f}, rsi={rsi:.0f}, bb={bb_pos:.2f}, llm={llm_action})"
 
    if score <= SELL_THRESH:
        decision = "sell"
        reason = f"SELL signal: score={score:.1f} (sent={sentiment:.2f}, rsi={rsi:.0f}, bb={bb_pos:.2f}, llm={llm_action})"
 
    # # EMERGENCY GATE #2: rsi > 75 overrides everything
    if rsi > 75:
        decision = "sell"
        reason = f"EMERGENCY SELL: RSI overbought ({rsi:.1f} > 75)"
 
    return decision, reason, confidence
 
 
# 📊 MODULE 5: BACKTESTING ENGINE
# ============================================================
def run_backtest(trades_df):
    """
    simulate trading with $10k starting capital
    tracks positions, calculates sharpe, drawdown, win rate
    """
    print("\n📈 === BACKTESTING ===")
    capital = STARTING_CAPITAL
    positions = {}          # ticker → {qty, avg_price}
    daily_portfolio_values = []
    trade_results = []      # track each trade for win rate
 
    dates = sorted(trades_df["date"].unique())
    for date in dates:
        day_trades = trades_df[trades_df["date"] == date]
        
        # # execute all trades for this day
        for _, trade in day_trades.iterrows():
            ticker, price, decision = trade["ticker"], float(trade["price"]), trade["decision"]
 
            if decision == "buy" and capital > 0:
                # # allocate fixed % of remaining capital
                alloc = capital * ALLOC_PCT
                fee = alloc * TRANSACTION_FEE
                invest = alloc - fee
                qty = invest / price
                
                if ticker not in positions:
                    positions[ticker] = {"qty": 0, "avg_price": 0}
                
                old = positions[ticker]
                new_qty = old["qty"] + qty
                # # update average cost basis
                if new_qty > 0:
                    positions[ticker] = {
                        "qty": new_qty,
                        "avg_price": (old["qty"] * old["avg_price"] + invest) / new_qty
                    }
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
 
        # # mark portfolio to market at day end
        portfolio_value = capital
        for ticker, pos in positions.items():
            if pos["qty"] > 0:
                ticker_row = day_trades[day_trades["ticker"] == ticker]
                if not ticker_row.empty:
                    portfolio_value += pos["qty"] * float(ticker_row.iloc[0]["price"])
        daily_portfolio_values.append({"date": date, "value": portfolio_value})
 
    # # calculate performance metrics
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
 
        print(f"   ✦ Sharpe Ratio:  {sharpe:.4f}")
        print(f"   ✦ Total Return:  {total_return:+.2f}%")
        print(f"   ✦ Max Drawdown:  {drawdown:.2f}%")
        print(f"   ✦ Win Rate:      {win_rate:.1f}%")
        print(f"   ✦ Total Trades:  {len(trade_results)} ({len(buy_trades)} buys, {len(sell_trades)} sells)")
        print(f"   ✦ Final Value:   ${pv['value'].iloc[-1]:.2f}")
        
        return {
            "sharpe": sharpe, 
            "total_return": total_return, 
            "max_drawdown": drawdown,
            "win_rate": win_rate, 
            "final_value": pv["value"].iloc[-1]
        }
    return None
 
 
# 🧪 MODULE 6: A/B TEST ENGINE
# ============================================================
# Strategy A = current optimized weights (baseline)
# Strategy B = aggressive sentiment + stricter RSI gate
W_SENTIMENT_B = 22.0
W_RSI_B       = 1.5
W_BB_B        = 10.0
W_ACTION_B    = 22.0
BUY_THRESH_B  = 4.0
SELL_THRESH_B = 4.0
 
def trading_decision_b(analysis, indicators):
    """Strategy B — more aggressive LLM sentiment weighting."""
    sentiment  = float(analysis.get("sentiment_score", 0) or 0)
    risk_level = str(analysis.get("risk_level", "medium")).lower()
    llm_action = str(analysis.get("recommended_action", "hold")).lower()
    confidence = float(analysis.get("confidence", 0) or 0)
 
    rsi    = float(indicators.get("rsi", 50))
    bb_pos = float(indicators.get("bb_position", 0.5))
 
    if risk_level == "extreme":
        return "sell", "EXTREME RISK (forced sell)", confidence
 
    score = (W_SENTIMENT_B * sentiment
             + W_RSI_B * (50 - rsi) / 50
             + W_BB_B  * (0.5 - bb_pos) / 0.5)
 
    if llm_action == "buy":
        score += W_ACTION_B
    elif llm_action == "sell":
        score -= W_ACTION_B
 
    decision = "hold"
    reason   = f"score={score:.1f}"
 
    if score >= BUY_THRESH_B:
        decision = "buy"
        reason   = f"B-BUY: score={score:.1f}"
    if score <= -SELL_THRESH_B:
        decision = "sell"
        reason   = f"B-SELL: score={score:.1f}"
    if rsi > 75:
        decision = "sell"
        reason   = f"B-EMERGENCY SELL: RSI={rsi:.1f}"
 
    return decision, reason, confidence
 
 
def run_ab_test(cache_path="llm_cache.json"):
    """
    Run both strategies on cached LLM results.
    Returns DataFrame with columns:
      date, value_a, value_b, winner  (A/B/tie)
    Saves ab_results.csv for the Telegram bot to read.
    """
    if not os.path.exists(cache_path):
        print("⚠️  llm_cache.json not found — run main() first.")
        return None
 
    with open(cache_path) as f:
        cache = json.load(f)
 
    rows_a, rows_b = [], []
    for item in cache:
        analysis   = item["analysis"]
        indicators = {
            "rsi":          float(item["indicators"].get("rsi", 50)),
            "bb_position":  float(item["indicators"].get("bb_position", 0.5)),
        }
        dec_a, _, _ = trading_decision(analysis, indicators)
        dec_b, _, _ = trading_decision_b(analysis, indicators)
 
        base = {"date": item["date"], "ticker": item["ticker"], "price": item["price"]}
        rows_a.append({**base, "decision": dec_a})
        rows_b.append({**base, "decision": dec_b})
 
    def simulate(rows):
        df = pd.DataFrame(rows)
        capital   = STARTING_CAPITAL
        positions = {}
        daily     = []
 
        for date in sorted(df["date"].unique()):
            day = df[df["date"] == date]
            for _, t in day.iterrows():
                ticker, price, dec = t["ticker"], float(t["price"]), t["decision"]
                if dec == "buy" and capital > 0:
                    alloc = capital * ALLOC_PCT
                    qty   = (alloc * (1 - TRANSACTION_FEE)) / price
                    if ticker not in positions:
                        positions[ticker] = {"qty": 0, "avg_price": 0}
                    old = positions[ticker]
                    new_qty = old["qty"] + qty
                    positions[ticker] = {
                        "qty": new_qty,
                        "avg_price": (old["qty"] * old["avg_price"] + alloc * (1 - TRANSACTION_FEE)) / new_qty
                        if new_qty > 0 else price
                    }
                    capital -= alloc
                elif dec == "sell" and ticker in positions and positions[ticker]["qty"] > 0:
                    qty      = positions[ticker]["qty"]
                    proceeds = qty * price * (1 - TRANSACTION_FEE)
                    capital += proceeds
                    positions[ticker] = {"qty": 0, "avg_price": 0}
 
            portfolio = capital + sum(
                pos["qty"] * float(day[day["ticker"] == t]["price"].iloc[0])
                for t, pos in positions.items()
                if pos["qty"] > 0 and t in day["ticker"].values
            )
            daily.append({"date": date, "value": portfolio})
 
        return pd.DataFrame(daily)
 
    sim_a = simulate(rows_a).rename(columns={"value": "value_a"})
    sim_b = simulate(rows_b).rename(columns={"value": "value_b"})
    ab    = sim_a.merge(sim_b, on="date")
 
    ab["winner"] = ab.apply(
        lambda r: "B" if r["value_b"] > r["value_a"] else ("A" if r["value_a"] > r["value_b"] else "tie"),
        axis=1
    )
 
    ab.to_csv("ab_results.csv", index=False)
    print(f"✅ A/B results saved → ab_results.csv  ({len(ab)} days)")
 
    b_wins = (ab["winner"] == "B").sum()
    a_wins = (ab["winner"] == "A").sum()
    print(f"   Strategy A wins: {a_wins} days | Strategy B wins: {b_wins} days")
    print(f"   Final A: ${ab['value_a'].iloc[-1]:.2f} | Final B: ${ab['value_b'].iloc[-1]:.2f}")
    return ab
 
 
# 🚀 MODULE 7: MAIN PIPELINE
# ============================================================
def process_row(idx, row, total):
    """process a single row: build prompt, call llm, make decision"""
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
    
    print("🌸" + "="*70)
    print("🌸 CRYPTO TRADING BOT · 4-COMPONENT LINEAR MODEL")
    print("🌸" + "="*70)
 
    # # Check if cache exists (saves time + api costs)
    if os.path.exists("llm_cache.json"):
        print("\n💾 found llm_cache.json - using cached results (no API calls)")
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
        # # first run: call LLM API for every row
        print("\n🆕 no cache found - calling LLM API (this will take ~30min)...")
        features = load_data()
        total = len(features)
        print(f"\n🚀 processing {total} rows (20 concurrent workers)...")
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
                        print(f"   ⏳ progress: {completed}/{total} ({completed/total*100:.0f}%)")
                except Exception as e:
                    idx = futures[future]
                    row = features.iloc[idx]
                    # # fallback on error
                    results[idx] = {
                        "idx": idx, "date": row["date"], "ticker": row["ticker"],
                        "price": float(row["close"]), "decision": "hold",
                        "reason": f"Error: {e}", "sentiment": "0.000",
                        "rsi": f"{float(row.get('rsi', 0)):.2f}", "confidence": "0.000",
                    }
                    completed += 1
 
        # # Save cache for future runs
        cache_data = []
        for r in results:
            if r:
                cache_data.append({
                    "idx": r.get("idx", 0), "date": r["date"], "ticker": r["ticker"],
                    "price": r["price"], "analysis": r.get("analysis", {}),
                    "indicators": r.get("indicators", {}),
                })
                # # remove internal fields before exporting
                r.pop("idx", None)
                r.pop("analysis", None)
                r.pop("indicators", None)
 
        with open("llm_cache.json", "w") as f:
            json.dump(cache_data, f)
        print(f"\n💾 saved cache to llm_cache.json (next runs will be instant)")
 
    # # Export results
    trades_df = pd.DataFrame(results)
    trades_df.to_csv(OUTPUT_CSV, index=False)
    
    total = len(trades_df)
    buy_c = len(trades_df[trades_df["decision"] == "buy"])
    sell_c = len(trades_df[trades_df["decision"] == "sell"])
    hold_c = len(trades_df[trades_df["decision"] == "hold"])
    
    print(f"\n📊 exported {total} trades to {OUTPUT_CSV}")
    print(f"   💗 buys:  {buy_c} ({buy_c/total*100:.1f}%)")
    print(f"   💔 sells: {sell_c} ({sell_c/total*100:.1f}%)")
    print(f"   🤍 holds: {hold_c} ({hold_c/total*100:.1f}%)")
 
    # # Run backtest
    metrics = run_backtest(trades_df)
    
    # Run A/B test if cache available
    if os.path.exists("llm_cache.json"):
        print("\n🧪 === A/B TEST ===")
        run_ab_test()
 
    elapsed = time.time() - start_time
    print(f"\n✨ completed in {elapsed:.1f}s")
    return metrics
 
 
if __name__ == "__main__":
    main()
 
