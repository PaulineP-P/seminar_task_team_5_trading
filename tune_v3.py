"""
V3 Tuning - simplified 4-component linear model.
Fewer features, simpler logic than V2. Optimized via differential_evolution.
"""

import json
import math
from scipy.optimize import differential_evolution

STARTING_CAPITAL = 10000
TRANSACTION_FEE = 0.001

with open("llm_cache.json") as f:
    cache = json.load(f)

features = []
for item in cache:
    a = item["analysis"]
    ind = item["indicators"]
    features.append({
        "date": item["date"],
        "ticker": item["ticker"],
        "price": item["price"],
        "sentiment": float(a.get("sentiment_score", 0) or 0),
        "risk_level": str(a.get("risk_level", "medium")).lower(),
        "llm_action": str(a.get("recommended_action", "hold")).lower(),
        "rsi": float(ind.get("rsi", 50)),
        "bb_pos": float(ind.get("bb_position", 0.5)),
    })


def evaluate(params_vec):
    """
    4-component linear model (7 params total):
    [0] w_sentiment, [1] w_rsi, [2] w_bb, [3] w_action,
    [4] buy_thresh, [5] sell_thresh, [6] alloc_pct
    """
    w_sent, w_rsi, w_bb, w_action, buy_thresh, sell_thresh, alloc_pct = params_vec

    decisions = []
    for f in features:
        if f["risk_level"] == "extreme":
            decisions.append({"date": f["date"], "ticker": f["ticker"], "price": f["price"], "decision": "sell"})
            continue

        score = (w_sent * f["sentiment"]
                 + w_rsi * (50 - f["rsi"]) / 50
                 + w_bb * (0.5 - f["bb_pos"]) / 0.5)

        if f["llm_action"] == "buy": score += w_action
        elif f["llm_action"] == "sell": score -= w_action

        dec = "hold"
        if score >= buy_thresh: dec = "buy"
        if score <= sell_thresh: dec = "sell"
        if f["rsi"] > 75: dec = "sell"

        decisions.append({"date": f["date"], "ticker": f["ticker"], "price": f["price"], "decision": dec})

    capital = STARTING_CAPITAL
    positions = {}
    daily_values = []
    buys = sells = 0

    dates = sorted(set(d["date"] for d in decisions))
    for date in dates:
        day_rows = [d for d in decisions if d["date"] == date]
        for r in day_rows:
            tk, price, dec = r["ticker"], r["price"], r["decision"]
            if dec == "buy" and capital > 100:
                alloc = capital * alloc_pct
                fee = alloc * TRANSACTION_FEE
                invest = alloc - fee
                qty = invest / price
                if tk not in positions: positions[tk] = {"qty": 0, "avg_price": 0}
                old = positions[tk]
                new_qty = old["qty"] + qty
                positions[tk] = {"qty": new_qty, "avg_price": (old["qty"] * old["avg_price"] + invest) / new_qty if new_qty > 0 else 0}
                capital -= alloc
                buys += 1
            elif dec == "sell" and tk in positions and positions[tk]["qty"] > 0:
                qty = positions[tk]["qty"]
                capital += qty * price * (1 - TRANSACTION_FEE)
                positions[tk] = {"qty": 0, "avg_price": 0}
                sells += 1

        pv = capital
        pt = {r["ticker"]: r["price"] for r in day_rows}
        for tk, pos in positions.items():
            if pos["qty"] > 0 and tk in pt: pv += pos["qty"] * pt[tk]
        daily_values.append(pv)

    if buys < 5 or sells < 3 or len(daily_values) < 2:
        return 0.0

    returns = [(daily_values[i] / daily_values[i-1] - 1) for i in range(1, len(daily_values))]
    avg_ret = sum(returns) / len(returns)
    std_ret = (sum((r - avg_ret)**2 for r in returns) / len(returns)) ** 0.5
    sharpe = (avg_ret / std_ret) * math.sqrt(365) if std_ret > 0 else 0

    max_val = daily_values[0]
    max_dd = 0
    for v in daily_values:
        max_val = max(max_val, v)
        dd = (v - max_val) / max_val
        max_dd = min(max_dd, dd)
    if max_dd < -0.10: sharpe *= 0.5

    return -sharpe


print("V3 Optimization - simplified 4-component model")
print("=" * 70)

bounds = [
    (5, 50),      # w_sentiment
    (1, 30),      # w_rsi
    (1, 25),      # w_bb
    (2, 25),      # w_action
    (3, 40),      # buy_thresh
    (-30, 5),     # sell_thresh
    (0.03, 0.20), # alloc_pct
]

best_so_far = [0]
gen = [0]
def callback(xk, convergence):
    gen[0] += 1
    val = evaluate(xk)
    if -val > best_so_far[0]: best_so_far[0] = -val
    if gen[0] % 10 == 0:
        print(f"  Gen {gen[0]}: Sharpe = {best_so_far[0]:+.4f}")

result = differential_evolution(
    evaluate, bounds, seed=123, maxiter=300, popsize=30,
    tol=1e-6, mutation=(0.5, 1.5), recombination=0.8, callback=callback,
)

print(f"\n{'='*70}")
print(f"Best Sharpe: {-result.fun:+.4f}")
names = ["w_sentiment", "w_rsi", "w_bb", "w_action", "buy_thresh", "sell_thresh", "alloc_pct"]
print("Optimal parameters:")
for n, v in zip(names, result.x):
    print(f"  {n:16s} = {v:.4f}")
