"""
🎮 INTERACTIVE TRADING MODE · Human-in-the-loop
Запускает Telegram бота для получения команд и ручной торговли
"""

import os
import json
import time
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configuration
TELEGRAM_BOT_TOKEN = "8645012892:AAEv9JLyi3tSPOcyxQitJrBM4mz5Swzi3OQ"
TELEGRAM_CHAT_ID = "123456789"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = "google/gemini-2.5-flash-lite"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# File paths
FEATURES_CSV = "features.csv"
POSITIONS_FILE = "positions.json"
TRADES_LOG = "manual_trades.csv"


class InteractiveTradingBot:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.last_update_id = 0
        self.positions = self.load_positions()
        
    def load_positions(self):
        """Load positions from file"""
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def save_positions(self):
        """Save positions to file"""
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(self.positions, f, indent=2)
    
    def log_trade(self, action, ticker, price, qty, pnl=0):
        """Log manual trade to CSV"""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'ticker': ticker,
            'price': price,
            'quantity': qty,
            'pnl': pnl
        }
        
        df = pd.DataFrame([trade])
        if os.path.exists(TRADES_LOG):
            existing = pd.read_csv(TRADES_LOG)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(TRADES_LOG, index=False)
    
    def send_message(self, text):
        """Send message to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"⚠️ Telegram send error: {e}")
            return False
    
    def get_updates(self):
        """Get new messages from Telegram"""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {
                "offset": self.last_update_id + 1,
                "timeout": 30,
                "allowed_updates": ["message"]
            }
            response = requests.get(url, params=params, timeout=35)
            response.raise_for_status()
            data = response.json()
            
            if data.get("ok") and data.get("result"):
                for update in data["result"]:
                    self.last_update_id = update["update_id"]
                    self.handle_update(update)
                    
        except Exception as e:
            print(f"⚠️ Get updates error: {e}")
    
    def handle_update(self, update):
        """Handle incoming message"""
        try:
            message = update.get("message", {})
            text = message.get("text", "")
            chat_id = message.get("chat", {}).get("id")
            
            # Only respond to our chat
            if str(chat_id) != str(self.chat_id):
                return
            
            # Parse command
            if text.startswith("/"):
                parts = text.split()
                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                if command == "/start":
                    self.cmd_start(chat_id)
                elif command == "/help":
                    self.cmd_help(chat_id)
                elif command == "/status":
                    self.cmd_status(chat_id)
                elif command == "/positions":
                    self.cmd_positions(chat_id)
                elif command == "/price":
                    self.cmd_price(chat_id, args)
                elif command == "/buy":
                    self.cmd_buy(chat_id, args)
                elif command == "/sell":
                    self.cmd_sell(chat_id, args)
                elif command == "/analyze":
                    self.cmd_analyze(chat_id, args)
                else:
                    self.send_message(f"❌ Unknown command: {command}\nTry /help")
        except Exception as e:
            print(f"⚠️ Handle update error: {e}")
    
    def cmd_start(self, chat_id):
        """Handle /start command"""
        welcome = """🌸 <b>Trading Assistant Bot</b> · Human-in-the-loop

Я помогаю анализировать рынок и принимать торговые решения.

<b>Доступные команды:</b>
/help - показать все команды
/status - текущий анализ рынка
/positions - открытые позиции
/price [ticker] - цена монеты
/analyze [ticker] - детальный анализ
/buy [ticker] [amount] - купить (в USDT)
/sell [ticker] [amount] - продать

<i>Пример: /buy BTCUSDT 100</i>"""
        self.send_message(welcome)
    
    def cmd_help(self, chat_id):
        """Handle /help command"""
        help_text = """<b>📚 Все команды:</b>

<b>📊 Информация:</b>
/status - общий анализ рынка
/positions - открытые позиции
/price BTCUSDT - цена монеты
/analyze BTCUSDT - детальный анализ

<b>💰 Торговля:</b>
/buy BTCUSDT 100 - купить на 100 USDT
/sell BTCUSDT 50 - продать 50 монет

<b>⚙️ Другое:</b>
/help - это сообщение
/start - приветствие"""
        self.send_message(help_text)
    
    def cmd_status(self, chat_id):
        """Handle /status command - overall market analysis"""
        self.send_message("⏳ Анализирую рынок...")
        
        try:
            # Load latest data
            features = pd.read_csv(FEATURES_CSV)
            latest_date = features['date'].max()
            latest_data = features[features['date'] == latest_date]
            
            # Create summary
            summary = f"📊 <b>Рыночный анализ</b> · {latest_date}\n\n"
            
            for _, row in latest_data.iterrows():
                ticker = row['ticker']
                price = float(row['close'])
                rsi = float(row.get('rsi', 50))
                change = float(row.get('returns', 0)) * 100
                
                emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
                summary += f"{emoji} <b>{ticker}</b>: ${price:.2f} ({change:+.2f}%) | RSI: {rsi:.1f}\n"
            
            summary += "\n<i>Используйте /analyze TICKER для детального анализа</i>"
            self.send_message(summary)
            
        except Exception as e:
            self.send_message(f"❌ Ошибка анализа: {e}")
    
    def cmd_positions(self, chat_id):
        """Handle /positions command"""
        if not self.positions:
            self.send_message("📭 Нет открытых позиций")
            return
        
        msg = "📊 <b>Открытые позиции:</b>\n\n"
        total_value = 0
        
        for ticker, pos in self.positions.items():
            qty = pos.get('qty', 0)
            avg_price = pos.get('avg_price', 0)
            
            # Get current price
            current_price = self.get_current_price(ticker)
            if current_price:
                value = qty * current_price
                pnl = value - (qty * avg_price)
                pnl_pct = (pnl / (qty * avg_price)) * 100 if qty * avg_price > 0 else 0
                
                emoji = "🟢" if pnl > 0 else "🔴"
                msg += f"{emoji} <b>{ticker}</b>\n"
                msg += f"   Qty: {qty:.4f}\n"
                msg += f"   Avg: ${avg_price:.2f}\n"
                msg += f"   Now: ${current_price:.2f}\n"
                msg += f"   PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%)\n\n"
                
                total_value += value
        
        msg += f"💰 Total Value: ${total_value:.2f}"
        self.send_message(msg)
    
    def cmd_price(self, chat_id, args):
        """Handle /price command"""
        if not args:
            self.send_message("❌ Укажите тикер: /price BTCUSDT")
            return
        
        ticker = args[0].upper()
        price = self.get_current_price(ticker)
        
        if price:
            self.send_message(f"💰 <b>{ticker}</b>: ${price:.4f}")
        else:
            self.send_message(f"❌ Не могу получить цену для {ticker}")
    
    def cmd_analyze(self, chat_id, args):
        """Handle /analyze command - detailed LLM analysis"""
        if not args:
            self.send_message("❌ Укажите тикер: /analyze BTCUSDT")
            return
        
        ticker = args[0].upper()
        self.send_message(f"⏳ Анализирую {ticker}...")
        
        try:
            # Get latest data for this ticker
            features = pd.read_csv(FEATURES_CSV)
            ticker_data = features[features['ticker'] == ticker].iloc[-1]
            
            # Build analysis prompt
            prompt = self.build_analysis_prompt(ticker_data)
            
            # Call LLM
            analysis = self.call_llm(prompt)
            
            # Format response
            response = f"""📊 <b>Детальный анализ {ticker}</b>

{analysis.get('explanation', 'Анализ недоступен')}

<code>RSI: {ticker_data.get('rsi', 'N/A'):.1f}
Sentiment: {analysis.get('sentiment_score', 0):.2f}
Risk: {analysis.get('risk_level', 'medium')}</code>"""
            
            self.send_message(response)
            
        except Exception as e:
            self.send_message(f"❌ Ошибка анализа: {e}")
    
    def cmd_buy(self, chat_id, args):
        """Handle /buy command"""
        if len(args) < 2:
            self.send_message("❌ Использование: /buy BTCUSDT 100")
            return
        
        ticker = args[0].upper()
        try:
            amount_usdt = float(args[1])
        except:
            self.send_message("❌ Сумма должна быть числом")
            return
        
        # Get current price
        price = self.get_current_price(ticker)
        if not price:
            self.send_message(f"❌ Не могу получить цену для {ticker}")
            return
        
        # Calculate quantity
        qty = amount_usdt / price
        
        # Update positions
        if ticker not in self.positions:
            self.positions[ticker] = {"qty": 0, "avg_price": 0}
        
        old = self.positions[ticker]
        new_qty = old["qty"] + qty
        new_avg = (old["qty"] * old["avg_price"] + amount_usdt) / new_qty if new_qty > 0 else price
        
        self.positions[ticker] = {"qty": new_qty, "avg_price": new_avg}
        self.save_positions()
        
        # Log trade
        self.log_trade('buy', ticker, price, qty)
        
        # Confirm
        self.send_message(
            f"✅ <b>Куплено {ticker}</b>\n"
            f"💰 Цена: ${price:.4f}\n"
            f"📦 Количество: {qty:.4f}\n"
            f"💵 Сумма: ${amount_usdt:.2f}"
        )
    
    def cmd_sell(self, chat_id, args):
        """Handle /sell command"""
        if len(args) < 2:
            self.send_message("❌ Использование: /sell BTCUSDT 0.1")
            return
        
        ticker = args[0].upper()
        try:
            qty_to_sell = float(args[1])
        except:
            self.send_message("❌ Количество должно быть числом")
            return
        
        # Check if we have position
        if ticker not in self.positions or self.positions[ticker]["qty"] < qty_to_sell:
            self.send_message(f"❌ Недостаточно {ticker} для продажи")
            return
        
        # Get current price
        price = self.get_current_price(ticker)
        if not price:
            self.send_message(f"❌ Не могу получить цену для {ticker}")
            return
        
        # Calculate PnL
        old = self.positions[ticker]
        proceeds = qty_to_sell * price
        cost_basis = qty_to_sell * old["avg_price"]
        pnl = proceeds - cost_basis
        
        # Update position
        new_qty = old["qty"] - qty_to_sell
        if new_qty > 0:
            self.positions[ticker]["qty"] = new_qty
        else:
            del self.positions[ticker]
        
        self.save_positions()
        
        # Log trade
        self.log_trade('sell', ticker, price, qty_to_sell, pnl)
        
        # Confirm
        emoji = "🟢" if pnl > 0 else "🔴"
        self.send_message(
            f"{emoji} <b>Продано {ticker}</b>\n"
            f"💰 Цена: ${price:.4f}\n"
            f"📦 Количество: {qty_to_sell:.4f}\n"
            f"💵 Выручка: ${proceeds:.2f}\n"
            f"📊 PnL: ${pnl:+.2f}"
        )
    
    def get_current_price(self, ticker):
        """Get current price for ticker"""
        try:
            # For demo, get from features.csv
            features = pd.read_csv(FEATURES_CSV)
            latest = features[features['ticker'] == ticker].iloc[-1]
            return float(latest['close'])
        except:
            return None
    
    def build_analysis_prompt(self, row):
        """Build prompt for LLM analysis"""
        close = float(row.get("close", 0))
        rsi = float(row.get("rsi", 50))
        bb_pos = float(row.get("bb_position", 0.5))
        
        prompt = f"""You are a crypto trading assistant. Analyze this market data:

Ticker: {row['ticker']}
Date: {row['date']}
Price: ${close}
RSI: {rsi:.1f}
Bollinger Position: {bb_pos:.2f}

Provide a brief analysis including:
1. Current market condition
2. Key levels to watch
3. Risk assessment

Keep it under 150 words. Be educational."""
        
        return prompt
    
    def call_llm(self, prompt):
        """Call LLM for analysis"""
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "google/gemini-2.5-flash-lite",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 300,
            }
            
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            explanation = data["choices"][0]["message"]["content"]
            
            return {
                "explanation": explanation,
                "sentiment_score": 0,
                "risk_level": "medium"
            }
            
        except Exception as e:
            print(f"LLM call error: {e}")
            return {
                "explanation": f"Анализ временно недоступен. Текущие метрики: RSI={row.get('rsi', 'N/A')}",
                "sentiment_score": 0,
                "risk_level": "medium"
            }
    
    def run(self):
        """Main loop"""
        print("🌸 Interactive Trading Bot started")
        print(f"🤖 Chat ID: {self.chat_id}")
        print("📡 Listening for commands...\n")
        
        # Send startup message
        self.send_message(
            "🤖 <b>Interactive Trading Bot запущен</b>\n"
            "Используйте /help для списка команд"
        )
        
        while True:
            try:
                self.get_updates()
                time.sleep(1)  # Don't spam API
            except KeyboardInterrupt:
                print("\n👋 Bot stopped")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                time.sleep(5)


def main():
    bot = InteractiveTradingBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    bot.run()


if __name__ == "__main__":
    main()
