from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import re
import urllib.request
import urllib.error

app = Flask(__name__)
CORS(app)

# OpenRouter - free AI access with much higher limits
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# openrouter/free automatically picks best available free model
# Never gets 404, 200 requests/day free, no credit card needed
AI_MODEL = "openrouter/free"


def get_twelve_candles(symbol, interval="5min", outputsize=100):
    url = "https://api.twelvedata.com/time_series"
    url += "?symbol=" + symbol
    url += "&interval=" + interval
    url += "&outputsize=" + str(outputsize)
    url += "&apikey=" + os.environ.get("TWELVE_API_KEY", "")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    if data.get("status") == "error":
        raise ValueError("Twelve Data error: " + data.get("message", "unknown"))
    values = data.get("values", [])
    candles = []
    for v in reversed(values):
        candles.append({
            "open": float(v["open"]),
            "high": float(v["high"]),
            "low": float(v["low"]),
            "close": float(v["close"])
        })
    return candles


def compute_indicators(candles):
    closes = [c["close"] for c in candles]
    n = len(closes) - 1

    def ema(arr, period):
        k = 2 / (period + 1)
        e = arr[0]
        result = []
        for v in arr:
            e = v * k + e * (1 - k)
            result.append(e)
        return result

    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    e50 = ema(closes, 50)

    # RSI
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    p = 14
    ag = sum(gains[:p]) / p
    al = sum(losses[:p]) / p
    for i in range(p, len(gains)):
        ag = (ag * (p-1) + gains[i]) / p
        al = (al * (p-1) + losses[i]) / p
    rsi = 100 - (100 / (1 + ag / al)) if al != 0 else 100

    # MACD
    e12 = ema(closes, 12)
    e26 = ema(closes, 26)
    macd_line = [e12[i] - e26[i] for i in range(len(closes))]
    signal_line = ema(macd_line, 9)
    macd_hist = macd_line[n] - signal_line[n]

    # BB
    period = 20
    sl = closes[n-period+1:n+1]
    mn = sum(sl) / period
    sd = (sum((x-mn)**2 for x in sl) / period) ** 0.5
    bb_upper = mn + 2*sd
    bb_lower = mn - 2*sd

    # Stochastic
    sl14 = candles[n-13:n+1]
    h14 = max(c["high"] for c in sl14)
    l14 = min(c["low"] for c in sl14)
    stoch = ((closes[n] - l14) / (h14 - l14)) * 100 if h14 != l14 else 50

    # ATR
    trs = []
    for i in range(1, len(candles)):
        tr = max(
            candles[i]["high"] - candles[i]["low"],
            abs(candles[i]["high"] - candles[i-1]["close"]),
            abs(candles[i]["low"] - candles[i-1]["close"])
        )
        trs.append(tr)
    atr = sum(trs[-14:]) / 14

    trend_up = e9[n] > e21[n] and e21[n] > e50[n]
    trend_down = e9[n] < e21[n] and e21[n] < e50[n]
    trend = "STRONG UP" if trend_up else "STRONG DOWN" if trend_down else "WEAK UP" if e9[n] > e21[n] else "WEAK DOWN"

    high20 = max(c["high"] for c in candles[-20:])
    low20 = min(c["low"] for c in candles[-20:])

    last10 = candles[-10:]
    candle_summary = ""
    for i, c in enumerate(last10):
        direction = "UP" if c["close"] > c["open"] else "DOWN"
        candle_summary += str(i+1) + ". " + direction + " C:" + str(round(c["close"], 5)) + "\n"

    return {
        "price": round(closes[n], 5),
        "ema9": round(e9[n], 5),
        "ema21": round(e21[n], 5),
        "ema50": round(e50[n], 5),
        "trend": trend,
        "rsi": round(rsi, 1),
        "stoch": round(stoch, 1),
        "macd_hist": round(macd_hist, 5),
        "bb_upper": round(bb_upper, 5),
        "bb_lower": round(bb_lower, 5),
        "bb_mid": round(mn, 5),
        "atr": round(atr, 5),
        "high20": round(high20, 5),
        "low20": round(low20, 5),
        "candle_summary": candle_summary.strip()
    }


def build_prompt(market_data, pair, pair_type):
    note = "Note: OTC pair - focus on momentum and oscillator signals."
    if pair_type != "otc":
        note = "Note: Real forex pair - trend and momentum signals are reliable."

    prompt = "You are an expert trader analyzing " + pair
    prompt += " (" + ("OTC" if pair_type == "otc" else "Forex") + ")"
    prompt += " for a short-term binary option on Pocket Option.\n\n"
    prompt += "Market Data:\n"
    prompt += "Price: " + str(market_data.get("price")) + "\n"
    prompt += "Trend: " + str(market_data.get("trend")) + "\n"
    prompt += "EMA9: " + str(market_data.get("ema9")) + "\n"
    prompt += "EMA21: " + str(market_data.get("ema21")) + "\n"
    prompt += "EMA50: " + str(market_data.get("ema50")) + "\n"
    prompt += "RSI: " + str(market_data.get("rsi")) + "\n"
    prompt += "Stoch: " + str(market_data.get("stoch")) + "\n"
    prompt += "MACD Hist: " + str(market_data.get("macd_hist")) + "\n"
    prompt += "BB Upper: " + str(market_data.get("bb_upper")) + "\n"
    prompt += "BB Lower: " + str(market_data.get("bb_lower")) + "\n"
    prompt += "BB Mid: " + str(market_data.get("bb_mid")) + "\n"
    prompt += "ATR: " + str(market_data.get("atr")) + "\n"
    prompt += "20-candle High: " + str(market_data.get("high20")) + "\n"
    prompt += "20-candle Low: " + str(market_data.get("low20")) + "\n"
    prompt += "Session: " + str(market_data.get("session", "N/A")) + "\n\n"
    prompt += "Last 10 candles:\n" + str(market_data.get("candle_summary", "")) + "\n\n"
    prompt += note + "\n\n"
    prompt += "Give a trading signal. Reply with ONLY valid JSON matching this exact structure (these are field names and types, NOT example values to copy):\n"
    prompt += '{"direction":"CALL or PUT","confidence":number 50-85,"strength":"STRONG or MODERATE or WEAK","signal_type":"TREND or REVERSAL or MOMENTUM","bull_score":number,"bear_score":number,"summary":"your own 2-3 sentence analysis of THIS specific market data","key_reasons":["specific reason from the data above","another specific reason","third specific reason"],"risk_note":"specific risk for this trade","recommended_expiry":"1 or 3 or 5"}\n\n'
    prompt += "IMPORTANT: Do NOT copy the words 'brief explanation' or 'brief risk' literally. Write your own real analysis based on the actual price, RSI, MACD and trend values given above.\n"
    prompt += "Rules: direction=CALL or PUT, confidence=50-85, strength=STRONG/MODERATE/WEAK, signal_type=TREND/REVERSAL/MOMENTUM, recommended_expiry=1/3/5, no line breaks in strings, output raw JSON only, no markdown"
    return prompt


def extract_json(text):
    if not text or not text.strip():
        raise ValueError("Empty response from AI")
    clean = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        pass
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        candidate = match.group(0)
        candidate = candidate.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)
        return json.loads(candidate)
    raise ValueError("No JSON found in AI response")


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "PO AI Bot Server running! Powered by Gemini AI + Twelve Data"})


@app.route("/analyze", methods=["POST"])
def analyze():
    raw_text = ""
    try:
        data = request.json
        market_data = data.get("market_data", {})
        pair = data.get("pair", "EUR/USD")
        pair_type = data.get("pair_type", "forex")

        # Fetch from Twelve Data if forex and no market data
        if pair_type == "forex" and not market_data.get("price"):
            try:
                raw = get_twelve_candles(pair)
                market_data = compute_indicators(raw)
                market_data["session"] = data.get("session", "")
            except Exception as e:
                return jsonify({"success": False, "error": "Twelve Data error: " + str(e)}), 500

        prompt = build_prompt(market_data, pair, pair_type)

        # Call OpenRouter API
        payload = json.dumps({
            "model": AI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.3
        }).encode()

        req = urllib.request.Request(
            OPENROUTER_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + OPENROUTER_API_KEY,
                "HTTP-Referer": "https://po-ai-bot.onrender.com",
                "X-Title": "PO AI Bot"
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        # Extract text from OpenRouter response
        raw_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not raw_text:
            return jsonify({"success": False, "error": "Empty response from AI", "full": str(result)}), 500

        try:
            signal = extract_json(raw_text)
        except Exception as pe:
            return jsonify({
                "success": False,
                "error": "JSON parse error: " + str(pe),
                "raw": raw_text[:500]
            }), 500

        return jsonify({"success": True, "signal": signal})

    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return jsonify({"success": False, "error": "AI API error " + str(e.code) + ": " + body}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "raw": raw_text[:300]}), 500


@app.route("/telegram", methods=["POST"])
def telegram():
    try:
        data = request.json
        token = data.get("token")
        chat_id = data.get("chat_id")
        text = data.get("text")
        if not all([token, chat_id, text]):
            return jsonify({"ok": False, "error": "Missing token, chat_id or text"}), 400
        url = "https://api.telegram.org/bot" + token + "/sendMessage"
        payload = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return jsonify(json.loads(resp.read()))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
