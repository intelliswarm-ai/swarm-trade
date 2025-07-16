# 🤖 Swarm Trade - AI-Powered Forex Analysis

**Talk to your charts like never before.** Get instant AI-powered insights from your trading screenshots using natural language conversations.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements-lite.txt

# Start the interactive chat mode
python main.py --chat
```

## 💬 Interactive Chat Mode - The Game Changer

Transform how you analyze forex charts with our **AI chat interface**. Simply take a screenshot and start asking questions in plain English.

### Why Chat Mode?

- **Instant Analysis**: Get immediate insights from any chart screenshot
- **Natural Conversation**: Ask questions like you would to a trading mentor
- **Multiple AI Models**: Switch between different AI models for varied perspectives
- **Specialized Commands**: Access specific analysis tools with simple commands

### Chat Commands

```
/screenshot      - Capture your trading screen and analyze instantly
/analyze [path]  - Analyze any chart image file
/signals         - Get AI-powered trading recommendations
/patterns        - Identify chart patterns and formations
/risk           - Get comprehensive risk assessment
/sentiment      - Analyze market sentiment from charts
/summary        - Get complete analysis overview
/quick          - Fast analysis for quick decisions
```

### Example Chat Session

```
💬 You: /screenshot
📸 Taking screenshot...
✅ Screenshot saved: screenshots/trading_screenshot_20250716_203544.png
🤖 Analyzing screenshot...

🤖 LLM Response:
Looking at this EUR/USD chart, I can see a clear bullish engulfing pattern 
forming at the 1.0850 support level. The RSI is showing oversold conditions 
with positive divergence. This suggests a potential reversal...

💬 You: What's the risk if I go long here?
🤖 Based on the current setup, a long position would have:
- Stop loss at 1.0820 (30 pips risk)
- Target at 1.0920 (70 pips reward)
- Risk/Reward ratio: 1:2.33
- Probability of success: 65%...

💬 You: /patterns
🤖 Chart patterns detected:
1. Bullish Engulfing (Confidence: 85%)
2. Double Bottom forming (Confidence: 70%)
3. Ascending Triangle breakout potential (Confidence: 60%)...
```

## 🎯 Other Usage Modes

**Single Analysis** (Traditional):
```bash
python main.py
```

**GUI Mode** (Visual Interface):
```bash
python main.py --gui
```

**Continuous Monitoring** (Automated):
```bash
python main.py --continuous
```

## 📦 Installation Options

### Lightweight (Recommended)
```bash
pip install -r requirements-lite.txt
```

### Full Features
```bash
pip install -r requirements.txt
```

### Windows One-Click
```bash
install_windows.bat
```

## 🔧 Configuration

Set up your environment variables:
```bash
NEWS_API_KEY=your_news_api_key_here
CONFIDENCE_THRESHOLD=0.6
RISK_LEVEL=medium
```

## 🎨 Features

### 🧠 AI-Powered Analysis
- **Multi-model support**: Choose from different AI models
- **Natural language processing**: Ask questions in plain English
- **Real-time insights**: Get instant feedback on chart patterns
- **Risk assessment**: Comprehensive risk/reward analysis

### 📊 Technical Analysis
- Pattern recognition (engulfing, doji, hammer, etc.)
- Support/resistance level identification
- Trend analysis and breakout detection
- Volume analysis and confirmation

### 📰 Market Sentiment
- Real-time news sentiment analysis
- Economic impact assessment
- Central bank sentiment tracking
- Multi-timeframe sentiment correlation

### 🖥️ Platform Support
Works with any trading platform:
- MetaTrader 4/5
- TradingView
- ThinkOrSwim
- Interactive Brokers
- NinjaTrader
- cTrader

## 💡 Pro Tips

1. **Start with /screenshot** - Capture your current chart and get instant analysis
2. **Use /quick** for fast decisions during active trading
3. **Try /patterns** to identify specific chart formations
4. **Ask follow-up questions** - The AI remembers context from your current session
5. **Switch models** with /switch for different perspectives

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading forex and securities carries significant risk and this tool should not be used as the sole basis for trading decisions. Always conduct your own research and consider consulting with a financial advisor.

## 🤝 Contributing

We welcome contributions! Fork the repository, make your changes, and submit a pull request.

## 📄 License

MIT License - see LICENSE file for details.
