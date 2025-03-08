# 📊 Regime Detection with Hidden Markov Model (HMM)

This project implements **Market Regime Detection** using **Hidden Markov Models (HMMs)** to classify different market states and optimize trading strategies. It applies **backtesting techniques** on major stock indices to analyze performance.

---

## 🚀 **Project Overview**
Financial markets exhibit periods of uptrends (bull), downtrends (bear), and sideways movements. Detecting these market regimes allows traders to **adjust strategies dynamically**.

### **Key Features:**
✅ **Uses Hidden Markov Models (HMMs) to classify market regimes** (Bull, Bear, Sideways).  
✅ **Applies leverage & stop-loss strategies** based on market conditions.  
✅ **Backtests trading strategies across multiple indices.**  
✅ **Computes Sharpe Ratio, Drawdowns, and P&L analysis.**  
✅ **Integrates Alpaca API for live trading (pending fix for Paper Trading).**  

---

## 🔧 **Installation**
```bash
git clone https://github.com/Poulami-Nandi/RegimeDetectionHMM.git
cd RegimeDetectionHMM
pip install -r requirements.txt
📊 Methodology & Implementation
This project follows these key steps:

1️⃣ Data Collection & Preprocessing
Fetch historical price data for major indices (S&P 500, NASDAQ, Dow Jones).
Compute daily returns and log returns.
Handle missing values and normalize data.
2️⃣ Hidden Markov Model (HMM) for Regime Detection
Train HMM to classify Bullish, Bearish, and Sideways market regimes.
Tune model hyperparameters for better classification.
Visualize market state transitions.
3️⃣ Backtesting & Trading Strategy
Implement a risk-managed trading strategy using detected market regimes.
Apply stop-loss and leverage adjustments based on HMM classification.
4️⃣ Performance Evaluation
Compute Sharpe Ratio, Maximum Drawdown, and Cumulative Returns.
Compare strategy performance across multiple indices.
5️⃣ Live Trading with Alpaca API
Integrate with Alpaca API for automated execution (pending fix for Paper Trading).
Deploy strategy in a simulated trading environment.
📈 Results & Performance Metrics
1️⃣ Market Regimes for S&P 500 ETF

2️⃣ Regime Distribution - SP500

3️⃣ Daily Returns for SP500

4️⃣ Cumulative Returns Comparison
SP500

QTUM

ROBO

5️⃣ Market Regime for QTUM

📊 Pending Tasks
🔹 Optimize Position Sizing Based on Market Regime Confidence.
🔹 Develop Interactive Dashboards with Streamlit for Visual Reports.
🔹 Fix Alpaca API Issue for Paper Trading Integration.

🎯 Future Enhancements
1️⃣ Enhance Live Trading with Alpaca & Interactive Brokers.
2️⃣ Implement Reinforcement Learning for Strategy Optimization.
3️⃣ Add Feature Engineering for Improved Regime Classification.

📬 Contact & Contributions
💡 Found this useful? Star this repo and contribute!
📧 Contact: nandi.poulami91@gmail.com
📌 LinkedIn: Poulami Nandi
