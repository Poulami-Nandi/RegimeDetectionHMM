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
```
---

## 📊 Methodology & Implementation
This project follows these key steps:

- 1️⃣ Data Collection & Preprocessing
Fetch historical price data for major indices (S&P 500, NASDAQ, Dow Jones).
Compute daily returns and log returns.
Handle missing values and normalize data.
- 2️⃣ Hidden Markov Model (HMM) for Regime Detection
Train HMM to classify Bullish, Bearish, and Sideways market regimes.
Tune model hyperparameters for better classification.
Visualize market state transitions.
- 3️⃣ Backtesting & Trading Strategy
Implement a risk-managed trading strategy using detected market regimes.
Apply stop-loss and leverage adjustments based on HMM classification.
- 4️⃣ Performance Evaluation
Compute Sharpe Ratio, Maximum Drawdown, and Cumulative Returns.
Compare strategy performance across multiple indices.
- 5️⃣ Paper Trading with Alpaca API
Integrate with Alpaca API for automated execution (pending fix for Paper Trading).  
Deploy strategy in a simulated trading environment.
---

## 📈 Results & Performance Metrics
- 1️⃣ **Market Regimes for S&P 500 ETF**
  ![Market Regimes for S&P 500 ETF](https://github.com/Poulami-Nandi/RegimeDetectionHMM/blob/main/images/results/sample/market_regime_sp500.png)  
- 2️⃣ **Regime Distribution - SP500**
  ![Regime Distribution - SP500 ETF](https://github.com/Poulami-Nandi/RegimeDetectionHMM/blob/main/images/results/sample/dist_market_regime_sp500.png)
- 3️⃣ **Daily Returns for SP500**
  ![Daily Returns for SP500 ETF](https://github.com/Poulami-Nandi/RegimeDetectionHMM/blob/main/images/results/sample/dist_returns_sp500.png)  
- 4️⃣ Cumulative Returns Comparison  
SP500
  ![Cumulative Returns for SP500 ETF](https://github.com/Poulami-Nandi/RegimeDetectionHMM/blob/main/images/results/sample/cumulative_return_sp500.png)  
QTUM
  ![Cumulative Returns for QTUM ETF](https://github.com/Poulami-Nandi/RegimeDetectionHMM/blob/main/images/results/sample/cumulative_return_qtum.png)  
ROBO
  ![Cumulative Returns for ROBO ETF](https://github.com/Poulami-Nandi/RegimeDetectionHMM/blob/main/images/results/sample/cumulative_return_robo.png)  

- 5️⃣ **Market Regime for QTUM**  
  ![Market Regime for QTUM ETF](https://github.com/Poulami-Nandi/RegimeDetectionHMM/blob/main/images/results/sample/market_regime_qtum.png)  

---
## Streamlit

Prepared streamlit dashborad for highly volatile stock Tesla(last 3y)
https://regimedetectionhmm-mkn4ypczlw7vojrqr95bwp.streamlit.app

---

## 🎯 Future Enhancements
- 1️⃣ Enhance Live Trading with Alpaca & Interactive Brokers.
- 2️⃣ Implement Reinforcement Learning for Strategy Optimization.
- 3️⃣ Add Feature Engineering for Improved Regime Classification.

---

## **📬 Contact & Contributions**
💡 Found this useful? Feel free to ⭐ star this repo and contribute!  
**Author**: [Dr. Poulami Nandi](https://www.linkedin.com/in/poulami-nandi-a8a12917b/)  
<img src="https://github.com/Poulami-Nandi/IV_surface_analyzer/raw/main/images/own/own_image.jpg" alt="Profile" width="150"/>  
Physicist · Quant Researcher · Data Scientist  
[University of Pennsylvania](https://live-sas-physics.pantheon.sas.upenn.edu/people/poulami-nandi) | [IIT Kanpur](https://www.iitk.ac.in/) | [TU Wien](http://www.itp.tuwien.ac.at/CPT/index.htm?date=201838&cats=xbrbknmztwd)

📧 [nandi.poulami91@gmail.com](mailto:nandi.poulami91@gmail.com), [pnandi@sas.upenn.edu](mailto:pnandi@sas.upenn.edu)  
🔗 [LinkedIn](https://www.linkedin.com/in/poulami-nandi-a8a12917b/) • [GitHub](https://github.com/Poulami-Nandi) • [Google Scholar](https://scholar.google.co.in/citations?user=bOYJeAYAAAAJ&hl=en)  

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
