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





# Regime Detection with Hidden Markov Models (HMM)
This project implements market regime detection using Hidden Markov Models (HMMs) combined with confirmation rules and backtesting logic. The goal is to classify regimes such as bull and bear markets, analyze their properties, and demonstrate how regime awareness can improve forecasting and strategy design.

---

## Project Overview

Financial markets exhibit recurring phases of uptrends (bull runs), downtrends (bear phases), and transitional states. Identifying these regimes in real time allows traders and researchers to adapt strategies dynamically — for example, tightening risk controls in bears while allowing growth exposure in bulls.

This project demonstrates:
- How HMMs can be used to infer latent states of a time series.
- How to map those hidden states to interpretable regimes using composite “bearishness” scores.
- How additional confirmation rules (trend filters, drawdown thresholds, persistence checks) improve reliability.
- How to visualize regimes and integrate them into simple forecasts.
- The framework is flexible and can be applied to major indices or single volatile stocks such as Tesla (TSLA).

---

## Methodology

The pipeline consists of the following steps:
1. **Data Collection & Feature Engineering**
- Historical stock/ETF data is fetched (daily frequency).
- Features include returns, drawdowns, volatility, and momentum gaps (e.g., EMA20 vs EMA100).
2. **Hidden Markov Model Training**
- An HMM is fitted with diagonal covariance matrices.
- Multiple restarts are used for stability.
- States are labeled as bullish or bearish using a composite score based on forward returns, drawdowns, and moving-average conditions.
3. **Probability Smoothing**
- Posterior bear probabilities are smoothed with an exponential moving average (EMA) to reduce noise.
4. **Confirmation Logic**
- Candidate bear/bull states are promoted to confirmed if conditions such as persistent trend weakness (EMA20 < EMA100) or drawdown thresholds are met.
- Adjustable knobs allow experimentation with persistence, thresholds, and minimum run lengths.
5. **Hysteresis Path Construction**
- A hysteresis filter with separate enter/exit thresholds ensures regimes do not flip erratically.
- Optional gates (directional, trend-based, profit-exit) refine the entry/exit rules.
6. **Post-processing**
- Short “islands” of regimes are cleaned unless explicitly protected.
- Optional strict mode discards bear runs that do not exhibit meaningful losses or drawdowns.

---
## Results and Visualizations
The following figures illustrate regime detection and forecasts for Tesla (TSLA) over the last 3 years.

## Regime-aware forecast vs baseline ARIMA
Baseline ARIMA(1,1,1) one-step forecast compared with regime-aware forecast.

---

## Detected regimes (green = bull, red = bear)
Shaded regions show candidate vs confirmed phases.


## Close with EMA overlays
TSLA closing price with EMA20 (fast) and EMA100 (slow).

---

## Streamlit Dashboard
A demonstration dashboard is deployed with Streamlit, focused on TSLA as a volatile test case:
Live demo on Streamlit Cloud
Features include:
- Toggleable parameters (confirmation persistence, thresholds, gates).
- Visual regime overlays and tables of confirmed bull/bear runs.
- Copyable tables for quick inspection of segment statistics.

---

## Installation
```bash
git clone https://github.com/Poulami-Nandi/RegimeDetectionHMM.git
cd RegimeDetectionHMM
pip install -r requirements.txt
```
---
## **Future Work**

Planned extensions include:
- Extending regime-aware forecasting with deep learning and ensemble models.
- Adding reinforcement learning to optimize trading strategies conditioned on regimes.
- Multi-asset applications with cross-regime correlations.

---
## **Author**

[Dr. Poulami Nandi](https://www.linkedin.com/in/poulami-nandi-a8a12917b/)  
<img src="https://github.com/Poulami-Nandi/IV_surface_analyzer/raw/main/images/own/own_image.jpg" alt="Profile" width="150"/>  
Physicist · Quant Researcher · Data Scientist  
[University of Pennsylvania](https://live-sas-physics.pantheon.sas.upenn.edu/people/poulami-nandi) | [IIT Kanpur](https://www.iitk.ac.in/) | [TU Wien](http://www.itp.tuwien.ac.at/CPT/index.htm?date=201838&cats=xbrbknmztwd)

📧 [nandi.poulami91@gmail.com](mailto:nandi.poulami91@gmail.com),    
🔗 [LinkedIn](https://www.linkedin.com/in/poulami-nandi-a8a12917b/) • [GitHub](https://github.com/Poulami-Nandi) • [Google Scholar](https://scholar.google.co.in/citations?user=bOYJeAYAAAAJ&hl=en)  

---

## License
This demo is for learning purposes. Data comes from public, free sources for educational use.
