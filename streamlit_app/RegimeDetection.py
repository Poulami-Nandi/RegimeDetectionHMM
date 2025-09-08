import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from src.regime_detection import detect_regimes, BEAR

st.title('Regime Detection')

ticker = st.sidebar.text_input('Ticker', 'SPY')
n_components = st.sidebar.slider('HMM States', 2, 5, 2)

df, _ = detect_regimes(ticker=ticker, n_components=n_components)

st.subheader(f'{ticker} — Price & Regimes')
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df.index, df['Close'], label='Close')
in_bear = df['regime'] == BEAR
ax.fill_between(df.index, df['Close'].min(), df['Close'].max(), where=in_bear, alpha=0.2, step='pre')
ax.legend()
st.pyplot(fig)

st.subheader('Regime Probabilities (last 10 rows)')
st.dataframe(df[['p_bear']].tail(10))
