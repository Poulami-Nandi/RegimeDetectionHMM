import streamlit as st
import pandas as pd
from pathlib import Path
from src.regime_detection import detect_regimes
from src.utils import ensure_dir

st.title('Stress Testing & Drift Alerts')

ticker = st.sidebar.text_input('Ticker', 'SPY')
shock = st.sidebar.slider('Return shock (daily additive)', -0.1, 0.1, -0.03, 0.005)
vol_scale = st.sidebar.slider('Volatility scale', 0.5, 3.0, 1.2, 0.1)

df, _ = detect_regimes(ticker=ticker, n_components=2)

# Apply shock
import numpy as np
out = df.copy()
out['ret_shocked'] = out['ret'] * vol_scale + shock
out['Close_shocked'] = out['Close'].iloc[0] * np.exp(out['ret_shocked'].cumsum())

st.subheader('Shocked Series Preview')
st.dataframe(out[['Close','Close_shocked']].tail(10))

# Drift via simple histogram shift of returns
base_hist, bins = np.histogram(df['ret'].dropna(), bins=30, density=True)
shock_hist, _ = np.histogram(out['ret_shocked'].dropna(), bins=bins, density=True)
# Jensen-Shannon distance
from scipy.spatial.distance import jensenshannon
js = float(jensenshannon(base_hist + 1e-12, shock_hist + 1e-12))
alert = js > 0.25

st.metric('JS drift', f'{js:.3f}', help='Jensen–Shannon distance between base vs shocked return histograms')
st.write('**Alert:**', '🚨 Drift high' if alert else '✅ Within range')

# Allow CSV download
import io
csv_buf = io.StringIO()
pd.DataFrame({'js_drift':[js],'alert':[alert],'shock':[shock],'vol_scale':[vol_scale]}).to_csv(csv_buf, index=False)
st.download_button('Download stress report CSV', csv_buf.getvalue(), file_name=f'stress_report_{ticker}.csv', mime='text/csv')
