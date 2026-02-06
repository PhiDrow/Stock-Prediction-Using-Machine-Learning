import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import plotly.graph_objects as go
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Stock Price Prediction Dashboard")
st.caption("📌 Bank Rakyat Indonesia (BBRI) – LSTM Model")


st.sidebar.header("Parameter Input")

stock_ticker = st.sidebar.text_input("Kode Saham", "BBRI.JK")
start_date = st.sidebar.date_input("Tanggal Mulai", dt.datetime(2015, 1, 1))
end_date = st.sidebar.date_input("Tanggal Akhir", dt.datetime.now())

ma1 = st.sidebar.slider("Moving Average 1", 10, 200, 100)
ma2 = st.sidebar.slider("Moving Average 2", 50, 300, 200)


@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start, end)

with st.spinner("📥 Mengunduh data saham..."):
    df = load_data(stock_ticker, start_date, end_date)

if df.empty:
    st.error("❌ Data saham kosong. Periksa ticker atau koneksi internet.")
    st.stop()


df.index = pd.to_datetime(df.index)

last_price = float(df["Close"].iloc[-1])
return_pct = float((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100)
volatility = float(df["Close"].pct_change().std() * np.sqrt(252) * 100)

col1, col2, col3 = st.columns(3)
col1.metric("Harga Terakhir", f"{last_price:,.0f}")
col2.metric("Return (%)", f"{return_pct:.2f}%")
col3.metric("Volatilitas Tahunan", f"{volatility:.2f}%")

st.divider()

# MOVING AVERAGE (ANTI NaN)
st.subheader("📉 Moving Average Analysis")

st.subheader('Analisis Moving Average (MA)')
ma100 = df.Close.rolling(100).mean() 
ma200 = df.Close.rolling(200).mean() 
fig_ma = plt.figure(figsize=(12, 6)) 
plt.plot(df.Close, 'b', label='Harga Close') 
plt.plot(ma100, 'r', label='MA 100') 
plt.plot(ma200, 'g', label='MA 200') 
plt.legend() 
st.pyplot(fig_ma)

# PREDIKSI LSTM
st.subheader("🤖 Prediksi Harga Saham (LSTM)")

try:
    model = load_model("stock_dl_model.keras")

    data_training = pd.DataFrame(df["Close"][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df["Close"][int(len(df)*0.70):])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_training)

    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing])

    input_data = scaler.transform(final_df)

    x_test, y_test = [], []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    y_pred = model.predict(x_test)

    scale_factor = 1 / scaler.scale_[0]
    y_pred = y_pred * scale_factor
    y_test = y_test * scale_factor

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(y=y_test, name="Harga Aktual"))
    fig_pred.add_trace(go.Scatter(y=y_pred.flatten(), name="Harga Prediksi"))

    fig_pred.update_layout(
        height=450,
        xaxis_title="Waktu",
        yaxis_title="Harga"
    )

    st.plotly_chart(fig_pred, use_container_width=True)

except Exception as e:
    st.warning("⚠️ Model LSTM tidak ditemukan atau gagal dimuat.")
    st.write(e)

# DATA VIEW
with st.expander("📄 Lihat Data Saham"):
    st.dataframe(df.tail(20))




