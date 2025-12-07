# app.py — Crypto Coin Prediction (robust MultiIndex handling)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# -------------- Config --------------
APP_TITLE = "Crypto Coin And Future Forecast"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.markdown(
    "Per-coin pages: BTC, ADA, SOL. Each coin page fetches history, trains a small model, "
    "then creates nearest-random samples from the most recent window (Gaussian noise)."
)

# -------------- Helpers --------------
@st.cache_data
def fetch_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Robustly fetch from yfinance and return DataFrame with columns:
    date, open, high, low, close, volume
    Works when yfinance returns single-level or MultiIndex columns like ('Close','BTC-USD').
    """
    try:
        raw = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
    except Exception as e:
        st.error(f"yfinance raised an exception for {ticker}: {e}")
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    # If columns are MultiIndex, we need to pick the right (first-level, second-level==ticker) tuple
    cols = raw.columns
    df = raw.copy()

    def pick_col(primary_names):
        """
        Try to pick a column from raw matching primary_names list like ['Close','Adj Close'].
        Handles MultiIndex and single Index.
        Returns column label (may be tuple for MultiIndex) or None.
        """
        # MultiIndex case
        if isinstance(cols, pd.MultiIndex):
            # prefer where second level equals ticker (or matches str ticker)
            for p in primary_names:
                for col in cols:
                    # col is a tuple like ('Close', 'BTC-USD') or ('Close', '')
                    if str(col[0]).lower() == p.lower() and (len(col) > 1 and str(col[1]) == ticker):
                        return col
            # fallback: match primary name ignoring second level
            for p in primary_names:
                for col in cols:
                    if str(col[0]).lower() == p.lower():
                        return col
            return None
        else:
            # single-level index
            for p in primary_names:
                for col in cols:
                    if str(col).lower() == p.lower():
                        return col
            return None

    # Preferred primary names
    close_col = pick_col(["Close", "Adj Close", "Adj_Close", "AdjClose"])
    open_col = pick_col(["Open"])
    high_col = pick_col(["High"])
    low_col = pick_col(["Low"])
    vol_col = pick_col(["Volume", "Vol"])

    # Date handling: raw.index is usually DatetimeIndex; we'll reset_index so 'Date' appears as a column.
    df = df.reset_index()

    # If reset_index produced MultiIndex columns (e.g. ('Date','')), handle accordingly
    # Build a helper to access df column safely whether label is tuple or string
    def get_series(label):
        try:
            return df[label]
        except Exception:
            # if label is tuple and not found, try fallback by string name match
            if isinstance(label, tuple):
                # try first element match
                for c in df.columns:
                    if str(c).lower() == str(label[0]).lower():
                        return df[c]
            # fallback None
            return None

    # Find date column in the reset df: prefer a column equal to 'Date' or the first column (index)
    date_col = None
    for c in df.columns:
        if str(c).lower() in ("date", "datetime", "index"):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    # Pull the actual series for each required column using the labels found earlier
    close_ser = get_series(close_col) if close_col is not None else None
    open_ser = get_series(open_col) if open_col is not None else None
    high_ser = get_series(high_col) if high_col is not None else None
    low_ser = get_series(low_col) if low_col is not None else None
    vol_ser = get_series(vol_col) if vol_col is not None else None

    if close_ser is None:
        # show diagnostic to user
        st.warning(f"No close column found for {ticker}. Available columns after reset: {list(df.columns)}")
        return pd.DataFrame()

    norm = pd.DataFrame()
    norm["date"] = pd.to_datetime(df[date_col])
    norm["open"] = open_ser if open_ser is not None else pd.NA
    norm["high"] = high_ser if high_ser is not None else pd.NA
    norm["low"] = low_ser if low_ser is not None else pd.NA
    norm["close"] = close_ser
    norm["volume"] = vol_ser if vol_ser is not None else pd.NA

    norm = norm.dropna(subset=["close"]).reset_index(drop=True)
    return norm

def create_lag_features(closes: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(closes)):
        X.append(closes[i-window:i])
        y.append(closes[i])
    if len(X) == 0:
        return np.empty((0, window), dtype=float), np.empty((0,), dtype=float)
    return np.array(X, dtype=float), np.array(y, dtype=float)

def train_model_on_closes(closes: np.ndarray, window: int, n_estimators: int, test_ratio: float):
    if len(closes) <= window:
        raise ValueError(f"Not enough data: need more than window={window}, got {len(closes)}")
    X, y = create_lag_features(closes, window)
    valid = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
    X, y = X[valid], y[valid]
    if X.shape[0] < 2:
        raise ValueError("Not enough valid training rows after dropping NaNs.")
    split = int(len(X) * (1 - test_ratio))
    if split < 1:
        split = 1
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test) if len(X_test) > 0 else np.array([])
    mse = float(mean_squared_error(y_test, preds)) if len(preds) > 0 else None
    return model, mse

def predict_with_nearest_randoms(model, last_window: np.ndarray, n_samples: int, noise_scale: float):
    last_window = np.array(last_window, dtype=float).reshape(1, -1)
    w = last_window.shape[1]
    base_std = float(np.std(last_window)) if np.std(last_window) > 0 else 1.0
    noise_sigma = base_std * noise_scale
    samples = []
    for i in range(n_samples):
        noise = np.random.normal(loc=0.0, scale=noise_sigma, size=(w,))
        sample = (last_window.flatten() + noise).astype(float)
        samples.append(sample)
    X_samples = np.vstack(samples).astype(float)
    preds = model.predict(X_samples)
    return preds, X_samples

def save_model(model, ticker):
    path = os.path.join(MODEL_DIR, f"rf_{ticker.replace('-', '_')}.joblib")
    joblib.dump(model, path)
    return path

def plot_history_and_preds(df, pred_point=None, preds_samples=None):
    fig, ax = plt.subplots(figsize=(7,3.5))
    ax.plot(df['date'], df['close'], label='Close')
    if pred_point is not None:
        next_day = df['date'].max() + pd.Timedelta(days=1)
        ax.scatter([next_day], [pred_point], color='red', marker='X', s=80, label='Predicted next-day')
    if preds_samples is not None:
        next_day = df['date'].max() + pd.Timedelta(days=1)
        ax.scatter([next_day]*len(preds_samples), preds_samples, alpha=0.4, label='Nearest-random preds')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close')
    ax.legend()
    plt.tight_layout()
    return fig

# -------------- Sidebar settings --------------
st.sidebar.header("Global settings")
page = st.sidebar.selectbox("Open page", options=["Home", "BTC-USD", "ADA-USD", "SOL-USD"])
window = st.sidebar.slider("Lag window (days)", min_value=3, max_value=30, value=7)
n_estimators = st.sidebar.slider("RF n_estimators", min_value=20, max_value=500, value=120, step=10)
test_ratio = st.sidebar.slider("Validation ratio", min_value=0.05, max_value=0.4, value=0.1, step=0.05)
period = st.sidebar.selectbox("Data period", options=["1y","2y","5y","max"], index=0)
interval = st.sidebar.selectbox("Data interval", options=["1d","1wk"], index=0)
n_samples = st.sidebar.number_input("Nearest-random samples", min_value=5, max_value=500, value=50, step=5)
noise_scale = st.sidebar.slider("Noise scale (fraction of recent std)", min_value=0.01, max_value=1.0, value=0.05, step=0.01)

# -------------- Pages --------------
def page_home():
    st.header("Home")
    st.markdown("""
    This app has separate pages for each coin (BTC, ADA, SOL).
    Use the sidebar to choose a page. Each coin page trains a small RandomForest on lag features,
    then generates nearest-random windows by adding small Gaussian noise to the latest window.
    """)

def page_coin(ticker: str):
    st.header(f"{ticker} — per-coin page")
    st.write("Fetches historical data, trains model, and shows nearest-random sample distribution.")

    with st.spinner("Fetching data..."):
        df = fetch_data(ticker, period=period, interval=interval)
    if df is None or df.empty:
        st.error(f"Could not fetch data for {ticker}. Try changing period/interval or check network.")
        return

    st.write(f"Data fetched: {len(df)} rows. Showing last 5 rows:")
    st.dataframe(df[['date','close']].tail(5))

    if st.button(f"Train & sample predict ({ticker})", key=f"train_{ticker}"):
        try:
            closes = df['close'].values.astype(float)
            model, mse = train_model_on_closes(closes, window=window, n_estimators=n_estimators, test_ratio=test_ratio)
            model_path = save_model(model, ticker)
            st.success(f"Trained model saved to {model_path}")
            if mse is not None:
                st.write("Validation MSE:", round(mse, 6))
            last_window = closes[-window:]
            preds_samples, X_samples = predict_with_nearest_randoms(model, last_window, n_samples=n_samples, noise_scale=noise_scale)
            mean_pred = float(np.mean(preds_samples))
            std_pred = float(np.std(preds_samples))
            st.metric(label="Predicted next-day (mean of samples)", value=f"{mean_pred:.6f}")
            st.write(f"Sample preds — mean: {mean_pred:.6f}, std: {std_pred:.6f}, n_samples: {len(preds_samples)}")
            fig_hist, ax = plt.subplots(figsize=(6,3))
            ax.hist(preds_samples, bins=25, alpha=0.8)
            ax.set_title("Histogram of predictions")
            st.pyplot(fig_hist)
            fig_time = plot_history_and_preds(df, pred_point=mean_pred, preds_samples=preds_samples)
            st.pyplot(fig_time)
            if st.checkbox("Show first 5 sampled input windows (nearest-random)", key=f"sample_table_{ticker}"):
                sampled_df = pd.DataFrame(X_samples[:5], columns=[f"lag_{i}" for i in range(window,0,-1)])
                st.dataframe(sampled_df)
        except Exception as e:
            st.error("Train/predict failed: " + str(e))

    if st.button(f"Quick predict (single) {ticker}", key=f"quick_{ticker}"):
        try:
            closes = df['close'].values.astype(float)
            model, mse = train_model_on_closes(closes, window=window, n_estimators=n_estimators, test_ratio=test_ratio)
            last_window = closes[-window:]
            next_pred = float(model.predict(last_window.reshape(1,-1))[0])
            st.metric(label="Single-run predicted next-day", value=f"{next_pred:.6f}")
            if mse is not None:
                st.write("Validation MSE:", round(mse,6))
            fig_time = plot_history_and_preds(df, pred_point=next_pred)
            st.pyplot(fig_time)
        except Exception as e:
            st.error("Quick predict failed: " + str(e))

# Router
if page == "Home":
    page_home()
elif page == "BTC-USD":
    page_coin("BTC-USD")
elif page == "ADA-USD":
    page_coin("ADA-USD")
elif page == "SOL-USD":
    page_coin("SOL-USD")
else:
    st.info("Select a page.")

st.markdown("---")
st.caption("Notes: educational demo. For real trading use thorough backtesting and better features.")