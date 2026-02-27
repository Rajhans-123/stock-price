import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import re_train
import threading

app = Flask(__name__)

# ==============================
# CONFIG
# ==============================

MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, 'best_xgb_model.pkl')
FULL_DATA_FILE = os.path.join(DATA_DIR, 'data1.csv')
BUFFER_FILE = os.path.join(DATA_DIR, 'history_buffer.csv')

WINDOW_SIZE = 100   # Keep more than 50 for safety

# Load model
model = pickle.load(open(MODEL_PATH, "rb"))
ticker = 0
model_lock = threading.Lock()

TRAINED_FEATURES = [
    "Price",
    "Open",
    "High",
    "Low",
    "Vol.",
    "Change %",
    "Rel_Vol.",
    "Return_10",
    "Return_3",
    "Volatility",
    "Volatility_5",
    "Trend20",
    "Trend50",
    "HL_Range",
    "Gap",
    "RSI",
    "Momentum_5",
    "Momentum_10"
]

# ==============================
# HISTORY MANAGEMENT
# ==============================

def load_full_data():
    if os.path.exists(FULL_DATA_FILE):
        df = pd.read_csv(FULL_DATA_FILE)

        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)

        return df.reset_index(drop=True)

    else:
        return pd.DataFrame(
            columns=["Date", "Open", "High", "Low", "Price", "Change %", "Vol."]
        )


def save_full_data(df):
    df = df.copy()

    df.drop_duplicates(subset=["Date"], inplace=True)
    df.sort_values("Date", inplace=True)

    df.to_csv(FULL_DATA_FILE, index=False)

    print("Full dataset safely updated.")


def load_history():
    if os.path.exists(BUFFER_FILE):
        df = pd.read_csv(BUFFER_FILE)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        return df
    else:
        return pd.DataFrame(columns=[
            "Date", "Open", "High", "Low",
            "Price", "Change %", "Vol."
        ])


def save_history(df):
    df.drop_duplicates(subset=["Date"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.to_csv(BUFFER_FILE, index=False)


# ==============================
# FEATURE ENGINEERING
# ==============================

def feature_engineer(df):

    df = df.copy()

    # LOG TRANSFORM (must match training pipeline)
    df["Open"] = np.log(df["Open"])
    df["High"] = np.log(df["High"])
    df["Low"] = np.log(df["Low"])
    df["Price"] = np.log(df["Price"])
    df["Vol."] = np.log(df["Vol."] + 1)

    # Relative Volume
    df["Rel_Vol."] = df["Vol."] / df["Vol."].rolling(20).mean()

    # Returns
    df["Return_10"] = df["Price"].pct_change(10)
    df["Return_3"] = df["Price"].pct_change(3)

    # Volatility
    df["Volatility"] = df["Change %"].rolling(20).std() / 100
    df["Volatility_5"] = df["Change %"].rolling(5).std() / 100

    # Moving Averages
    df["MA20"] = df["Price"].rolling(20).mean()
    df["MA50"] = df["Price"].rolling(50).mean()

    df["Trend20"] = (df["Price"] - df["MA20"]) / df["MA20"]
    df["Trend50"] = (df["Price"] - df["MA50"]) / df["MA50"]

    # Range & Gap
    df["HL_Range"] = (df["High"] - df["Low"]) / df["Price"]
    df["Gap"] = (df["Open"] - df["Price"].shift(1)) / df["Price"].shift(1)
    df['Momentum_5'] = df['Price'] / df['Price'].shift(5) - 1
    df['Momentum_10'] = df['Price'] / df['Price'].shift(10) - 1
    df['Change %'] = df['Price'].pct_change() * 100

    # RSI (Wilder)
    delta = df["Price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Remove helper MAs
    df.drop(columns=["MA20", "MA50"], inplace=True)

    return df

def background_retrain(df):
    global model

    try:
        re_train.retrain_model(df, MODELS_DIR)

        # Reload only after retrain completes
        with model_lock:
            model = pickle.load(open(MODEL_PATH, 'rb'))
        print("Model successfully retrained and reloaded")

    except Exception as e:
        print("Retrain failed:", e)

# ==============================
# ROUTE
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    global model
    global ticker

    prediction = None
    message = None

    if request.method == "POST":

        history_df = load_history()
        full_df = load_full_data()

        try:
            open_price = float(request.form["Open"])
            high_price = float(request.form["High"])
            low_price = float(request.form["Low"])
            close_price = float(request.form["Price"])
            volume = float(request.form["Vol."])
        except:
            message = "Invalid input format"
            return render_template("index.html", prediction=None, message=message)

        new_row = pd.DataFrame([{
            "Date": pd.Timestamp.now(),
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
            "Price": close_price,
            "Vol.": volume
        }])

        # Calculate Change %
        if len(history_df) > 0:
            prev_price = history_df.iloc[-1]["Price"]
            change_pct = ((close_price - prev_price) / prev_price) * 100
        else:
            change_pct = 0.0

        new_row["Change %"] = change_pct

        # Append to full dataset (never truncated)
        raw_row = new_row[[
            "Date",
            "Open",
            "High",
            "Low",
            "Price",
            "Vol.",
            "Change %"
        ]]

        full_df = pd.concat([full_df, raw_row], axis=0)
        full_df.reset_index(drop=True, inplace=True)

        save_full_data(full_df)

        # Append
        history_df = pd.concat([history_df, new_row], ignore_index=True)

        # Keep rolling window
        if len(history_df) > WINDOW_SIZE:
            history_df = history_df.iloc[-WINDOW_SIZE:]

        save_history(history_df)

        # ==============================
        # PREDICTION PIPELINE
        # ==============================

        df_features = feature_engineer(history_df)

        # Check enough history
        if len(df_features) < 60:
            message = "Not enough historical data (need ~60 candles)"
            return render_template("index.html", prediction=None, message=message)

        latest_row = df_features[TRAINED_FEATURES].iloc[-1]

        if latest_row.isna().any():
            message = "Features not ready yet (rolling window building)"
            return render_template("index.html", prediction=None, message=message)

        try:
            with model_lock:
                pred = model.predict(latest_row.values.reshape(1, -1))[0]
            prediction = float(pred)

            ticker += 1

            if ticker > 50:
                threading.Thread(
                    target=background_retrain,
                    args=(full_df.copy(),),
                    daemon=True
                ).start()

                ticker = 0

        except Exception as e:
            message = f"Model prediction error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=False)