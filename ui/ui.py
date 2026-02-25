import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# ==============================
# CONFIG
# ==============================

MODEL_PATH = "models/best_xgb_model.pkl"
BUFFER_FILE = "history_buffer.csv"
WINDOW_SIZE = 100   # Keep more than 50 for safety

# Load model
model = pickle.load(open(MODEL_PATH, "rb"))

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
    "RSI"
]

# ==============================
# HISTORY MANAGEMENT
# ==============================

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


# ==============================
# ROUTE
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    message = None

    if request.method == "POST":

        history_df = load_history()

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
            pred = model.predict(latest_row.values.reshape(1, -1))[0]
            prediction = float(pred)
        except Exception as e:
            message = f"Model prediction error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)