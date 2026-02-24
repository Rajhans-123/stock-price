import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model
model = pickle.load(open("../NIFTY50/models/best_lgbm_model.pkl", "rb"))

# Training feature list
TRAINED_FEATURES = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "Return",
    "Rel_Volume",
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

# -----------------------------
# Historical Window Buffer
# -----------------------------

BUFFER_FILE = "history_buffer.csv"
WINDOW_SIZE = 50


def load_history():
    try:
        df = pd.read_csv(BUFFER_FILE)
    except:
        df = pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    return df


def save_history(df):
    df.to_csv(BUFFER_FILE, index=False)


# -----------------------------
# Feature Engineering Pipeline
# -----------------------------

def feature_engineer(df):

    df['Return'] = df['Close'].pct_change()
    df['Rel_Volume'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-6)
    df['Return_10'] = df['Close'].pct_change(10)
    df['Return_3'] = df['Close'].pct_change(3)

    delta = df['Close'].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-6)

    df['Volatility'] = df['Return'].rolling(20).std()
    df['Volatility_5'] = df['Return'].rolling(5).std()

    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    df['Trend20'] = (df['Close'] - df['MA20']) / (df['MA20'] + 1e-6)
    df['Trend50'] = (df['Close'] - df['MA50']) / (df['MA50'] + 1e-6)

    df['HL_Range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-6)

    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-6)
    df['RSI'] = 100 - (100 / (1 + rs))

    df.drop(columns=['MA20','MA50'], inplace=True)

    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(0, inplace=True)

    return df


# -----------------------------
# Flask Route
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        history_df = load_history()

        new_row = pd.DataFrame([{
            "Open": float(request.form["Open"]),
            "High": float(request.form["High"]),
            "Low": float(request.form["Low"]),
            "Close": float(request.form["Close"]),
            "Volume": float(request.form["Volume"])
        }])

        # Append new candle
        history_df = pd.concat([history_df, new_row], ignore_index=True)

        # Keep window size
        if len(history_df) > WINDOW_SIZE:
            history_df = history_df.iloc[-WINDOW_SIZE:]

        save_history(history_df)

        # Feature pipeline
        df = feature_engineer(history_df.copy())

        if len(df) > 0:
            prediction = float(
                model.predict(
                    df[TRAINED_FEATURES].iloc[-1:].values
                )[0]
            )

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)