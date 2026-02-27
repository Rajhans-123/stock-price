import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pickle
from xgboost import XGBRegressor
import json

def retrain_model(df, MODELS_DIR):
    import model
    from ui import ui
    df = model.feature_engineering(df)

    # Creating Target column (next day's return)
    df['Target'] = df['Change %'].shift(-1) / 100

    df = df.dropna()  # Drop rows with missing values

    # Features AND Target
    y = df['Target'] # Next day's return
    X = df[ui.TRAINED_FEATURES]

    with open(os.path.join(MODELS_DIR, 'best_xgb_params.json'), 'r') as f:
        best_params = json.load(f)

    model_train = XGBRegressor(**best_params, random_state = 42)

    if ui.ticker < 150:
        model_train.fit(X, y)
        with open(ui.MODEL_PATH, "wb") as f:
            pickle.dump(model_train, f)
        print("Model retrained and saved.")
    else:
        best_model = model.train_and_tune(df)
        with open(os.path.join(MODELS_DIR, 'best_xgb_model.pkl'), 'wb') as f:
            pickle.dump(best_model, f)
        print("Model retrained and saved.")

if __name__ == '__main__':
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(DATA_DIR, 'data1.csv'), parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)

    retrain_model(df, MODELS_DIR)