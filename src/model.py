""" 
NIFTY50 PRICE PREDICTION

This Program is made to predict the next days NIFTY50 index price using machine learning 

Importing Useful libraries
From sklearn - StandardScaler, mean_squared_error, r2_score, mean_absolute_error, train_test_split, LinearRegression, RandomForestRegressor, SVR, DecisionTreeRegressor and some tree libraries Lightgbm and Xgboost Pandas, Numpy, Joblib 
"""

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os
import shap


# Feature Engineering
def feature_engineering(data):
    data = data.copy()
    data['Open'] = np.log(data['Open'])
    data['High'] = np.log(data['High'])
    data['Low'] = np.log(data['Low'])
    data['Price'] = np.log(data['Price'])
    data['Vol.'] = np.log(data['Vol.'] + 1)
    data['Rel_Vol.'] = data['Vol.'] / data['Vol.'].rolling(20).mean()
    data['Return_10'] = data['Price'].pct_change(10)
    data['Return_3'] = data['Price'].pct_change(3)
    data['Volatility'] = data['Change %'].rolling(20).std() / 100
    data['Volatility_5'] = data['Change %'].rolling(5).std() / 100
    data['MA20'] = data['Price'].rolling(20).mean()
    data['MA50'] = data['Price'].rolling(50).mean()
    data['Trend20'] = (data['Price'] - data['MA20']) / data['MA20']
    data['Trend50'] = (data['Price'] - data['MA50']) / data['MA50']
    data['HL_Range'] = (data['High'] - data['Low']) / data['Price']
    data['Gap'] = (data['Open'] - data['Price'].shift(1)) / data['Price'].shift(1)
    data['Momentum_5'] = data['Price'] / data['Price'].shift(5) - 1
    data['Momentum_10'] = data['Price'] / data['Price'].shift(10) - 1

    # Calculate RSI
    delta = data['Price'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data['Target'] = data['Change %'].shift(-1) / 100  # Predict next day's return

    data.drop(columns=['MA20', 'MA50'], inplace=True)
    data.dropna(inplace=True)

    return data


# Creating Piplines to Train Different models and measure metrics
def creating_pipeline(X_train, X_test, y_train, y_test):
    tree_models = {
        "rf": RandomForestRegressor(random_state=42),
        "xgb": XGBRegressor(random_state=42),
        "lgbm": LGBMRegressor(random_state=42)
    }

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    linear_models = {
        "lr": LinearRegression(),
        "svr": SVR(kernel='rbf')
    }

    # Tree models
    for name, model in tree_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print_metrics(name, y_test, y_pred)

    # Linear models
    for name, model in linear_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print_metrics(name, y_test, y_pred)


# If I trade based on the prediction, how much money do I make? -> That’s strategy return.
# If prediction > 0 → go LONG
# If prediction < 0 → go SHORT

def print_metrics(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    direction_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

    position = np.sign(y_pred)

    strategy_return = position * y_true
    transaction_cost = 0.0005  # 5 bps example
    position_change = np.abs(np.diff(position, prepend=0))

    strategy_return -= transaction_cost * position_change

    strategy_return = np.nan_to_num(strategy_return)
    strategy_return = np.clip(strategy_return, -0.99, 10)

    cumulative_return = np.cumprod(1 + strategy_return)

    all_metrics = {
        "MSE": mse,
        "MAE": mae,
        "Directional Accuracy": direction_acc,
        "Final Equity": float(cumulative_return[-1])
    }

    with open(os.path.join(RESULTS_DIR, f'{name}_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)


# Fine Tuning XGB model as it is best so far
def fine_tuning(X_train, X_test, y_train, y_test):
    def equity_score(y_test, y_pred):
        position = np.sign(y_pred)
        strategy_return = position * y_test

        transaction_cost = 0.0005
        position_change = np.abs(np.diff(position, prepend=0))
        strategy_return -= transaction_cost * position_change

        strategy_return = np.nan_to_num(strategy_return)
        strategy_return = np.clip(strategy_return, -0.99, 10)

        cumulative_return = np.cumprod(1 + strategy_return)
        return float(cumulative_return[-1])

    scorer = make_scorer(equity_score, greater_is_better=True)

    param_grid = {
        "n_estimators": [200, 300, 400],
        "learning_rate": [0.01, 0.02, 0.03, 0.0001, 0.005],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3, 5]
    }

    model = XGBRegressor(random_state=42)

    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=20,
        scoring=scorer,
        cv=tscv,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    print_metrics("best_xgb", y_test, y_pred)

    # Pickling the model for production
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_xgb_model.pkl'))

    with open(os.path.join(MODELS_DIR, 'best_xgb_params.json'), 'w') as f:
        json.dump(search.best_params_, f, indent=4)
    
    with open(os.path.join(MODELS_DIR, 'trained_features.json'), 'w') as f:
        json.dump(list(X_train.columns), f)

    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
    plt.savefig(os.path.join(PLOTS_DIR, 'shap_summary.png'), bbox_inches='tight')


def train_and_tune(data):
    data = feature_engineering(data)

    # Splitting the Data into Train and Test
    train_size = int(len(data) * 0.8)

    y_train = data['Target'][:train_size]
    y_test = data['Target'][train_size:]

    data.drop(columns=['Target'], inplace=True)
    X_train = data[:train_size]
    X_test = data[train_size:]

    creating_pipeline(X_train, X_test, y_train, y_test)
    best_model = fine_tuning(X_train, X_test, y_train, y_test)

    return best_model

if __name__ == "__main__":
    # Loading Datasets
    # Converting Date column to datetime type
    # Setting index of DataFrame to Date column

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    data = pd.read_csv(os.path.join(DATA_DIR, 'data1.csv'), parse_dates=['Date'],index_col='Date')
    data.sort_index(inplace=True)

    train_and_tune(data)