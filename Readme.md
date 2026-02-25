# NIFTY50 Stock Price Prediction

This project predicts the next day's NIFTY50 index price using various machine learning models. It includes data preprocessing, feature engineering, model training, evaluation, and a simple trading strategy backtest.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features
- Data collection and preprocessing for NIFTY50 historical data
- Feature engineering including technical indicators (RSI, moving averages, volatility, etc.)
- Training and evaluation of multiple ML models
- Directional accuracy and strategy return metrics
- Jupyter notebooks for EDA and modeling

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Rajhans-123/stock-price.git
   cd stock-price
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Data Collection**: Run `src/getting_data.py` to fetch or prepare the data.
2. **Exploratory Data Analysis**: Open `src/eda.ipynb` for data exploration.
3. **Model Training and Evaluation**: Run `src/model.ipynb` to train models and view results.
4. **Results**: Check `results/` for model performance metrics.

## Data
- **Source**: NIFTY50 historical data (CSV format)
- **Features**: Date, Open, High, Low, Close, Volume, plus engineered features like Return, RSI, Volatility, etc.
- **Target**: Next day's percentage return

## Models
The following models are trained and evaluated:
- **Linear Regression (lr)**: Baseline linear model
- **Random Forest (rf)**: Ensemble tree-based model
- **XGBoost (xgb)**: Gradient boosting
- **LightGBM (lgbm)**: Efficient gradient boosting
- **SVR (svr)**: Support Vector Regression

### Metrics
- **MSE**: Mean Squared Error (lower better)
- **MAE**: Mean Absolute Error (lower better)
- **Directional Accuracy**: Percentage of correct direction predictions (higher better)
- **Final Equity**: Cumulative return from a simple long/short strategy based on predictions (higher better)

## Results
Based on the latest evaluation:

| Model | MSE                   | MAE                   | Directional Accuracy   | Final Equity           |
|-------|-----------------------|-----------------------|------------------------|------------------------|
| rf    | 6.385532388821028e-05 | 0.005718401569638982  | 0.5046583850931677     | 1.3472072423355714     |
| xgb   | 6.580884924759668e-05 | 0.005803201699954351  | 0.5077639751552795     | 0.9168006578547507     |
| lgbm  | 6.148874471814435e-05 | 0.005657574433607601  | 0.5201863354037267     | 1.2918106564069654     |
| lr    | 6.01706938586862e-05  | 0.0055856916861299465 | 0.5015527950310559     | 1.1231970568765066     |
| svr   | 0.0005249218994347663 | 0.021713136362065108  | 0.46273291925465837    | 0.7315102318087634     |

**Best Model**: LightGBM (highest final equity)

### Final LightGBM Metrics
- **MSE**: 6.0311266597022825e-05
- **MAE**: 0.0055584900320102566
- **Directional Accuracy**: 0.5667701863354038
- **Final Equity**: 1.9595122230739497

### Plots
Below are two key EDA plots included in the repository:

- **100 Day Moving Average**

   ![100 Day Moving Average](plots/100_Day_Moving_Average.png)

- **Price to Rolling Close Ratio**

   ![Price to Rolling Close Ratio](plots/Price_to_Rolling_Close_Ratio.png)

### Key Learning
Applying Log Transformation before and after Feature Engineering - 

Before -> Final Equity = 1.003
After -> Final Equity = 1.959

In this experiment, the difference in performance comes from how the data pipeline interacts with financial time-series structure. Applying log transformation after feature engineering preserved the raw market movement characteristics during feature construction, allowing rolling statistics and momentum signals to capture more meaningful price dynamics. When log transformation was applied early, it likely reduced signal amplitude and weakened relationships between engineered features, leading to a model that could not extract strong predictive patterns. Since financial market data is highly noisy and does not strictly follow normal distribution assumptions, empirical performance can sometimes be better than theoretical preprocessing order. The comparison shows that the late log-transformation pipeline produced stronger directional signals and higher backtest equity, indicating better practical predictive utility for your dataset and model.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.</content>

## Contact Information
email - rentalarrajhans@gmail.com
