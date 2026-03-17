# ADA-group-project

Weather forecasting with HKO (Hong Kong Observatory) data: data preprocessing, three models (Logistic Regression, LSTM, SARIMA), evaluation, and prediction/visualization.

## Setup

Use a Python environment with the dependencies in `requirements.txt` (e.g. conda/venv with pandas, scikit-learn, matplotlib, statsmodels; optional: tensorflow for LSTM). Run all cells in `ada_project.ipynb` (kernel: e.g. withPytorch or any env with the packages).

## Structure

- `ada_project.ipynb` — Full pipeline: load/preprocess (Section 1), train/validate/test three models (Section 2), compare and predict (Section 3).
- `utils/` — Helpers: `data_utils` (load, split), `metrics` (MSE, RMSE, MAE, MAPE, Accuracy, F1), `model_logistic`, `model_lstm`, `model_sarima`, `visualization`.
- `data/` — Input and generated CSVs (raw, for ARIMA, normalized for LSTM).