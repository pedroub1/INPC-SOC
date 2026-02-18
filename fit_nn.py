"""
Pre-entrena modelos MLP (sklearn) y LSTM (PyTorch) para cada serie del INPC.
Guarda modelos en data/nn_models/ y configuracion en data/nn_config.json.

Ejecutar manualmente cuando se quiera recalibrar:
    python fit_nn.py
    python fit_nn.py --quincenal
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from data_loader import load_data, load_data_q

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(__file__)


# ============================================================
# Preparacion de datos
# ============================================================

def make_supervised(serie, n_lags, period):
    """Convierte serie en tabla supervisada con rezagos + features calendario."""
    df = pd.DataFrame({"y": serie.values}, index=serie.index)
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = serie.shift(i).values
    df["sin_month"] = np.sin(2 * np.pi * serie.index.month / period)
    df["cos_month"] = np.cos(2 * np.pi * serie.index.month / period)
    df = df.dropna()
    X = df.drop("y", axis=1).values
    y = df["y"].values
    return X, y


# ============================================================
# MLP
# ============================================================

def fit_mlp(serie, n_lags, period):
    """Entrena MLP con GridSearch y devuelve modelo + scalers."""
    X, y = make_supervised(serie, n_lags, period)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    param_grid = {
        "hidden_layer_sizes": [(64, 32), (32, 16), (64,)],
        "alpha": [0.001, 0.01, 0.1],
    }

    base_model = MLPRegressor(
        activation="relu",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )

    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(base_model, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(X_scaled, y_scaled)

    best = grid.best_estimator_
    best_params = grid.best_params_
    best_score = -grid.best_score_

    return best, scaler_X, scaler_y, best_params, best_score


# ============================================================
# LSTM
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size=26, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len=1, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def fit_lstm(serie, n_lags, period, val_size=12, epochs=100, batch_size=32, patience=10):
    """Entrena LSTM con early stopping y devuelve modelo + scalers."""
    X, y = make_supervised(serie, n_lags, period)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Train/val split
    X_train, X_val = X_scaled[:-val_size], X_scaled[-val_size:]
    y_train, y_val = y_scaled[:-val_size], y_scaled[-val_size:]

    # Convertir a tensores — shape (N, 1, features)
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val).unsqueeze(1)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

    input_size = X_train.shape[1]
    model = LSTMModel(input_size=input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        # Validacion
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, scaler_X, scaler_y, best_val_loss


# ============================================================
# Guardar modelos
# ============================================================

def save_models(serie_name, mlp_model, mlp_scaler_X, mlp_scaler_y,
                lstm_model, lstm_scaler_X, lstm_scaler_y, models_dir):
    """Guarda modelos MLP y LSTM para una serie."""
    safe_name = serie_name.strip().replace(" ", "_").replace(".", "")
    series_dir = os.path.join(models_dir, safe_name)
    os.makedirs(series_dir, exist_ok=True)

    # MLP
    joblib.dump(mlp_model, os.path.join(series_dir, "mlp_model.pkl"))
    joblib.dump(mlp_scaler_X, os.path.join(series_dir, "mlp_scaler_X.pkl"))
    joblib.dump(mlp_scaler_y, os.path.join(series_dir, "mlp_scaler_y.pkl"))

    # LSTM
    torch.save(lstm_model.state_dict(), os.path.join(series_dir, "lstm_model.pt"))
    joblib.dump(lstm_scaler_X, os.path.join(series_dir, "lstm_scaler_X.pkl"))
    joblib.dump(lstm_scaler_y, os.path.join(series_dir, "lstm_scaler_y.pkl"))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pre-entrena modelos MLP + LSTM")
    parser.add_argument("--quincenal", action="store_true",
                        help="Usar series quincenales en lugar de mensuales")
    args = parser.parse_args()

    if args.quincenal:
        df = load_data_q()
        n_lags = 48
        period = 24
        freq = None  # quincenal (1o y 16) no tiene freq estandar en pandas
        val_size = 24
        models_dir = os.path.join(BASE_DIR, "data", "nn_models_q")
        config_path = os.path.join(BASE_DIR, "data", "nn_config_q.json")
        label = "quincenal"
    else:
        df = load_data()
        n_lags = 24
        period = 12
        freq = "MS"
        val_size = 12
        models_dir = os.path.join(BASE_DIR, "data", "nn_models")
        config_path = os.path.join(BASE_DIR, "data", "nn_config.json")
        label = "mensual"

    if df is None:
        print("No hay datos. Ejecuta primero la descarga desde el dashboard.")
        return

    n_features = n_lags + 2

    series_names = df.columns.tolist()
    print(f"Entrenando MLP + LSTM para {len(series_names)} series ({label}, lags={n_lags})...\n")

    config = {}
    start = time.time()

    for name in series_names:
        t0 = time.time()
        serie = df[name].dropna()
        if freq is not None:
            serie.index.freq = freq

        print(f"  {name:45s}", end="", flush=True)

        # MLP
        mlp_model, mlp_sX, mlp_sY, mlp_params, mlp_mse = fit_mlp(serie, n_lags, period)

        # LSTM
        lstm_model, lstm_sX, lstm_sY, lstm_val_loss = fit_lstm(
            serie, n_lags, period, val_size=val_size)

        # Guardar
        save_models(name, mlp_model, mlp_sX, mlp_sY, lstm_model, lstm_sX, lstm_sY, models_dir)

        safe_name = name.strip().replace(" ", "_").replace(".", "")
        config[name] = {
            "safe_name": safe_name,
            "n_lags": n_lags,
            "n_features": n_features,
            "mlp_params": mlp_params,
            "mlp_cv_mse": float(mlp_mse),
            "lstm_val_mse": float(lstm_val_loss),
            "lstm_hidden_size": 32,
            "lstm_num_layers": 1,
        }

        dt = time.time() - t0
        print(f"  MLP{mlp_params['hidden_layer_sizes']} mse={mlp_mse:.6f}  "
              f"LSTM val_mse={lstm_val_loss:.6f}  ({dt:.1f}s)")

    elapsed = time.time() - start
    print(f"\nListo en {elapsed:.1f}s")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Configuracion guardada en {config_path}")


if __name__ == "__main__":
    main()
