"""
models.py – 38 modelos de pronostico para soc_us (CPI/PCE EE.UU.).

Diferencias respecto al modulo SOC Mexico:
  1. Phillips Curve: 3 predictores (MSC + slack + ENERGY), segun Verona (2026)
     PC1: MSC + U + ENERGY
     PC2: MSC + JWG + ENERGY
     PC3: MSC + SAHM + ENERGY
  2. PLS-2: igual a PLS-1 pero con predictores descompuestos por frecuencia;
     el expanding_window pasa X_filtered cuando el modelo tiene use_filtered_x=True.

Interfaz uniforme: fn(y_train, X_train, h) -> float
  y_train: np.ndarray (T,)
  X_train: np.ndarray (T,k) o None
  h: int
"""
import warnings
import numpy as np
from typing import Callable

from .config import AR_MAX_LAGS, PCA_COMPONENTS, PLS_COMPONENTS, PC_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_const(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def _ols_predict(X_train, y_train, x_pred) -> float:
    X_c = _add_const(X_train)
    x_c = np.concatenate([[1.0], x_pred])
    beta, _, _, _ = np.linalg.lstsq(X_c, y_train, rcond=None)
    return float(x_c @ beta)


def _lag_matrix(y: np.ndarray, lags: int, h: int):
    n = len(y)
    rows, targets = [], []
    min_idx = lags + h - 1
    for t in range(min_idx, n):
        rows.append([y[t - h - lag] for lag in range(lags)])
        targets.append(y[t])
    if not rows:
        return np.empty((0, lags)), np.empty(0)
    return np.array(rows), np.array(targets)


def _select_ar_lags_ic(y: np.ndarray, h: int, max_lags: int, ic: str = "aic") -> int:
    best_ic = np.inf
    best_p = 1
    n = len(y)
    for p in range(1, max_lags + 1):
        X, target = _lag_matrix(y, p, h)
        if len(target) < p + 2:
            break
        X_c = _add_const(X)
        beta, residuals, _, _ = np.linalg.lstsq(X_c, target, rcond=None)
        if len(residuals) == 0:
            fitted = X_c @ beta
            ssr = float((target - fitted) @ (target - fitted))
        else:
            ssr = float(residuals[0]) if hasattr(residuals, "__len__") else float(residuals)
        k = p + 1
        T = len(target)
        if ssr <= 0:
            continue
        log_lik = -T / 2 * np.log(ssr / T)
        val = -2 * log_lik + (2 * k if ic == "aic" else k * np.log(T))
        if val < best_ic:
            best_ic = val
            best_p = p
    return best_p


def _nanmean_fallback(y: np.ndarray, window: int = 12) -> float:
    return float(np.nanmean(y[-window:]))


# ---------------------------------------------------------------------------
# AR models
# ---------------------------------------------------------------------------

def ar_aic(y_train: np.ndarray, X_train, h: int) -> float:
    p = _select_ar_lags_ic(y_train, h, AR_MAX_LAGS, "aic")
    X, target = _lag_matrix(y_train, p, h)
    if len(target) < 2:
        return _nanmean_fallback(y_train)
    X_c = _add_const(X)
    beta, _, _, _ = np.linalg.lstsq(X_c, target, rcond=None)
    last_row = np.array([y_train[-h - lag] for lag in range(p)])
    return float(np.concatenate([[1.0], last_row]) @ beta)


def ar_sic(y_train: np.ndarray, X_train, h: int) -> float:
    p = _select_ar_lags_ic(y_train, h, AR_MAX_LAGS, "sic")
    X, target = _lag_matrix(y_train, p, h)
    if len(target) < 2:
        return _nanmean_fallback(y_train)
    X_c = _add_const(X)
    beta, _, _, _ = np.linalg.lstsq(X_c, target, rcond=None)
    last_row = np.array([y_train[-h - lag] for lag in range(p)])
    return float(np.concatenate([[1.0], last_row]) @ beta)


# ---------------------------------------------------------------------------
# Phillips Curve – 3 predictores: MSC, slack, ENERGY
# (Especificacion del paper: Verona 2026, seccion 4.2.2)
# ---------------------------------------------------------------------------

def _pc3_model(y_train: np.ndarray, X_train: np.ndarray | None, h: int) -> float:
    """
    PC generico con 3 predictores externos (columnas 0, 1, 2 de X_train).
    y_{t} = a + b1*x0_{t-h} + b2*x1_{t-h} + b3*x2_{t-h}
    """
    if X_train is None or X_train.shape[1] < 3:
        return _nanmean_fallback(y_train)
    n = len(y_train)
    T = min(n, X_train.shape[0])
    X_rows, targets = [], []
    for t in range(h, T):
        x_lag = t - h
        if x_lag < 0:
            continue
        row = X_train[x_lag, :3]
        if np.isnan(y_train[t]) or np.any(np.isnan(row)):
            continue
        X_rows.append(row)
        targets.append(y_train[t])
    if len(targets) < 4:
        return _nanmean_fallback(y_train)
    X_arr = np.array(X_rows)
    y_arr = np.array(targets)
    X_c = _add_const(X_arr)
    beta, _, _, _ = np.linalg.lstsq(X_c, y_arr, rcond=None)
    last_row = X_train[-1, :3].copy()
    # Imputar NaN con media
    for i in range(3):
        if np.isnan(last_row[i]):
            col_vals = X_train[:, i]
            last_row[i] = float(np.nanmean(col_vals[-12:]))
    pred = np.concatenate([[1.0], last_row])
    return float(pred @ beta)


def make_pc_model(label: str) -> Callable:
    """Crea modelo Phillips Curve con 3 predictores (MSC, slack, ENERGY)."""
    def pc_fn(y_train: np.ndarray, X_train: np.ndarray | None, h: int) -> float:
        return _pc3_model(y_train, X_train, h)
    pc_fn.__name__ = label
    return pc_fn


# ---------------------------------------------------------------------------
# Bivariate OLS (1 predictor externo)
# ---------------------------------------------------------------------------

def make_bivariate(col_name: str) -> Callable:
    """AR(1) + predictor externo."""
    def biv_fn(y_train: np.ndarray, X_train: np.ndarray | None, h: int) -> float:
        if X_train is None or X_train.shape[1] == 0:
            return _nanmean_fallback(y_train)
        n = len(y_train)
        x_col = X_train[:, 0]
        min_idx = max(h, 1)
        X_rows, targets = [], []
        for t in range(min_idx, n):
            x_lag = t - h
            if x_lag < 0 or np.isnan(y_train[t - 1]) or np.isnan(x_col[x_lag]):
                continue
            X_rows.append([y_train[t - 1], x_col[x_lag]])
            targets.append(y_train[t])
        if len(targets) < 3:
            return _nanmean_fallback(y_train)
        X_arr = np.array(X_rows)
        y_arr = np.array(targets)
        X_c = _add_const(X_arr)
        beta, _, _, _ = np.linalg.lstsq(X_c, y_arr, rcond=None)
        x_last = x_col[-1] if not np.isnan(x_col[-1]) else float(np.nanmean(x_col))
        pred_row = np.array([1.0, y_train[-1], x_last])
        return float(pred_row @ beta)
    biv_fn.__name__ = f"BIV_{col_name}"
    return biv_fn


# ---------------------------------------------------------------------------
# Reduccion de dimension (PCA, PLS-1, PLS-2)
# ---------------------------------------------------------------------------

def _prepare_macro_matrix(y_train, X_train, h):
    n = len(y_train)
    if X_train is None or X_train.shape[1] == 0:
        return None, None, None
    T = min(n, X_train.shape[0])
    X_al = X_train[:T]
    y_al = y_train[:T]
    X_rows, y_rows = [], []
    for t in range(h, T):
        row = X_al[t - h]
        if np.any(np.isnan(row)) or np.isnan(y_al[t]):
            continue
        X_rows.append(row)
        y_rows.append(y_al[t])
    if len(y_rows) < 5:
        return None, None, None
    X_out = np.array(X_rows)
    y_out = np.array(y_rows)
    x_pred_raw = X_al[-1].copy()
    if np.any(np.isnan(x_pred_raw)):
        col_means = np.nanmean(X_al, axis=0)
        x_pred_raw = np.where(np.isnan(x_pred_raw), col_means, x_pred_raw)
    return X_out, y_out, x_pred_raw


def pca_model(y_train: np.ndarray, X_train: np.ndarray | None, h: int) -> float:
    """PCA (3 componentes) + OLS."""
    X, y, x_pred = _prepare_macro_matrix(y_train, X_train, h)
    if X is None:
        return _nanmean_fallback(y_train)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X_std = (X - mu) / sd
    x_pred_std = (x_pred - mu) / sd
    n_comp = min(PCA_COMPONENTS, X_std.shape[1], X_std.shape[0] - 1)
    try:
        U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
        F = U[:, :n_comp] * S[:n_comp]
        f_pred = x_pred_std @ Vt[:n_comp].T
        return _ols_predict(F, y, f_pred)
    except np.linalg.LinAlgError:
        return _nanmean_fallback(y_train)


def _pls_core(X_std, y, x_pred_std, n_comp):
    """NIPALS PLS simplificado. Retorna (F_scores, t_preds) o None."""
    X_res = X_std.copy()
    y_res = y.copy()
    T_scores = []
    for _ in range(n_comp):
        w = X_res.T @ y_res
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-12:
            break
        w /= norm_w
        t = X_res @ w
        p = X_res.T @ t / (t @ t)
        q = y_res @ t / (t @ t)
        X_res = X_res - np.outer(t, p)
        y_res = y_res - t * q
        T_scores.append(t)
    if not T_scores:
        return None, None
    F = np.column_stack(T_scores)

    X_res2 = X_std.copy()
    X_res_pred = x_pred_std.copy()
    t_preds = []
    for _ in range(n_comp):
        w2 = X_res2.T @ y
        norm_w2 = np.linalg.norm(w2)
        if norm_w2 < 1e-12:
            break
        w2 /= norm_w2
        t2 = X_res2 @ w2
        p2 = X_res2.T @ t2 / (t2 @ t2)
        t_pred_val = float(X_res_pred @ w2)
        t_preds.append(t_pred_val)
        X_res2 = X_res2 - np.outer(t2, p2)
        X_res_pred = X_res_pred - t_pred_val * p2

    return F[:, :len(t_preds)], np.array(t_preds) if t_preds else None


def pls_model(y_train: np.ndarray, X_train: np.ndarray | None, h: int) -> float:
    """
    PLS-1: Factor extraido de predictores ORIGINALES (no filtrados).
    PLS-2: usa esta misma funcion pero el expanding_window pasa X_filtered.
    """
    X, y, x_pred = _prepare_macro_matrix(y_train, X_train, h)
    if X is None:
        return _nanmean_fallback(y_train)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X_std = (X - mu) / sd
    x_pred_std = (x_pred - mu) / sd
    n_comp = min(PLS_COMPONENTS, X_std.shape[1])
    try:
        F, f_pred = _pls_core(X_std, y, x_pred_std, n_comp)
        if F is None or f_pred is None:
            return _nanmean_fallback(y_train)
        return _ols_predict(F, y, f_pred)
    except np.linalg.LinAlgError:
        return _nanmean_fallback(y_train)


# ---------------------------------------------------------------------------
# Penalizados
# ---------------------------------------------------------------------------

def _penalized_model(y_train, X_train, h, model_class, **kwargs) -> float:
    X, y, x_pred = _prepare_macro_matrix(y_train, X_train, h)
    if X is None:
        return _nanmean_fallback(y_train)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X_std = (X - mu) / sd
    x_pred_std = (x_pred - mu) / sd
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = model_class(**kwargs)
        model.fit(X_std, y)
    return float(model.predict(x_pred_std.reshape(1, -1))[0])


def lasso_model(y_train, X_train, h):
    from sklearn.linear_model import LassoCV
    return _penalized_model(y_train, X_train, h, LassoCV,
                            n_alphas=20, cv=3, max_iter=500, tol=1e-2, selection="random")


def elasticnet_model(y_train, X_train, h):
    from sklearn.linear_model import ElasticNetCV
    return _penalized_model(y_train, X_train, h, ElasticNetCV,
                            n_alphas=20, cv=3, max_iter=500, tol=1e-2,
                            selection="random", l1_ratio=[0.1, 0.5, 0.9])


def ridge_model(y_train, X_train, h):
    from sklearn.linear_model import RidgeCV
    return _penalized_model(y_train, X_train, h, RidgeCV)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_model_registry(macro_cols: list[str]) -> list[dict]:
    """
    Construye el registro de 38 modelos para el SOC de EE.UU.

    Retorna lista de dicts:
      name       : str
      fn         : callable(y_train, X_train, h) -> float
      x_cols     : list[str] | None
      use_filtered_x : bool  (True => expanding_window usa X filtrado por frecuencia)
    """
    registry = []

    # AR (no usan macro)
    registry.append({"name": "AR_AIC", "fn": ar_aic, "x_cols": None, "use_filtered_x": False})
    registry.append({"name": "AR_SIC", "fn": ar_sic, "x_cols": None, "use_filtered_x": False})

    # Phillips Curve (3 predictores cada una)
    for pc_key, pc_cols in PC_COLS.items():
        available_pc = [c for c in pc_cols if c in macro_cols]
        if len(available_pc) >= 2:
            registry.append({
                "name": pc_key,
                "fn": make_pc_model(pc_key),
                "x_cols": available_pc,
                "use_filtered_x": False,
            })

    # Bivariate (un predictor por columna macro)
    for col in macro_cols:
        registry.append({
            "name": f"BIV_{col}",
            "fn": make_bivariate(col),
            "x_cols": [col],
            "use_filtered_x": False,
        })

    # PCA y PLS-1 (todos los predictores macro)
    if len(macro_cols) >= 2:
        registry.append({"name": "PCA",  "fn": pca_model,  "x_cols": macro_cols, "use_filtered_x": False})
        registry.append({"name": "PLS1", "fn": pls_model,  "x_cols": macro_cols, "use_filtered_x": False})
        # PLS-2: misma funcion, pero expanding_window provee X filtrado por frecuencia
        registry.append({"name": "PLS2", "fn": pls_model,  "x_cols": macro_cols, "use_filtered_x": True})

    # Penalizados
    if macro_cols:
        registry.append({"name": "LASSO",      "fn": lasso_model,      "x_cols": macro_cols, "use_filtered_x": False})
        registry.append({"name": "ELASTICNET", "fn": elasticnet_model, "x_cols": macro_cols, "use_filtered_x": False})
        registry.append({"name": "RIDGE",      "fn": ridge_model,      "x_cols": macro_cols, "use_filtered_x": False})

    return registry
