"""
models.py – 37 modelos de pronóstico para componentes wavelet.

Interfaz uniforme: fn(y_train, X_train, h) -> float
  - y_train: np.ndarray (T,)  – variable dependiente (componente wavelet)
  - X_train: np.ndarray (T,k) – predictores macro (puede ser None)
  - h: int – horizonte de pronóstico

Todos retornan un único float: el pronóstico para t+h.

Todos los OLS se implementan con numpy.linalg.lstsq.
"""
import warnings
import numpy as np
from typing import Callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_const(X: np.ndarray) -> np.ndarray:
    """Agrega columna de unos."""
    return np.column_stack([np.ones(len(X)), X])


def _ols_predict(X_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray) -> float:
    """OLS via lstsq; retorna pronóstico."""
    X_c = _add_const(X_train)
    x_c = np.concatenate([[1.0], x_pred])
    beta, _, _, _ = np.linalg.lstsq(X_c, y_train, rcond=None)
    return float(x_c @ beta)


def _lag_matrix(y: np.ndarray, lags: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye matriz de regresores rezagados para pronóstico directo.
    X_{t} = [y_{t-h}, y_{t-h-1}, ..., y_{t-h-lags+1}]
    y_{t} = y_t

    Retorna (X, y) donde y es el target a pronosticar.
    """
    n = len(y)
    rows = []
    targets = []
    min_idx = lags + h - 1
    for t in range(min_idx, n):
        row = [y[t - h - lag] for lag in range(lags)]
        rows.append(row)
        targets.append(y[t])
    if not rows:
        return np.empty((0, lags)), np.empty(0)
    return np.array(rows), np.array(targets)


def _select_ar_lags_ic(y: np.ndarray, h: int, max_lags: int, ic: str = "aic") -> int:
    """Selecciona lags AR por AIC o BIC (SIC)."""
    best_ic = np.inf
    best_p = 1
    n = len(y)
    for p in range(1, max_lags + 1):
        X, target = _lag_matrix(y, p, h)
        if len(target) < p + 2:
            break
        X_c = _add_const(X)
        beta, residuals, rank, _ = np.linalg.lstsq(X_c, target, rcond=None)
        if len(residuals) == 0:
            fitted = X_c @ beta
            residuals_vec = target - fitted
            ssr = float(residuals_vec @ residuals_vec)
        else:
            ssr = float(residuals[0]) if hasattr(residuals, "__len__") else float(residuals)
        k = p + 1  # lags + constante
        T = len(target)
        if ssr <= 0:
            continue
        log_lik = -T / 2 * np.log(ssr / T)
        if ic == "aic":
            val = -2 * log_lik + 2 * k
        else:  # sic/bic
            val = -2 * log_lik + k * np.log(T)
        if val < best_ic:
            best_ic = val
            best_p = p
    return best_p


# ---------------------------------------------------------------------------
# Modelos AR
# ---------------------------------------------------------------------------

def ar_aic(y_train: np.ndarray, X_train, h: int) -> float:
    """AR con selección de lags por AIC."""
    from .config import AR_MAX_LAGS
    p = _select_ar_lags_ic(y_train, h, AR_MAX_LAGS, "aic")
    X, target = _lag_matrix(y_train, p, h)
    if len(target) < 2:
        return float(y_train[-1])
    X_c = _add_const(X)
    beta, _, _, _ = np.linalg.lstsq(X_c, target, rcond=None)
    # Pronóstico: usar últimos valores observados
    last_row = np.array([y_train[-h - lag] for lag in range(p)])
    return float(np.concatenate([[1.0], last_row]) @ beta)


def ar_sic(y_train: np.ndarray, X_train, h: int) -> float:
    """AR con selección de lags por SIC (BIC)."""
    from .config import AR_MAX_LAGS
    p = _select_ar_lags_ic(y_train, h, AR_MAX_LAGS, "sic")
    X, target = _lag_matrix(y_train, p, h)
    if len(target) < 2:
        return float(y_train[-1])
    X_c = _add_const(X)
    beta, _, _, _ = np.linalg.lstsq(X_c, target, rcond=None)
    last_row = np.array([y_train[-h - lag] for lag in range(p)])
    return float(np.concatenate([[1.0], last_row]) @ beta)


# ---------------------------------------------------------------------------
# Phillips Curve (1 predictor + AR(1))
# ---------------------------------------------------------------------------

def _pc_model(y_train: np.ndarray, x_col: np.ndarray, h: int) -> float:
    """Phillips Curve: y_{t+h} = a + b*y_{t} + c*x_{t}"""
    n = len(y_train)
    if len(x_col) < n:
        x_col = np.pad(x_col, (n - len(x_col), 0), constant_values=np.nan)
    elif len(x_col) > n:
        x_col = x_col[-n:]

    min_idx = h
    X_rows, targets = [], []
    for t in range(min_idx, n):
        if t < 1:
            continue
        if np.isnan(y_train[t - 1]) or np.isnan(x_col[t - h]):
            continue
        X_rows.append([y_train[t - 1], x_col[t - h]])
        targets.append(y_train[t])

    if len(targets) < 3:
        return float(np.nanmean(y_train[-12:]))

    X_arr = np.array(X_rows)
    y_arr = np.array(targets)
    X_c = _add_const(X_arr)
    beta, _, _, _ = np.linalg.lstsq(X_c, y_arr, rcond=None)
    x_last = x_col[-1] if not np.isnan(x_col[-1]) else np.nanmean(x_col[-12:])
    pred_row = np.array([1.0, y_train[-1], x_last])
    return float(pred_row @ beta)


def make_pc_model(col_name: str) -> Callable:
    """Crea modelo Phillips Curve con predictor `col_name`."""
    def pc_fn(y_train: np.ndarray, X_train: np.ndarray | None, h: int) -> float:
        if X_train is None or X_train.shape[1] == 0:
            return float(np.nanmean(y_train[-12:]))
        # X_train es un DataFrame alineado externamente; aquí es ndarray
        # La columna se pasa por índice en el registry
        return _pc_model(y_train, X_train[:, 0], h)
    pc_fn.__name__ = f"PC_{col_name}"
    return pc_fn


# ---------------------------------------------------------------------------
# Bivariate OLS (1 predictor externo)
# ---------------------------------------------------------------------------

def make_bivariate(col_name: str) -> Callable:
    """AR(1) + predictor externo (bivariate)."""
    def biv_fn(y_train: np.ndarray, X_train: np.ndarray | None, h: int) -> float:
        if X_train is None or X_train.shape[1] == 0:
            return float(np.nanmean(y_train[-12:]))
        n = len(y_train)
        x_col = X_train[:, 0]

        min_idx = max(h, 1)
        X_rows, targets = [], []
        for t in range(min_idx, n):
            if np.isnan(y_train[t - 1]) or np.isnan(x_col[t - h] if t - h >= 0 else np.nan):
                continue
            x_val = x_col[t - h] if t - h >= 0 else np.nan
            if np.isnan(x_val):
                continue
            X_rows.append([y_train[t - 1], x_val])
            targets.append(y_train[t])

        if len(targets) < 3:
            return float(np.nanmean(y_train[-12:]))

        X_arr = np.array(X_rows)
        y_arr = np.array(targets)
        X_c = _add_const(X_arr)
        beta, _, _, _ = np.linalg.lstsq(X_c, y_arr, rcond=None)
        x_last = x_col[-1] if not np.isnan(x_col[-1]) else np.nanmean(x_col)
        pred_row = np.array([1.0, y_train[-1], x_last])
        return float(pred_row @ beta)
    biv_fn.__name__ = f"BIV_{col_name}"
    return biv_fn


# ---------------------------------------------------------------------------
# Reducción de dimensión
# ---------------------------------------------------------------------------

def _prepare_macro_matrix(y_train: np.ndarray, X_train: np.ndarray, h: int):
    """Prepara X, y para regresión con horizonte h."""
    n = len(y_train)
    if X_train is None or X_train.shape[1] == 0:
        return None, None, None
    # Alinear: y_t predice y_{t+h}
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
    # X_pred: últimos valores disponibles de X
    x_pred_raw = X_al[-1]
    if np.any(np.isnan(x_pred_raw)):
        # Imputar con media columna
        col_means = np.nanmean(X_al, axis=0)
        x_pred_raw = np.where(np.isnan(x_pred_raw), col_means, x_pred_raw)
    return X_out, y_out, x_pred_raw


def pca_model(y_train: np.ndarray, X_train: np.ndarray | None, h: int) -> float:
    """PCA (3 componentes) + OLS."""
    from .config import PCA_COMPONENTS
    X, y, x_pred = _prepare_macro_matrix(y_train, X_train, h)
    if X is None:
        return float(np.nanmean(y_train[-12:]))

    # Estandarizar
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X_std = (X - mu) / sd
    x_pred_std = (x_pred - mu) / sd

    # PCA via SVD
    n_comp = min(PCA_COMPONENTS, X_std.shape[1], X_std.shape[0] - 1)
    try:
        U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
        F = U[:, :n_comp] * S[:n_comp]  # scores
        f_pred = x_pred_std @ Vt[:n_comp].T
        return _ols_predict(F, y, f_pred)
    except np.linalg.LinAlgError:
        return float(np.nanmean(y_train[-12:]))


def pls_model(y_train: np.ndarray, X_train: np.ndarray | None, h: int) -> float:
    """PLS (2 componentes)."""
    from .config import PLS_COMPONENTS
    X, y, x_pred = _prepare_macro_matrix(y_train, X_train, h)
    if X is None:
        return float(np.nanmean(y_train[-12:]))

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X_std = (X - mu) / sd
    x_pred_std = (x_pred - mu) / sd

    n_comp = min(PLS_COMPONENTS, X_std.shape[1])
    try:
        # NIPALS simplificado
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
            return float(np.nanmean(y_train[-12:]))

        F = np.column_stack(T_scores)
        # Recalcular loadings para predicción
        # Proyectar x_pred sobre los componentes
        X_res_pred = x_pred_std.copy()
        t_preds = []
        X_res2 = X_std.copy()
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

        if not t_preds:
            return float(np.nanmean(y_train[-12:]))

        f_pred_arr = np.array(t_preds)
        return _ols_predict(F[:, :len(t_preds)], y, f_pred_arr)
    except np.linalg.LinAlgError:
        return float(np.nanmean(y_train[-12:]))


# ---------------------------------------------------------------------------
# Penalizados (LASSO, ElasticNet, Ridge)
# ---------------------------------------------------------------------------

def _penalized_model(y_train, X_train, h, model_class, **kwargs) -> float:
    X, y, x_pred = _prepare_macro_matrix(y_train, X_train, h)
    if X is None:
        return float(np.nanmean(y_train[-12:]))
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


def lasso_model(y_train: np.ndarray, X_train: np.ndarray | None, h: int) -> float:
    from sklearn.linear_model import LassoCV
    return _penalized_model(
        y_train, X_train, h, LassoCV,
        n_alphas=20,       # 100 → 20: 5x menos puntos en el grid de alpha
        cv=3,              # 5 → 3 folds: 40% menos fits
        max_iter=500,      # convergencia más agresiva
        tol=1e-2,          # tolerancia más laxa (suficiente para pronostico)
        selection="random", # coordenadas aleatorias: converge más rápido
    )


def elasticnet_model(y_train: np.ndarray, X_train: np.ndarray | None, h: int) -> float:
    from sklearn.linear_model import ElasticNetCV
    return _penalized_model(
        y_train, X_train, h, ElasticNetCV,
        n_alphas=20,
        cv=3,
        max_iter=500,
        tol=1e-2,
        selection="random",
        l1_ratio=[0.1, 0.5, 0.9],  # explorar mezclas LASSO/Ridge
    )


def ridge_model(y_train: np.ndarray, X_train: np.ndarray | None, h: int) -> float:
    from sklearn.linear_model import RidgeCV
    # RidgeCV usa SVD: siempre rapido, sin cambios necesarios
    return _penalized_model(y_train, X_train, h, RidgeCV)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_model_registry(macro_cols: list[str]) -> list[dict]:
    """
    Construye lista dinámica de modelos según predictores disponibles.

    Retorna lista de dicts:
      {
        'name': str,
        'fn': callable(y_train, X_train, h) -> float,
        'x_cols': list[str] | None,  # columnas de X_macro a usar
      }
    """
    registry = []

    # AR models (no usan macro)
    registry.append({"name": "AR_AIC", "fn": ar_aic, "x_cols": None})
    registry.append({"name": "AR_SIC", "fn": ar_sic, "x_cols": None})

    # Phillips Curve (3 predictores clave)
    for pc_col, pc_label in [("CETES28", "PC1"), ("USDMXN", "PC2"), ("M2_REAL", "PC3")]:
        if pc_col in macro_cols:
            registry.append({
                "name": pc_label,
                "fn": make_pc_model(pc_col),
                "x_cols": [pc_col],
            })

    # Bivariate (1 predictor por columna macro)
    for col in macro_cols:
        registry.append({
            "name": f"BIV_{col}",
            "fn": make_bivariate(col),
            "x_cols": [col],
        })

    # Reducción de dimensión (usan todas las columnas macro)
    if len(macro_cols) >= 2:
        registry.append({"name": "PCA", "fn": pca_model, "x_cols": macro_cols})
        registry.append({"name": "PLS", "fn": pls_model, "x_cols": macro_cols})

    # Penalizados
    if macro_cols:
        registry.append({"name": "LASSO",      "fn": lasso_model,      "x_cols": macro_cols})
        registry.append({"name": "ELASTICNET", "fn": elasticnet_model, "x_cols": macro_cols})
        registry.append({"name": "RIDGE",      "fn": ridge_model,      "x_cols": macro_cols})

    return registry
