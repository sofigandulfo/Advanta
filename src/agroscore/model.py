"""Modelo de rinde, validación por campaña y persistencia.

Validación: leave-one-campaign-out. En cada pliegue se entrena con todas las campañas
menos una y se evalúa en la campaña excluida. Nunca se usa división aleatoria, porque
lotes de una misma campaña comparten clima y eso inflaría las métricas.
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agroscore.features import VARIABLES_CLIMA, VARIABLES_CURVA

VARIABLES_NUMERICAS = VARIABLES_CURVA + VARIABLES_CLIMA
VARIABLES_CATEGORICAS = ["cultivo"]
OBJETIVO = "rinde_kg_ha"
TIPOS = ("baseline", "ridge", "gboost")


class BaselinePorCultivo(BaseEstimator, RegressorMixin):
    """Predice el rinde medio de cada cultivo en entrenamiento. Es la vara mínima a superar."""

    def fit(self, X, y):
        self.medias_ = pd.Series(np.asarray(y), index=X["cultivo"].to_numpy()).groupby(level=0).mean().to_dict()
        self.media_global_ = float(np.mean(y))
        return self

    def predict(self, X):
        return X["cultivo"].map(self.medias_).fillna(self.media_global_).to_numpy(dtype=float)


def crear_modelo(tipo: str):
    if tipo == "baseline":
        return BaselinePorCultivo()
    numericas = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categoricas = OneHotEncoder(handle_unknown="ignore")
    prep = ColumnTransformer([("num", numericas, VARIABLES_NUMERICAS), ("cat", categoricas, VARIABLES_CATEGORICAS)])
    if tipo == "ridge":
        return Pipeline([("prep", prep), ("modelo", RidgeCV(alphas=np.logspace(-2, 3, 20)))])
    if tipo == "gboost":
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=10,
                                            l2_regularization=1.0, random_state=0)
        return Pipeline([("prep", prep), ("modelo", reg)])
    raise ValueError(f"Tipo de modelo desconocido: {tipo}. Opciones: {TIPOS}")


def metricas(y_real, y_pred) -> dict:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_real)),
        "mae_kg_ha": float(mean_absolute_error(y_real, y_pred)),
        "rmse_kg_ha": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "r2": float(r2_score(y_real, y_pred)) if len(y_real) > 1 else np.nan,
        "mape_pct": float(np.mean(np.abs((y_real - y_pred) / y_real)) * 100),
        "sesgo_kg_ha": float(np.mean(y_pred - y_real)),
    }


def pliegues_por_campania(df: pd.DataFrame):
    """Genera (campaña_test, índices_train, índices_test)."""
    campanias = df["campania"].to_numpy()
    if len(np.unique(campanias)) < 2:
        raise ValueError("Se necesitan al menos 2 campañas con rinde conocido para validar por campaña")
    for train, test in LeaveOneGroupOut().split(df, groups=campanias):
        yield campanias[test[0]], train, test


def validar_por_campania(df: pd.DataFrame, tipo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predicciones fuera de muestra y métricas por campaña excluida."""
    X = df[VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS]
    y = df[OBJETIVO].to_numpy()
    pred = np.full(len(df), np.nan)
    filas = []
    for campania, train, test in pliegues_por_campania(df):
        modelo = crear_modelo(tipo).fit(X.iloc[train], y[train])
        pred[test] = modelo.predict(X.iloc[test])
        filas.append({"campania_test": campania, "modelo": tipo} | metricas(y[test], pred[test]))
    oof = df[["lote_id", "campania", "cultivo", OBJETIVO]].copy()
    oof["rinde_validacion_kg_ha"] = pred
    oof["modelo"] = tipo
    return oof, pd.DataFrame(filas)


def entrenar(df: pd.DataFrame, tipos=TIPOS) -> dict:
    """Valida cada tipo por campaña, elige el de menor MAE fuera de muestra y lo reentrena con todo.

    Devuelve un 'paquete' con el modelo final, métricas y predicciones de validación.
    Ojo: elegir el modelo con la misma validación con la que se reporta es levemente optimista.
    Con datos reales, reservar además una campaña final que no se mire hasta el final.
    """
    datos = df[df[OBJETIVO].notna()].reset_index(drop=True)
    if len(datos) < 20:
        raise ValueError(f"Muy pocos lotes con rinde conocido ({len(datos)}); se necesitan al menos 20")

    resultados = {}
    for tipo in tipos:
        oof, por_campania = validar_por_campania(datos, tipo)
        global_ = metricas(oof[OBJETIVO], oof["rinde_validacion_kg_ha"])
        por_cultivo = {c: metricas(g[OBJETIVO], g["rinde_validacion_kg_ha"]) for c, g in oof.groupby("cultivo")}
        resultados[tipo] = {"oof": oof, "por_campania": por_campania, "global": global_, "por_cultivo": por_cultivo}

    mejor = min(resultados, key=lambda t: resultados[t]["global"]["mae_kg_ha"])
    final = crear_modelo(mejor).fit(datos[VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS], datos[OBJETIVO])
    return {
        "modelo": final,
        "tipo": mejor,
        "variables": VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS,
        "rmse_por_cultivo": {c: m["rmse_kg_ha"] for c, m in resultados[mejor]["por_cultivo"].items()},
        "metricas_globales": {t: r["global"] for t, r in resultados.items()},
        "metricas_por_cultivo": {t: r["por_cultivo"] for t, r in resultados.items()},
        "metricas_por_campania": pd.concat([r["por_campania"] for r in resultados.values()], ignore_index=True),
        "predicciones_validacion": resultados[mejor]["oof"],
        "campanias": sorted(datos["campania"].unique().tolist()),
        "cultivos": sorted(datos["cultivo"].unique().tolist()),
        "n_lotes": int(len(datos)),
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def predecir(paquete: dict, df: pd.DataFrame) -> np.ndarray:
    return paquete["modelo"].predict(df[paquete["variables"]])


def guardar(paquete: dict, ruta) -> None:
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquete, ruta)


def cargar(ruta) -> dict:
    return joblib.load(ruta)
